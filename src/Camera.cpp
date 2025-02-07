#include "Camera.h"

Cameraa::Cameraa(cs::UsbCamera *camRef, cs::VideoMode config, AprilTagDetector *detRef, AprilTagPoseEstimator *estRef) {
  cam = camRef;
  detector = detRef;
  estimator = estRef;
  auto info = cam->GetInfo();
  id = info.dev;
  sink = new cs::CvSink{frc::CameraServer::GetVideo(*cam)};
  source = new cs::CvSource{"source" + id, config};
  frc::CameraServer::StartAutomaticCapture(*source);

  collector = std::move(std::thread(&Cameraa::StartCollector, this));
  converter = std::move(std::thread(&Cameraa::StartGrayscaleConverter, this));
  processor = std::move(std::thread(&Cameraa::StartProcessor, this));
  labeller = std::move(std::thread(&Cameraa::StartLabeller, this));
  poster = std::move(std::thread(&Cameraa::StartPosting, this));
}

std::vector<uint8_t> Cameraa::GetTargetTags() {
  return targetTags;
}

void Cameraa::SetTargetTags(std::vector<uint8_t> targets) {
  targetTags = targets;
}

// Draw AprilTag outline onto provided frame
void Cameraa::DrawAprilTagBox(cv::Mat frame, const frc::AprilTagDetection* tag) {
  // Draw boxes around tags for video feed                
  for(int i = 0; i < 4; i++) {
      auto point1 = tag->GetCorner(i);
      int secondIndex = i == 3 ? 0 : i + 1;   // out of bounds adjust for last iteration
      auto point2 = tag->GetCorner(secondIndex);
      cv::Point lineStart{(int)point1.x, (int)point1.y};
      cv::Point lineEnd{(int)point2.x, (int)point2.y};
      cv::line(frame, lineStart, lineEnd, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
  }
}

// Draw ML inference outlines onto provided frame
void Cameraa::DrawInferenceBox(cv::Mat frame, std::vector<Detection> &detections) {
  for (auto& detection : detections) {
      cv::Rect rect(detection.x, detection.y, detection.width, detection.height);
      auto color = cv::Scalar((detection.label == 0) * 255, (detection.label == 1) * 255, (detection.label == 2) * 255);
      cv::rectangle(frame, rect, color, 2, cv::LINE_4);
      for(int i = 0; i < detection.kps.size(); i += 3) {
        cv::Point center(detection.kps[i], detection.kps[i+1]);
        cv::circle(frame, center, detection.kps[i+2]*4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
      }
  }
}

bool Cameraa::ValidPresent() {
  return newFrame && validFrame;
}

void Cameraa::StartCollector() {
  while(true) {
    if(!newFrame) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      return;
    }
    // Get the current time from the system clock
    auto now = std::chrono::system_clock::now();

    // Convert the current time to time since epoch
    auto duration = now.time_since_epoch();
    unsigned long milliseconds
      = std::chrono::duration_cast<std::chrono::milliseconds>(
      duration).count();
      
    if(lastFail && milliseconds - lastFail < 3000) {
      continue;
    }
    auto success = sink->GrabFrame(frame);
    if(!success) {
      lastFail = milliseconds;
    } else {
      lastFail = 0;
    }
    validFrame = !lastFail && !frame.empty();
    if(validFrame) {
      newFrame = true;
      grayAvailable = false;
      frameLabelled = false;
      frameProcessed = false;
    }
  }
}

void Cameraa::StartGrayscaleConverter() {
  while(true) {
    if(!ValidPresent()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      return;
    }
    if(!grayAvailable) {
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      grayAvailable = true;
    }
  }
}

void Cameraa::StartProcessor() {
  while(true) {
    if(!ValidPresent() && grayAvailable) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      return;
    }
    if(!frameProcessed) {
      aprilTags = frc::AprilTagDetect(*detector, gray);
      for(const frc::AprilTagDetection* tag : aprilTags) {
        uint8_t id = tag->GetId();
        /*std::cout << "ID: " << (int)id << " found" << std::endl;*/
        uint8_t found = count(targetTags.begin(), targetTags.end(), id);
        if(!found) continue;  // tag not in request array, skip
        auto transform = estimator->Estimate(*tag);  // Estimate Transform3d relative t  
      }
      frameProcessed = true;
    }
  }
}

void Cameraa::StartInferencing(struct sockaddr_in *server_addr) {
  int sock = getSocket(server_addr);
  while(true) {
    if(!ValidPresent()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      return;
    }
    detections = remoteInference(sock, server_addr, frame);
  }
}

void Cameraa::StartLabeller() {
  while(true) {
    if(!ValidPresent() || (!frameProcessed && !detections.size())) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      return;
    }
    
    for(const frc::AprilTagDetection* tag : aprilTags) {
      DrawAprilTagBox(labelled, tag);
    }
    DrawInferenceBox(labelled, detections);
    frameLabelled = true;
  }
}

void Cameraa::StartPosting() {
  while(true) {
    if(!ValidPresent() || !frameLabelled) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      return;
    }
    source->PutFrame(labelled);
    newFrame = false;
  }
}
