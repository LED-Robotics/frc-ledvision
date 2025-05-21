#include "Camera.hpp"

Camera::Camera(cs::UsbCamera *camRef, cs::VideoMode config, AprilTagPoseEstimator::Config estConfig) 
  : estimator{estConfig} {
  cam = camRef;
  // Configure AprilTag detector
  detector.AddFamily("tag36h11");
  detector.SetConfig({});
  auto quadParams = detector.GetQuadThresholdParameters();
  quadParams.minClusterPixels = 3;
  detector.SetQuadThresholdParameters(quadParams);

  auto info = cam->GetInfo();
  id = info.dev;
  sink = new cs::CvSink{frc::CameraServer::GetVideo(*cam)};
  /*source = new cs::CvSource{"source" + id, config};*/
  cam->SetVideoMode(config);
  source = frc::CameraServer::PutVideo(std::string_view("source" + std::to_string(id)), 160, 120);
  /*frc::CameraServer::StartAutomaticCapture(*source);*/

  /*boxLabelVector = &boxDets1;*/
  /*inactiveBoxLabelVector = &boxDets2;*/
  /*poseLabelVector = &poseDets1;*/
  /*inactivePoseLabelVector = &poseDets2;*/
}

uint8_t Camera::GetID() {
  return id;
}

std::vector<uint8_t> Camera::GetTargetTags() {
  return targetTags;
}

void Camera::SetTargetTags(std::vector<uint8_t> targets) {
  targetTags = targets;
}

std::vector<Camera::TagDetection>* Camera::GetTagDetections() {
  return &tagDetections;
}

int Camera::GetTagDetectionCount() {
  return tagDetectionCount;
}

// Get current box ML Detection vector from Camera
std::vector<det::DetectObject>* Camera::GetBoxDetections() {
  return boxLabelVector;
}

// Get current box ML Detection vector from Camera
std::vector<det::PoseObject>* Camera::GetPoseDetections() {
  return poseLabelVector;
}

// Get current box ML Detection vector from Camera
std::vector<det::DetectObject>* Camera::GetInactiveBoxDetections() {
  return inactiveBoxLabelVector;
}

// Get current box ML Detection vector from Camera
std::vector<det::PoseObject>* Camera::GetInactivePoseDetections() {
  return inactivePoseLabelVector;
}

// Prevent buffer swapping
void Camera::FreezeMLBufs() {
  mlBufsFrozen = true;
}

// Allow buffer swapping
void Camera::UnfreezeMLBufs() {
  mlBufsFrozen = false;
}

// Switch toggle active buffer for reading
void Camera::SwitchActiveMLBuf() {
  if(mlBufsFrozen) return;
  if(mlMode == MLMode::Detect) {
    boxLabelVector = boxLabelVector == detVector1 ? detVector2 : detVector1;
    inactiveBoxLabelVector = inactiveBoxLabelVector == detVector1 ? detVector2 : detVector1;
  } else if (mlMode == MLMode::Pose) {
    poseLabelVector = poseLabelVector == poseVector1 ? poseVector2 : poseVector1;
    inactivePoseLabelVector = inactivePoseLabelVector == poseVector1 ? poseVector2 : poseVector1;
  }
}

// Get current ML detection mode
int Camera::GetMLDetectionMode() {
  return mlMode;
}

// Get current ML detection mode
int Camera::GetMLEnabled() {
  return mlEnabled;
}

// Set current ML detection mode
void Camera::SetMLDetectionMode(int mode) {
  mlMode = mode;
}

int Camera::GetMLDetectionCount() {
  if(mlMode == MLMode::Detect) {
    return boxLabelVector->size();
  } else if (mlMode == MLMode::Pose) {
    return poseLabelVector->size();
  } else {
    return 0;
  }
}

uint32_t Camera::GetCaptureTime() {
  return captureTime;
}

// Stop overwriting the tag detection buffer
void Camera::PauseTagDetection() {
  pauseTagDetections = true;
}

// Resume overwriting the tag detection buffer
void Camera::ResumeTagDetection() {
  pauseTagDetections = false;
}

// Draw AprilTag outline onto provided frame
void Camera::DrawAprilTagBox(cv::Mat frame, TagDetection* tag) {
  // Draw boxes around tags for video feed                
  for(int i = 0; i < 4; i++) {
      auto point1 = tag->corners[i];
      int secondIndex = i == 3 ? 0 : i + 1;   // out of bounds adjust for last iteration
      auto point2 = tag->corners[secondIndex];
      cv::Point lineStart{(int)point1.x, (int)point1.y};
      cv::Point lineEnd{(int)point2.x, (int)point2.y};
      cv::line(frame, lineStart, lineEnd, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
  }
}

// Draw ML detection on frame
void Camera::DrawDetectBox(cv::Mat frame, det::DetectObject &detection) {
  auto color = cv::Scalar((detection.label == 0) * 255, (detection.label == 1) * 255, (detection.label == 2) * 255);
  cv::rectangle(frame, detection.rect, color, 2, cv::LINE_4);
}

// Draw ML detection on frame
void Camera::DrawPoseBox(cv::Mat frame, det::PoseObject &detection) {
  auto color = cv::Scalar((detection.label == 0) * 255, (detection.label == 1) * 255, (detection.label == 2) * 255);
  cv::rectangle(frame, detection.rect, color, 2, cv::LINE_4);
  for(int i = 0; i < detection.kps.size(); i += 3) {
    cv::Point center(detection.kps[i], detection.kps[i+1]);
    cv::circle(frame, center, detection.kps[i+2]*4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
  }
}

bool Camera::ValidPresent() {
  return newFrame && validFrame;
}

// Check if ML frame is ready
bool Camera::IsMLFrameAvailable() {
  return mlFrameAvailable;
}

// Check if ML frame is ready
void Camera::SetMLFrameUnavailable() {
  mlFrameAvailable = false;
}


// Return ML frame for inference
cv::Mat Camera::GetMLFrame() {
  return mlFrame;
}

bool Camera::IsInferencePossible() {
  #ifdef CUDA_PRESENT
  return model != nullptr;
  #else
  return mlSessions.size();
  #endif
}

// Disable ML
void Camera::DisableInference() {
  mlEnabled = false;  
}

// Enable ML (if model is loaded)
void Camera::EnableInference() {
  if(IsInferencePossible()) {
    mlEnabled = true;
  }
}

// Start posting labelled frames
void Camera::DestroyModel() {
  if(IsInferencePossible()) {
    #ifdef CUDA_PRESENT
    delete model;
    model = nullptr;
    #endif
  }
}

// Start posting labelled frames
void Camera::LoadModel(std::string path) {
  #ifdef CUDA_PRESENT
  model = new YOLO11(path);
  model->make_pipe(true);
  #else
  #endif
}

// Run detect inference on frame
void Camera::RunInference(cv::Mat frame, std::vector<det::DetectObject> *dets) {
  #ifdef CUDA_PRESENT
  model->copy_from_Mat(frame);
  model->infer();
  model->detectPostprocess(*dets);
  #else
  #endif
}

// Run pose inference on frame
void Camera::RunInference(cv::Mat frame, std::vector<det::PoseObject> *dets) {
  #ifdef CUDA_PRESENT
  model->copy_from_Mat(frame);
  model->infer();
  model->posePostprocess(*dets);
  #else
  #endif
}

void Camera::StartStream() {
  std::cout << "Starting Capture for Cam " << (int)id << std::endl;
  collector = std::move(std::thread(&Camera::StartCollector, this));
  converter = std::move(std::thread(&Camera::StartGrayscaleConverter, this));
  processor = std::move(std::thread(&Camera::StartProcessor, this));
  labeller = std::move(std::thread(&Camera::StartLabeller, this));
  poster = std::move(std::thread(&Camera::StartPosting, this));
}

void Camera::StartCollector() {
  while(true) {
    if(newFrame && !framePosted) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      continue;
    }
    // Get the current time from the system clock
    auto now = std::chrono::system_clock::now();

    // Convert the current time to time since epoch
    auto duration = now.time_since_epoch();
    unsigned long milliseconds
      = std::chrono::duration_cast<std::chrono::milliseconds>(
      duration).count();
      
    if(lastFail && milliseconds - lastFail > 3000) {
      continue;
    }
    auto success = sink->GrabFrame(frame);
    if(success == 0) {
      lastFail = milliseconds;
    } else {
      lastFail = 0;
    }
    validFrame = !lastFail && !frame.empty();
    if(validFrame) {
      if(recording && !recordingLabelled) {
        outputVideo << frame;
      }
      captureTime = milliseconds + success;
      newFrame = true;
      grayAvailable = false;
      frameLabelled = false;
      frameProcessed = false;
      framePosted = false;
    }
  }
}

void Camera::StartGrayscaleConverter() {
  while(true) {
    if(!ValidPresent()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      continue;
    }
    if(!grayAvailable) {
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      if(!mlFrameAvailable) {
        mlFrame = frame.clone();
        mlFrameAvailable = true;
      }
      labelled = frame.clone();
      grayAvailable = true;
    }
  }
}

void Camera::StartProcessor() {
  while(true) {
    if(!ValidPresent() || !grayAvailable) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      continue;
    }
    if(!frameProcessed) {
      if(!pauseTagDetections) {
        tagDetections.clear();
      }
      auto aprilTags = frc::AprilTagDetect(detector, gray);
      for(const frc::AprilTagDetection* tag : aprilTags) {
        uint8_t id = tag->GetId();
        uint8_t found = count(targetTags.begin(), targetTags.end(), id);
        if(!found) continue;  // tag not in request array, skip
        auto transform = estimator.Estimate(*tag);  // Estimate Transform3d of tag
        std::vector<AprilTagDetection::Point> corners;
        // Generate rectangle for labelling tag 
        for(int i = 0; i < 4; i++) {
            auto point = tag->GetCorner(i);
            corners.push_back(point);
        }
        if(!pauseTagDetections) {
          TagDetection data{id, corners, transform};
          tagDetections.push_back(data);
        }
      }
      tagDetectionCount = tagDetections.size();
      frameProcessed = true;
    }
  }
}

void Camera::InferenceThread() {
  while(true) {
    if(!mlEnabled) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    if(!IsMLFrameAvailable()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    if(GetMLDetectionMode() == MLMode::Detect) {
      RunInference(mlFrame, GetInactiveBoxDetections());
    } else if(GetMLDetectionMode() == MLMode::Pose) {
      RunInference(mlFrame, GetInactivePoseDetections());
    }
    SwitchActiveMLBuf();
    mlFrameAvailable = false;
  }
}

void Camera::StartLabeller() {
  while(true) {
    if(!ValidPresent() || !frameProcessed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      continue;
    }
    for(TagDetection& tag : tagDetections) {
      DrawAprilTagBox(labelled, &tag);
    }
    if(mlEnabled && GetMLDetectionMode() == MLMode::Detect) {
      for(det::DetectObject& det : *boxLabelVector) {
        DrawDetectBox(labelled, det);
      }
    } else if(mlEnabled && GetMLDetectionMode() == MLMode::Pose) {
      for(det::PoseObject& det : *poseLabelVector) {
        DrawPoseBox(labelled, det);
      }
    }
    cv::putText(labelled, //target image
      "ID: " + std::to_string(GetID()), //text
      cv::Point(10, labelled.rows / 8), //top-left position
      cv::FONT_HERSHEY_DUPLEX,
      2.0,
      CV_RGB(255, 255, 255), //font color
      2);
    frameLabelled = true;
  }
}

void Camera::StartPosting() {
  while(true) {
    if(!ValidPresent() || !frameLabelled) {
      std::this_thread::sleep_for(std::chrono::milliseconds(threadDelay));
      continue;
    }
    if(recording && recordingLabelled) outputVideo << labelled;
    cv::Mat resized;
    cv::resize(labelled, resized, cv::Size(160, 120));
    source.PutFrame(resized);
    newFrame = false;
    frameProcessed = true;
  }
}

void Camera::StartInferencing(std::string path) {
  LoadModel(path);
  mlThread = std::move(std::thread(&Camera::InferenceThread, this));
}

bool Camera::StartRecording(std::string path, bool labelled) {
  recordState = true;
  auto config = cam->GetVideoMode();
  if(!recording && recordState) {
    recordingLabelled = labelled;
    outputVideo.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 30, {config.width, config.height});
    recording = outputVideo.isOpened();
    return recording;
  }
  return true;
}

bool Camera::StopRecording() {
  recordState = false;
  if(!recordState && recording) {
    outputVideo.release();
    recording = outputVideo.isOpened();
    return recording;
  }
  return true;
}
