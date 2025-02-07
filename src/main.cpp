#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <apriltag/frc/apriltag/AprilTagDetector.h>
#include <apriltag/frc/apriltag/AprilTagDetector_cv.h>
#include <apriltag/frc/apriltag/AprilTagPoseEstimator.h>
#include <apriltag/frc/apriltag/AprilTagFieldLayout.h>
#include <apriltag/frc/apriltag/AprilTagFields.h>
#include <cameraserver/CameraServer.h>
#include <units/length.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <linux/in.h>
#include <sys/socket.h>
#include <sys/select.h>


#include "networking.h"
#include "Camera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> 

using namespace frc;

// Camera resolution/format configs
int width = 640;
int height = 640;
cs::VideoMode camConfig{cs::VideoMode::PixelFormat::kMJPEG, width, height, 30};

// To store IDs of current valid cameras
std::vector<uint8_t> currentCams;

// Variables for sending AprilTag detections
std::vector<uint8_t> targetTags;
uint8_t targetCount = -1;
uint8_t *tagBuffer;
uint32_t tagBufSize = 0;

// Container for individual camera data
struct Camera {
  uint8_t id = -1;
  cs::UsbCamera* ref = nullptr;
  cs::CvSink* sink = nullptr;
  cs::CvSource* source = nullptr;
  cv::Mat frame{};
  cv::Mat gray{};
  cv::Mat labelledFrame{};
  bool newFrame = false;
  bool newInference = false;
  std::vector<Detection> detections;
  uint32_t captureTime = 0;
  unsigned long lastFail = 0;
  bool validData = false;
};

std::vector<Camera> cameras; // Global camera references

// Struct format for AprilTag detection
struct AprilTagFrame {
  uint8_t tagId = -1;
  uint8_t camId = -1;
  uint32_t timeCaptured;
  double tx;
  double ty;
  double tz;
  double rx;
  double ry;
  double rz;
};

const size_t TAG_FRAME_SIZE = sizeof(AprilTagFrame);

// Global data to send in the AprilTag frame
struct GlobalFrame {
  uint8_t size[2];
};

// Machine Learning inference variables
int inferTarget = -1;

// AprilTag detection objects
AprilTagDetector detector{};
AprilTagPoseEstimator estimator{{6.5_in, (double)640, (double)480, (double)320, (double)240}};  // dummy numbers



// Configure AprilTag detection parameters
void initAprilTagDetector() {
    // Configure AprilTag detector
    detector.AddFamily("tag36h11");
    detector.SetConfig({});
    auto quadParams = detector.GetQuadThresholdParameters();
    quadParams.minClusterPixels = 3;
    detector.SetQuadThresholdParameters(quadParams);
}

// Print coordinates Transform3d
void debugTagPrint(int id, Transform3d transform) {
    std::cout << "Tag " << id << " Pose Estimation:" << std::endl;
    std::cout << "X Off: " << transform.X().value();
    std::cout << " Y Off: " << transform.Y().value();
    std::cout << " Z Off: " << transform.Z().value() << std::endl;
    std::cout << "Rot Off: " << transform.Rotation().ToRotation2d().Degrees().value() << std::endl;
    std::cout << std::endl;
}

// Init and return all cameras plugged in
std::vector<cs::UsbCamera> initCameras(cs::VideoMode config) {
    CS_Status status = 0;
    std::vector<cs::UsbCamera> cameras{};
    for (const auto& caminfo : cs::EnumerateUsbCameras(&status)) {
        fmt::print("Dev {}: Path {} (Name {})\n", caminfo.dev, caminfo.path, caminfo.name);
        fmt::print("vid {}: pid {}\n", caminfo.vendorId, caminfo.productId);
        cs::UsbCamera cam{"camera-" + caminfo.dev, caminfo.path};
        cam.SetVideoMode(config);
        cameras.push_back(cam);
    }
    return cameras;
}

// Draw AprilTag outline onto provided frame
void drawAprilTagBox(cv::Mat frame, const frc::AprilTagDetection* tag) {
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
void drawInferenceBox(std::vector<Detection> &detections, cv::Mat frame) {
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


int main(int argc, char** argv)
{  
    // Initialize AprilTag detector
    initAprilTagDetector();
    // Initialize cameras
    auto rawCameras = initCameras(camConfig);
    for(cs::UsbCamera& cam : rawCameras) {
        auto info = cam.GetInfo();
        std::cout << "Camera found: " << std::endl;
        std::cout << info.path << ", " << info.name << std::endl;
        cameras.push_back(Camera{(uint8_t)info.dev, &cam});
    }

    // Construct camera sink/sources
    for(Camera& cam : cameras) {
      if(cam.ref == nullptr) continue;
      std::cout << "Cam ID: " << (int)cam.id << std::endl;
      cam.sink = new cs::CvSink{frc::CameraServer::GetVideo(*cam.ref)};
      cam.source = new cs::CvSource{"source" + cam.id, camConfig};
    }

    if(!cameras.size()) {
      std::cout << "No viable cameras found!" << std::endl;
      return 0;
    }

    inferTarget = cameras[0].id;

    // Start capture on CvSources
    // TCP ports start at 1181 
    for(Camera& cam : cameras) {
      frc::CameraServer::StartAutomaticCapture(*cam.source);
    }
    
    // NT Initialization
    auto inst = nt::NetworkTableInstance::GetDefault();
    inst.SetServerTeam(6722);
    inst.StartClient4("jetson-client");
    auto table = inst.GetTable("/jetson");
    
    // Spin up separate thread to request inferencing
    std::thread inferenceSpawner([&]{
        // Find Jetson IP and port
        struct sockaddr_in server_addr;
        int result = 0;
        while(result != 1) {
          result = getMLServer(&server_addr);
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        /*std::thread test(debugTagPrint, 0, Transform3d{});*/
        /*std::thread test{createInferThread, &server_addr, &cameras[0]};*/
        /*mlThreads.push_back(std::move(test));*/
    });


    /*std::cout << "Size of Tag Frame: " << (int)TAG_FRAME_SIZE << std::endl;*/

    while(true) {
      // TEST TARGET TAGS
      auto requestedTags = table->GetRaw("rqsted", targetTags);
      targetTags.clear();
      targetTags.insert(targetTags.end(), requestedTags.begin(), requestedTags.end());
      // reallocate buffer if size changed
      uint8_t currentSize = targetTags.size();
      if(targetCount != currentSize) {
        free(tagBuffer);
        tagBufSize = sizeof(GlobalFrame) + ((TAG_FRAME_SIZE + 2) * currentSize * cameras.size());
        tagBuffer = (uint8_t*)malloc(tagBufSize);
        /*std::cout << "Size of buffer changed: " << (int)tagBufSize << std::endl;*/
      }
      targetCount = currentSize;
      // Debug printout
      /*std::cout << "Targeting: " << std::endl;*/
      /*for(uint8_t& id : targetTags) {*/
      /*  std::cout << (int)id << ' ';*/
      /*}*/
      /*std::cout << std::endl;*/
      
      // Collect frames from cameras
      // Done separately to try and keep cameras synced in real-time
         // Get the current time from the system clock
      auto now = std::chrono::system_clock::now();

      // Convert the current time to time since epoch
      auto duration = now.time_since_epoch();

      // Convert duration to milliseconds
      unsigned long milliseconds
          = std::chrono::duration_cast<std::chrono::milliseconds>(
                duration)
                .count();
      for(Camera& cam : cameras) {
        if(cam.lastFail && milliseconds - cam.lastFail < 3000) {
          continue;
        }
        auto success = cam.sink->GrabFrame(cam.frame);
        if(!success) {
          cam.lastFail = milliseconds;
        } else {
          cam.lastFail = 0;
        }
        cam.validData = !cam.lastFail && !cam.frame.empty();
        if (cam.validData) {
          cam.captureTime = milliseconds + success;
          cv::cvtColor(cam.frame, cam.gray, cv::COLOR_BGR2GRAY);
          cam.labelledFrame = cam.frame.clone();
          
          // Clone target camera frame into inference buffer
          /*if(cam.id == inferTarget) {*/
            /*if(!cam.newFrame) {*/
            /*    cam.inferFrame = cam.frame.clone();*/
            /*}*/
            cam.newFrame = true;
          /*}*/
        }
      }

      // Fill detections array if new detections have been sent
      // This being up-to-date is the inferThread's responsibility
      /*if(newInference) {*/
      /*  constructDetections(detectionJson, detections);*/
      /*}*/
      
      uint8_t tagBufPos = 0;
      GlobalFrame frameGlobal;
      tagBufPos += sizeof(GlobalFrame);

      // Main AprilTag processing loop. Done once per camera
      for(Camera& cam : cameras) {
        if(!cam.validData) continue;
        
        // Draw detections onto frame
        /*if(cam.id == inferTarget) {*/
        drawInferenceBox(cam.detections, cam.labelledFrame);
        /*}*/
        
        auto detections = frc::AprilTagDetect(detector, cam.gray);
        for(const frc::AprilTagDetection* tag : detections) {
          uint8_t id = tag->GetId();
          /*std::cout << "ID: " << (int)id << " found" << std::endl;*/
          uint8_t found = count(targetTags.begin(), targetTags.end(), id);
          if(!found) continue;  // tag not in request array, skip
          if(tagBufPos + TAG_FRAME_SIZE > tagBufSize) continue; // whoopsie, this would overflow, skip
          auto transform = estimator.Estimate(*tag);  // Estimate Transform3d relative to camera
          
          // Data to get shoved into buffer
          AprilTagFrame frame {
            id, 
            cam.id,
            cam.captureTime,
            transform.X().value(),
            transform.Y().value(),
            transform.Z().value(),
            units::degree_t{transform.Rotation().X()}.value(),
            units::degree_t{transform.Rotation().Y()}.value(),
            units::degree_t{transform.Rotation().Z()}.value()
          };
          
          // copy into buffer and increment counter
          memset(tagBuffer + tagBufPos, 0x69, 2);
          memcpy(tagBuffer + tagBufPos + 2, &frame, TAG_FRAME_SIZE);
          tagBufPos += 2 + TAG_FRAME_SIZE;

          // Print relative offset
          /*debugTagPrint(id, transform);*/
          
          // Draw box on our frame
          drawAprilTagBox(cam.labelledFrame, tag);
        }
      }

      frameGlobal.size[0] = tagBufPos & 0x00ff;
      frameGlobal.size[1] = (tagBufPos & 0xff00) >> 8;
      memcpy(tagBuffer, &frameGlobal, sizeof(GlobalFrame));

      // Post tag buffer to NT
      std::vector<uint8_t> tagBuf(tagBuffer, tagBuffer + tagBufPos);
      table->PutRaw("tagBuf", tagBuf);

      // Write frames to publishing source
      // Done separately because synced web streams are nice
      for(Camera& cam : cameras) {
        if(cam.validData) cam.source->PutFrame(cam.labelledFrame);
      }
    }
}
