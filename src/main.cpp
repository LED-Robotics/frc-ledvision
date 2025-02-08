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

std::vector<cs::UsbCamera> rawCams; // Global raw camera references
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
void initCameras(cs::VideoMode config) {
  CS_Status status = 0;
  for (const auto& caminfo : cs::EnumerateUsbCameras(&status)) {
    fmt::print("Dev {}: Path {} (Name {})\n", caminfo.dev, caminfo.path, caminfo.name);
    fmt::print("vid {}: pid {}\n", caminfo.vendorId, caminfo.productId);
    cs::UsbCamera cam{"camera-" + caminfo.dev, caminfo.path};
    /*cam.SetVideoMode(config);*/
    rawCams.push_back(cam);
  }
}

int main(int argc, char** argv)
{  
  // Initialize cameras
 initCameras(camConfig);
  for(cs::UsbCamera& cam : rawCams) {
      auto info = cam.GetInfo();
      std::cout << "Camera found: " << std::endl;
      std::cout << info.path << ", " << info.name << std::endl;
      cameras.push_back({&cam, camConfig, {6.5_in, (double)640, (double)480, (double)320, (double)240}});
  }

  // Construct camera sink/sources
  for(Camera& cam : cameras) {
    /*if(cam.ref == nullptr) continue;*/
    std::cout << "Cam ID: " << cam.GetID() << std::endl;
  }

  if(!cameras.size()) {
    std::cout << "No viable cameras found!" << std::endl;
    return 0;
  }

  inferTarget = cameras[0].GetID();
  
  // NT Initialization
  auto inst = nt::NetworkTableInstance::GetDefault();
  inst.SetServerTeam(6722);
  inst.StartClient4("jetson-client");
  auto table = inst.GetTable("/jetson");
  
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  // Start capture on CvSources
  // TCP ports start at 1181 
  for(Camera& cam : cameras) {
    cam.StartStream();    
  }

  struct sockaddr_in ml_server_addr;
  int sock = -1;
  // Spin up separate thread to request inferencing
  std::thread inferenceSpawner([&]{
    // Find Jetson IP and port
    struct sockaddr_in ml_server_addr;
    int result = 0;
    while(result != 1) {
      result = getMLServer(&ml_server_addr);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    sock = getSocket(&ml_server_addr);
    for(Camera& cam : cameras) {
      cam.StartInferencing(&ml_server_addr, sock);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });


  /*std::cout << "Size of Tag Frame: " << (int)TAG_FRAME_SIZE << std::endl;*/

  while(true) {
    // TEST TARGET TAGS
    auto requestedTags = table->GetRaw("rqsted", targetTags);
    targetTags.clear();
    targetTags.insert(targetTags.end(), requestedTags.begin(), requestedTags.end());
    for(Camera& cam : cameras) {
      cam.SetTargetTags(targetTags);
    }
    
    // reallocate buffer if size changed
    uint8_t currentSize = targetTags.size();
    if(targetCount != currentSize) {
      free(tagBuffer);
      tagBufSize = sizeof(GlobalFrame) + ((TAG_FRAME_SIZE + 2) * currentSize * cameras.size());
      tagBuffer = (uint8_t*)malloc(tagBufSize);
      /*std::cout << "Size of buffer changed: " << (int)tagBufSize << std::endl;*/
    }
    targetCount = currentSize;
    for(Camera& cam : cameras) {
      if(!cam.GetTagDetectionCount()) continue;
      auto tagDetections = cam.GetTagDetections();
      auto camId = cam.GetID();
      auto capTime = cam.GetCaptureTime();
      for(Camera::TagDetection &det : tagDetections) {
        // Data to get shoved into buffer
        AprilTagFrame frame {
          det.id, 
          camId,
          capTime,
          det.transform.X().value(),
          det.transform.Y().value(),
          det.transform.Z().value(),
          units::degree_t{det.transform.Rotation().X()}.value(),
          units::degree_t{det.transform.Rotation().Y()}.value(),
          units::degree_t{det.transform.Rotation().Z()}.value()
        };
      }
    }

    /*std::this_thread::sleep_for(std::chrono::milliseconds(20));*/
  }
}
