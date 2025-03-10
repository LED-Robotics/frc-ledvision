#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <units/length.h>

#include "Camera.h"
#include "common.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> 

using namespace frc;

// Camera resolution/format configs
int width = 640;
int height = 480;
cs::VideoMode camConfig{cs::VideoMode::PixelFormat::kMJPEG, width, height, 30};

// To store IDs of current valid cameras
std::vector<uint8_t> currentCams;

// Variables for sending AprilTag detections
std::vector<uint8_t> targetTags;
uint8_t targetCount = 0xff;
uint8_t *tagBuffer = nullptr;
uint32_t tagBufSize = 0;

// Variables for sending ML detections
uint8_t maxDetections = 100;
uint8_t *mlBuffer = nullptr;
uint32_t mlBufSize = 0;
uint8_t camsInferencing = 0xff;
std::vector<uint8_t> camMLDisabled;

std::vector<cs::UsbCamera> rawCams; // Global raw camera references
std::vector<Camera> cameras; // Global camera references

// Struct format for AprilTag detection
struct AprilTagFrame {
  uint8_t tagId = 0;
  uint8_t camId = 0;
  uint32_t timeCaptured;
  double tx;
  double ty;
  double tz;
  double rx;
  double ry;
  double rz;
};

// Struct format for ML detection
struct MLDetectionFrame {
  uint8_t label = 0;
  uint8_t camId = 0;
  uint32_t timeCaptured;
  double x;
  double y;
  double w;
  double h;
};

const size_t TAG_FRAME_SIZE = sizeof(AprilTagFrame);
const size_t ML_FRAME_SIZE = sizeof(MLDetectionFrame);

// Global data to send in the AprilTag frame
struct GlobalFrame {
  uint8_t size[2];
};

std::string getNewFileName() {
  std::string path = "../videos";
  if(!IsPathExist(path)) {
    std::filesystem::create_directory(path);
  }
  int num = 0;
  for (const auto & entry : std::filesystem::directory_iterator(path)) {
    auto name = entry.path().generic_string();
    /*std::cout << name << std::endl;*/
    auto aviPos = name.find(".avi");
    if(aviPos != std::string::npos) {
      auto startPos = name.find("_") + 1;
      auto numStr = name.substr(startPos, aviPos - startPos);
      /*std::cout << "numstr: " << numStr << '\n';*/
      int currentNum = std::stoi(numStr);
      if(currentNum >= num) num = currentNum + 1;
    }
  }
  return path + "/output_" + std::to_string(num) + ".avi";
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

MLDetectionFrame generateMLFrame(det::DetectObject &det, uint8_t camId, uint32_t capTime) {
  return {
    (uint8_t)det.label, 
    camId,
    capTime,
    det.rect.x,
    det.rect.y,
    det.rect.width,
    det.rect.height,
  };
}

MLDetectionFrame generateMLFrame(det::PoseObject &det, uint8_t camId, uint32_t capTime) {
  return {
    (uint8_t)det.label, 
    camId,
    capTime,
    det.rect.x,
    det.rect.y,
    det.rect.width,
    det.rect.height,
  };
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
    std::cout << "Cam ID: " << (int)cam.GetID() << std::endl;
  }

  if(!cameras.size()) {
    std::cout << "No viable cameras found!" << std::endl;
    return 0;
  }

  // NT Initialization
  auto inst = nt::NetworkTableInstance::GetDefault();
  inst.SetServerTeam(6722);
  inst.StartClient4("jetson-client");
  auto table = inst.GetTable("/jetson");

  /*model = new YOLO11("../engines/reefscape_v5.engine");*/
  /*model->make_pipe(true);*/
  
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  // Start capture on CvSources
  // TCP ports start at 1181 
  std::string onnxPath = "../engines/reefscape_v5.onnx";
  std::string enginePath = onnxPath.substr(0, onnxPath.size() - 4) + "engine";  
  bool modelFound = false;
  if(!IsPathExist(enginePath)) {
    if(IsPathExist(onnxPath)) {
      YOLO11::generateEngine(onnxPath);
      modelFound = IsPathExist(enginePath);
    }
  } else {
    modelFound = true;
  }

  for(Camera& cam : cameras) {
    cam.StartStream();    
    if(modelFound) {
      cam.SetMLDetectionMode(Camera::MLMode::Detect);
      cam.StartInferencing(enginePath);
    }
    /*cam.SetMLDetectionMode(Camera::MLMode::Pose);*/
    /*cam.LoadModel("../engines/reefscape_v5.engine");*/
    
    /*cam.StartInferencing("../engines/yolo11n-pose.engine");*/
    /*cam.StartInferencing("../engines/yolo11x-pose.engine");*/
  }

  /*std::cout << "Size of Tag Frame: " << (int)TAG_FRAME_SIZE << std::endl;*/

    bool lastRecordState = false;

    while(true) {
    bool recordState = table->GetBoolean("recordState", false);
    bool recordLabelled = table->GetBoolean("recordLabelled", false);
    if(recordState != lastRecordState) {
      lastRecordState = recordState;

      auto orig = getNewFileName();
      for(Camera& cam : cameras) {
        if(recordState) {
          std::string newPath = orig.substr(0, orig.size() - 4) + "_id_" + std::to_string(cam.GetID()) + ".avi";  
          cam.StartRecording(newPath, recordLabelled);
        } else {
          cam.StopRecording();
        }
      }
    }

    auto requestedTags = table->GetRaw("rqsted", targetTags);
    camMLDisabled = table->GetRaw("mlOff", camMLDisabled);
    targetTags.clear();
    targetTags.insert(targetTags.end(), requestedTags.begin(), requestedTags.end());
    for(Camera& cam : cameras) {
      cam.SetTargetTags(targetTags);
      auto id = cam.GetID();
      for(int i = 0; i < camMLDisabled.size(); i++) {
        uint8_t found = count(targetTags.begin(), targetTags.end(), id);
        if(found) cam.DisableInference();
        else cam.EnableInference();
      }
    }
    
    // reallocate tag buffer if size changed
    uint8_t currentSize = targetTags.size();
    if(targetCount != currentSize) {
      free(tagBuffer);
      tagBufSize = sizeof(GlobalFrame) + ((TAG_FRAME_SIZE + 2) * currentSize * cameras.size());
      /*std::cout << "Size of buffer changed: " << (int)tagBufSize << std::endl;*/
      tagBuffer = (uint8_t*)malloc(tagBufSize);
    }
    targetCount = currentSize;

    // reallocate ML buffer if size changed
    currentSize = cameras.size();
    if(camsInferencing != currentSize) {
      free(mlBuffer);
      mlBufSize = sizeof(GlobalFrame) + ((ML_FRAME_SIZE + 2) * currentSize * maxDetections);
      /*std::cout << "Size of buffer changed: " << (int)tagBufSize << std::endl;*/
      mlBuffer = (uint8_t*)malloc(mlBufSize);
    }
    camsInferencing = currentSize;
    
    uint32_t tagBufPos = 0;
    GlobalFrame tagFrameGlobal;
    tagBufPos += sizeof(GlobalFrame);

    for(Camera& cam : cameras) {
      if(!cam.GetTagDetectionCount()) continue;
      auto tagDetections = cam.GetTagDetections();
      auto camId = cam.GetID();
      auto capTime = cam.GetCaptureTime();
      /*cam.PauseTagDetection();*/
      for(Camera::TagDetection &det : *tagDetections) {
        if(tagBufPos + TAG_FRAME_SIZE > tagBufSize) continue; // whoopsie, this would overflow, skip
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

        // copy into buffer and increment counter
        memset(tagBuffer + tagBufPos, 0x69, 2);
        memcpy(tagBuffer + tagBufPos + 2, &frame, TAG_FRAME_SIZE);
        tagBufPos += 2 + TAG_FRAME_SIZE;

      }
      /*cam.ResumeTagDetection();*/
    }

    tagFrameGlobal.size[0] = tagBufPos & 0x00ff;
    tagFrameGlobal.size[1] = (tagBufPos & 0xff00) >> 8;
    memcpy(tagBuffer, &tagFrameGlobal, sizeof(GlobalFrame));

    // Post tag buffer to NT
    std::vector<uint8_t> tagBuf(tagBuffer, tagBuffer + tagBufPos);
    table->PutRaw("tagBuf", tagBuf);

    uint32_t mlBufPos = 0;
    GlobalFrame mlFrameGlobal;
    mlBufPos += sizeof(GlobalFrame);

    for(Camera& cam : cameras) {
      if(!cam.GetMLEnabled()) continue;
      if(!cam.GetMLDetectionCount()) continue;
      cam.FreezeMLBufs();
      auto camId = cam.GetID();
      auto capTime = cam.GetCaptureTime();
      if(cam.GetMLDetectionMode() == Camera::MLMode::Detect) {
        auto mlDetections = cam.GetBoxDetections();
        for(det::DetectObject &det : *mlDetections) {
          if(mlBufPos + ML_FRAME_SIZE > mlBufSize) continue; // whoopsie, this would overflow, skip
          auto frame = generateMLFrame(det, camId, capTime);
          // copy into buffer and increment counter
          memset(mlBuffer + mlBufPos, 0x69, 2);
          memcpy(mlBuffer + mlBufPos + 2, &frame, ML_FRAME_SIZE);
          mlBufPos += 2 + ML_FRAME_SIZE;
        }
      } else if(cam.GetMLDetectionMode() == Camera::MLMode::Pose) {
        auto mlDetections = cam.GetPoseDetections();
        for(det::PoseObject &det : *mlDetections) {
          if(mlBufPos + ML_FRAME_SIZE > mlBufSize) continue; // whoopsie, this would overflow, skip
          auto frame = generateMLFrame(det, camId, capTime);
          // copy into buffer and increment counter
          memset(mlBuffer + mlBufPos, 0x69, 2);
          memcpy(mlBuffer + mlBufPos + 2, &frame, ML_FRAME_SIZE);
          mlBufPos += 2 + ML_FRAME_SIZE;
        }
      }
      cam.UnfreezeMLBufs();
    }

    mlFrameGlobal.size[0] = mlBufPos & 0x00ff;
    mlFrameGlobal.size[1] = (mlBufPos & 0xff00) >> 8;
    memcpy(mlBuffer, &mlFrameGlobal, sizeof(GlobalFrame));

    // Post tag buffer to NT
    std::vector<uint8_t> mlBuf(mlBuffer, mlBuffer + mlBufPos);
    table->PutRaw("mlBuf", mlBuf);


    /*std::this_thread::sleep_for(std::chrono::milliseconds(20));*/
  }
}
