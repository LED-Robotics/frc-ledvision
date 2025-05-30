#pragma once

#include <apriltag/frc/apriltag/AprilTagDetector.h>
#include <apriltag/frc/apriltag/AprilTagDetector_cv.h>
#include <apriltag/frc/apriltag/AprilTagPoseEstimator.h>
#include <cameraserver/CameraServer.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#ifdef CUDA_PRESENT
#include "yolo11.hpp"
#else
#include "PeripherySession.hpp"
#include "common.hpp"
#endif

using namespace frc;

class Camera {
public:
  Camera(cs::UsbCamera *cam, cs::VideoMode config,
         AprilTagPoseEstimator::Config estConfig);

  // AprilTag Detection struct
  struct TagDetection {
    uint8_t id = -1;
    std::vector<AprilTagDetection::Point> corners;
    Transform3d transform;
  };

  enum MLMode { Detect, Pose };

  // Fetch Camera id
  uint8_t GetID();

  // Get current AprilTags being estimated
  std::vector<uint8_t> GetTargetTags();

  // Set current AprilTags being estimated
  void SetTargetTags(std::vector<uint8_t> targets);

  // Get current TagDetection vector from Camera
  std::vector<Camera::TagDetection> *GetTagDetections();

  // Get total current tag detections
  int GetTagDetectionCount();

  // Get current box ML Detection vector from Camera
  std::vector<det::BoxObject> *GetBoxDetections();

  // Get current box ML Detection vector from Camera
  std::vector<det::PoseObject> *GetPoseDetections();

  // Get stale box ML Detection vector from Camera
  std::vector<det::BoxObject> *GetInactiveBoxDetections();

  // Get stale box ML Detection vector from Camera
  std::vector<det::PoseObject> *GetInactivePoseDetections();

  // Prevent buffer swapping
  void FreezeMLBufs();

  // Allow buffer swapping
  void UnfreezeMLBufs();

  // Switch toggle active buffer for reading
  void SwitchActiveMLBuf();

  // Get total current ML detections
  int GetMLDetectionCount();

  // Get current ML detection mode
  int GetMLDetectionMode();

  // Get current ML detection mode
  int GetMLEnabled();

  // Set current ML detection mode
  void SetMLDetectionMode(int mode);

  // Get system time (millis) of last frame grab
  uint32_t GetCaptureTime();

  // Stop overwriting the tag detection buffer
  void PauseTagDetection();

  // Resume overwriting the tag detection buffer
  void ResumeTagDetection();

  // Draw AprilTag outline on frame
  void DrawAprilTagBox(cv::Mat frame, TagDetection *tag);

  // Draw ML detection on frame
  void DrawDetectBox(cv::Mat frame, det::BoxObject &detection);

  // Draw ML detection on frame
  void DrawPoseBox(cv::Mat frame, det::PoseObject &detection);

  // Check if ML frame is ready
  bool IsMLFrameAvailable();

  // Set ML frame for refresh
  void SetMLFrameUnavailable();

  // Return ML frame for inference
  cv::Mat GetMLFrame();

  // Check if there is currently a valid frame from the Camera
  bool ValidPresent();

  // Start all threads except machine learning
  void StartStream();

  // Start frame collector
  void StartCollector();

  // Start frame cloning/transformation
  void StartGrayscaleConverter();

  // Start processing frames
  void StartProcessor();

  // Start ML thread
  void StartInferencing(std::string path);

#ifndef CUDA_PRESENT
  void StartInferencing(PeripherySession session);
#endif

  // Actual ML lambda
  void InferenceThread();

  // Start labelling frames
  void StartLabeller();

  // Start posting labelled frames
  void StartPosting();

  // Check if model/remote session is available
  bool IsInferencePossible();

  // Disable ML
  void DisableInference();

  // Enable ML (if model is loaded)
  void EnableInference();

  // Dealloc ML model
  void DestroyModel();

  // Load ML model into memory
  void LoadModel(std::string path);

  // Run detect inference on frame
  void RunInference(cv::Mat frame, std::vector<det::BoxObject> *dets);

  // Run pose inference on frame
  void RunInference(cv::Mat frame, std::vector<det::PoseObject> *dets);

  // Start recording video
  bool StartRecording(std::string path, bool labelled = false);

  // Stop recording video
  bool StopRecording();

  // Check if cam is recording
  bool GetRecording();

  // Return if an ML session is present
  bool GetMLSessionAvailable();

  // Return ML Session ID
  uint32_t GetMLSessionID();

private:
  const int threadDelay = 1;
  std::vector<uint8_t> targetTags{22, 18};

  uint8_t id = -1;
  cs::UsbCamera *cam = nullptr;
  cs::CvSink *sink = nullptr;
  cs::CvSource source;
  AprilTagDetector detector{};
  AprilTagPoseEstimator estimator;
  cv::Mat frame{};
  cv::Mat mlFrame{};
  cv::Mat gray{};
  cv::Mat labelled{};
  bool mlEnabled = true;

  uint32_t captureTime = 0;
  unsigned long lastFail = 0;
  bool newFrame = false;
  bool newInference = false;
  bool frameProcessed = false;
  bool validFrame = false;
  bool grayAvailable = false;
  bool frameLabelled = false;
  bool framePosted = false;
  bool mlFrameAvailable = false;
  bool newDetections = false;
  bool pauseTagDetections = false;
  bool mlBufsFrozen = false;

  std::vector<TagDetection> tagDetections;
  int tagDetectionCount = 0;
  int mlDetectionCount = 0;
  int mlMode = MLMode::Detect;
  // Buffers for inference
  std::vector<det::BoxObject> *detVector1 = new std::vector<det::BoxObject>();
  std::vector<det::BoxObject> *detVector2 = new std::vector<det::BoxObject>();
  std::vector<det::BoxObject> *boxLabelVector = detVector1;
  std::vector<det::BoxObject> *inactiveBoxLabelVector = detVector2;
  std::vector<det::PoseObject> *poseVector1 =
      new std::vector<det::PoseObject>();
  std::vector<det::PoseObject> *poseVector2 =
      new std::vector<det::PoseObject>();
  std::vector<det::PoseObject> *poseLabelVector = poseVector1;
  std::vector<det::PoseObject> *inactivePoseLabelVector = poseVector2;

#ifdef CUDA_PRESENT
  // Pointer for on-device ML model
  YOLO11 *model = nullptr;

#else
  // Remote inference session
  std::vector<PeripherySession> mlSessions;

#endif

  std::thread collector;
  std::thread converter;
  std::thread processor;
  std::thread labeller;
  std::thread poster;
  std::thread mlThread;

  bool recordState = false;
  bool recording = false;
  bool recordingLabelled = false;

  cv::VideoWriter outputVideo;
};
