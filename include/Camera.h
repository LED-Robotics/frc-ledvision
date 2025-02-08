
#include <iostream>
#include <vector>
#include <cameraserver/CameraServer.h>
#include <thread>
#include <chrono>
#include "networking.h" 
#include <apriltag/frc/apriltag/AprilTagDetector.h>
#include <apriltag/frc/apriltag/AprilTagDetector_cv.h>
#include <apriltag/frc/apriltag/AprilTagPoseEstimator.h>

using namespace frc;

class Camera {
  public:
    Camera(cs::UsbCamera *cam, cs::VideoMode config, AprilTagPoseEstimator::Config estConfig);

    struct TagDetection {
      int id = -1;
      std::vector<AprilTagDetection::Point> corners;
      Transform3d transform;
    };

    int GetID();
    
    std::vector<uint8_t> GetTargetTags();

    void SetTargetTags(std::vector<uint8_t> targets);

    void DrawAprilTagBox(cv::Mat frame, TagDetection* tag);

    void DrawInferenceBox(cv::Mat frame, std::vector<Detection> &detections);

    bool ValidPresent();

    void StartStream();

    void StartCollector();

    void StartGrayscaleConverter();

    void StartProcessor();

    void StartInferencing(struct sockaddr_in *server_addr, int client_sock);

    void InferenceThread();

    void StartLabeller();

    void StartPosting();

    
  
  private:
    const int threadDelay = 0;
    std::vector<uint8_t> targetTags{22, 18};

    int id = -1;
    cs::UsbCamera *cam = nullptr;
    cs::CvSink *sink = nullptr;
    cs::CvSource *source = nullptr;
    AprilTagDetector detector{};
    AprilTagPoseEstimator estimator;
    cv::Mat frame{};
    cv::Mat gray{};
    cv::Mat labelled{};
    bool inferenceEnabled = false;
    struct sockaddr_in *ml_addr;
    int sock = -1;
  
    uint32_t captureTime = 0;
    unsigned long lastFail = 0;
    bool newFrame = false;
    bool newInference = false;
    bool frameProcessed = false;
    bool validFrame = false;
    bool grayAvailable = false;
    bool frameLabelled = false;
    bool framePosted = false;
    std::vector<TagDetection> detectionData;
    std::vector<Detection> detections;

    std::thread collector;
    std::thread converter;
    std::thread processor;
    std::thread mlThread;
    std::thread labeller;
    std::thread poster;
};
