
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

class Cameraa {
  public:
    Cameraa(cs::UsbCamera *cam, cs::VideoMode config, AprilTagDetector *detRef, AprilTagPoseEstimator *estRef);
    
    std::vector<uint8_t> GetTargetTags();

    void SetTargetTags(std::vector<uint8_t> targets);

    void DrawAprilTagBox(cv::Mat frame, const frc::AprilTagDetection* tag);

    void DrawInferenceBox(cv::Mat frame, std::vector<Detection> &detections);

    bool ValidPresent();

    void StartCollector();

    void StartGrayscaleConverter();

    void StartProcessor();

    void StartInferencing(struct sockaddr_in *server_addr);

    void StartLabeller();

    void StartPosting();
  
  private:
    const int threadDelay = 20;
    std::vector<uint8_t> targetTags;

    int id = -1;
    cs::UsbCamera *cam;
    cs::CvSink *sink;
    cs::CvSource *source;
    AprilTagDetector *detector;
    AprilTagPoseEstimator *estimator;
    cv::Mat frame{};
    cv::Mat gray{};
    cv::Mat labelled{};
    bool inferenceEnabled = false;
  
    uint32_t captureTime = 0;
    unsigned long lastFail = 0;
    bool newFrame = false;
    bool newInference = false;
    bool frameProcessed = false;
    bool validFrame = false;
    bool grayAvailable = false;
    bool frameLabelled = false;
    AprilTagDetector::Results aprilTags;
    std::vector<Detection> detections;

    std::thread collector;
    std::thread converter;
    std::thread processor;
    std::thread mlThread;
    std::thread labeller;
    std::thread poster;
};
