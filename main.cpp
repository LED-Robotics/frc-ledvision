#include <iostream>
#include <vector>
#include <filesystem>
#include "chrono"
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <apriltag/frc/apriltag/AprilTagDetector.h>
#include <apriltag/frc/apriltag/AprilTagDetector_cv.h>
#include <apriltag/frc/apriltag/AprilTagPoseEstimator.h>
#include <apriltag/frc/apriltag/AprilTagFieldLayout.h>
#include <apriltag/frc/apriltag/AprilTagFields.h>
#include <cameraserver/CameraServer.h>
#include <units/length.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> 

using namespace frc;

int width = 640;
int height = 640;
cs::VideoMode camConfig{cs::VideoMode::PixelFormat::kMJPEG, width, height, 30};

AprilTagDetector detector{};
AprilTagPoseEstimator estimator{{140_mm, width, height, width/2, height/2}};  // dummy numbers
auto tagLayout = AprilTagFieldLayout::LoadField(AprilTagField::k2024Crescendo);

void printCoordinates(int id) {
    auto idSearch = tagLayout.GetTagPose(id);
    if(idSearch.has_value()) {  // ensure tag ID exists in our field coordinate system
        auto tag = idSearch.value();
        auto tagTranslation = tag.Translation();
        std::cout << "Tag " << id << " Pose:" << std::endl;
        std::cout << "X: " << tagTranslation.X().value() << "Y: " << tagTranslation.Y().value()<< "Z: " << tagTranslation.Z().value() << std::endl;
        std::cout << "Rotation: " << tag.Rotation().ToRotation2d().Degrees().value() << std::endl;
    } else {
        std::cout << "Tag with id " << id << " was not found" << std::endl;
    }
}

// Init and return all cameras plugged in
std::vector<cs::UsbCamera> initCameras(cs::VideoMode config) {
    CS_Status status = 0;
    std::vector<cs::UsbCamera> cameras{};
    int number = 0;
    for (const auto& caminfo : cs::EnumerateUsbCameras(&status)) {
        fmt::print("Dev {}: Path {} (Name {})\n", caminfo.dev, caminfo.path, caminfo.name);
        fmt::print("vid {}: pid {}\n", caminfo.vendorId, caminfo.productId);
        cs::UsbCamera cam{"camera-" + number, caminfo.path};
        cam.SetVideoMode(config);
        cameras.push_back(cam);
    }
    return cameras;
}

int main(int argc, char** argv)
{
    detector.AddFamily("tag36h11");
    detector.SetConfig({});
    auto quadParams = detector.GetQuadThresholdParameters();
    quadParams.minClusterPixels = 3;
    detector.SetQuadThresholdParameters(quadParams);
    auto cameras = initCameras(camConfig);
    cs::UsbCamera* testCam = nullptr;
    for(cs::UsbCamera& cam : cameras) {
        auto info = cam.GetInfo();
        std::cout << "Camera found: " << std::endl;
        std::cout << info.path << ", " << info.name << std::endl;
        if(info.name == "Arducam OV9782 USB Camera") {
            std::cout << "Test cam found!" << std::endl;
            testCam = &cam;
        }
    }
    if(testCam != nullptr) {
        cs::CvSink testSink{frc::CameraServer::GetVideo(*testCam)};
        cs::CvSource testSource{"testSource", camConfig};
        frc::CameraServer::StartAutomaticCapture(testSource);
        cv::Mat frame, grayFrame;

        while(true) {
            int frameTime = testSink.GrabFrameNoTimeout(frame);
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            
            auto detections = frc::AprilTagDetect(detector, grayFrame);
            // std::cout << detections.size() << " detections!" << std::endl;
            for(const frc::AprilTagDetection* tag : detections) {
                auto transform = estimator.Estimate(*tag);
                std::cout << "Tag " << tag->GetId() << "Pose Estimation:" << std::endl;
                std::cout << "X Off: " << transform.X().value();
                std::cout << "Y Off: " << transform.Y().value();
                std::cout << "Z Off: " << transform.Z().value() << std::endl;
                std::cout << "Rot Off: " << transform.Rotation().ToRotation2d().Degrees().value() << std::endl;
                transform.X();
                transform.Y();
                // Draw boxes around tags
                for(int i = 0; i < 4; i++) {
                    auto point1 = tag->GetCorner(i);
                    int secondIndex = i == 3 ? 0 : i + 1;   // out of bounds adjust for last iteration
                    auto point2 = tag->GetCorner(secondIndex);
                    cv::Point lineStart{point1.x, point1.y};
                    cv::Point lineEnd{point2.x, point2.y};
                    cv::line(frame, lineStart, lineEnd, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
                }
            }



            testSource.PutFrame(frame); // post to stream
        }
    }
    // print coordinates/rotation of every AprilTag from Crescendo
    // for(int i = 1; i < 17; i++) {
        // printCoordinates(i);
    // }
    // std::cout << "Done printing." << std::endl;
}
