#include <iostream>
#include <filesystem>
#include "chrono"
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <apriltag/frc/apriltag/AprilTagDetector.h>
#include <apriltag/frc/apriltag/AprilTagPoseEstimator.h>
#include <apriltag/frc/apriltag/AprilTagFieldLayout.h>
#include <apriltag/frc/apriltag/AprilTagFields.h>
#include <cameraserver/CameraServer.h>
#include <units/length.h>

using namespace frc;

AprilTagDetector detector{};
AprilTagPoseEstimator estimator{{140_mm, 40, 40, 40, 40}};  // dummy numbers
auto tagLayout = AprilTagFieldLayout::LoadField(AprilTagField::k2024Crescendo);

void printCoordinates(int id) {
    auto idSearch = tagLayout.GetTagPose(id);
    if(idSearch.has_value()) {  // ensure tag ID exists in our field coordinate system
        auto tag = idSearch.value();
        auto tagTranslation = tag.Translation();
        std::cout << "Tag " << id << " Pose:" << std::endl;
        std::cout << "X: " << tagTranslation.X().value() << "Y: " << tagTranslation.Y().value()<< "Z: " << tagTranslation.Z().value() << std::endl;
        std::cout << "Rotation: " << tag.Rotation().ToRotation2d().Degrees().value() << std::endl;
    }
}

int main(int argc, char** argv)
{
    // print coordinates/rotation of every AprilTag from Crescendo
    for(int i = 1; i < 17; i++) {
        printCoordinates(i);
    }
    std::cout << "Done printing." << std::endl;
}
