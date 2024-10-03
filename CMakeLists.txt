cmake_minimum_required(VERSION 3.11)
project(frc_ledvision_2025)

find_package(wpilib REQUIRED)

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(frc_ledvision main.cpp) # executable name as first parameter
target_link_libraries(frc_ledvision cameraserver ntcore cscore wpiutil wpimath apriltag nlohmann_json::nlohmann_json)
