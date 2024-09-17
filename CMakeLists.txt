cmake_minimum_required(VERSION 3.11)
project(frc_ledvision_2025)

find_package(wpilib REQUIRED)

add_executable(frc_ledvision main.cpp) # executable name as first parameter
target_link_libraries(frc_ledvision cameraserver ntcore cscore wpiutil wpimath apriltag)
