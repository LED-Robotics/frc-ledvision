cmake_minimum_required(VERSION 3.11)
project(frc_ledvision_2025)

find_package(wpilib REQUIRED)

include_directories(include)

add_executable(
  frc_ledvision src/main.cpp
  src/Camera.cpp
  src/Networking.cpp
  src/PeripheryClient.cpp
  src/PeripherySession.cpp
  include/Camera.h
  include/Networking.h
  include/PeripheryClient.h
  include/PeripherySession.h
  ) # executable name as first parameter
target_link_libraries(frc_ledvision cameraserver ntcore cscore wpiutil wpimath apriltag)
