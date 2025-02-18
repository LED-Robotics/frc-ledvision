cmake_minimum_required(VERSION 3.11)
project(frc_ledvision_2025)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

find_package(wpilib REQUIRED)

include_directories(include ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

add_executable(
  frc_ledvision src/main.cpp
  src/Camera.cpp
  include/Camera.h
  include/yolo11.hpp
  include/common.hpp
  ) # executable name as first parameter
target_link_libraries(frc_ledvision cameraserver ntcore cscore wpiutil wpimath apriltag nvinfer nvinfer_plugin nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBS})
