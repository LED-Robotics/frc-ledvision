cmake_minimum_required(VERSION 3.11)
project(frc_ledvision_2025)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

if(CUDA_FOUND)
  message("Found CUDA")
else()
  message("Couldn't find CUDA")
endif()


# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

find_package(wpilib REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})
set(SOURCE_FILES
  src/main.cpp
  src/Camera.cpp
  include/Camera.hpp
  include/common.hpp
)

if(CUDA_FOUND)
  include_directories(include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
  set(SOURCE_FILES 
    ${SOURCE_FILES}
    src/yolo11.cpp
    include/yolo11.hpp
  )
else()
  set(SOURCE_FILES 
    ${SOURCE_FILES}
    src/Networking.cpp
    src/PeripheryClient.cpp
    src/PeripherySession.cpp
    include/Networking.hpp
    include/PeripheryClient.hpp
    include/PeripherySession.hpp
  )
endif()


add_executable(frc_ledvision ${SOURCE_FILES})

target_link_libraries(frc_ledvision cameraserver ntcore cscore wpiutil wpimath apriltag)
if(CUDA_FOUND)
  target_link_libraries(frc_ledvision nvinfer nvinfer_plugin nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBS})
  target_compile_definitions(frc_ledvision PUBLIC CUDA_PRESENT=$(CUDA_FOUND))
endif()

