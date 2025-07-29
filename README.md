A custom C++ vision application for use in the FIRST Robotics Competition. 

Currently featuring automatic AprilTag and ML target detection on up to 4 USB cameras simultaneously. 

Leveraging WPILib for 3D AprilTag pose estimation, and a custom TensorRT engine for blazing fast YOLOv11 inference. 

Currently targeting Jetson Orin devices running JetPack 6.2

This repo includes the source code for the application and some Dockerfiles to build a dockerized version of the application for various architectures. 

To compile from source ensure you have CUDA, TensorRT, OpenCV, and WPILib available to cmake. 
