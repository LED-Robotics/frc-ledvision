#pragma once

#include "Networking.hpp"
#include "common.hpp"

class PeripherySession {
  public:
    PeripherySession(uint32_t id, struct sockaddr_in session_addr, bool correctlyConfigured = true);
    
    // Return session ID
    uint32_t GetID();

    // Fill a DetectObject from buffer, return pointer to last byte
    static uchar* FillDetectObject(det::DetectObject *object, uchar *buf);

    // Fill a keypoints vector from buffer, return pointer to last byte
    static uchar* FillKeypoints(std::vector<float> &kps, uchar *buf);

    // Format given buffer into a DetectObject
    static det::DetectObject ConstructDetectObject(uchar *buf);

    // Format given buffer into a PoseObject
    static det::PoseObject ConstructPoseObject(uchar *buf);

    std::vector<det::DetectObject> RunDetectInference(cv::Mat frame);

    std::vector<det::PoseObject> RunPoseInference(cv::Mat frame);

    bool valid = false;

  private:
    struct sockaddr_in session_address;
    uint32_t sessionId = 0;
    int sock = -1;
    int timeoutfd = -1;
    struct pollfd fd;

    // Max datagram length for image stream
    constexpr static int MaxDatagram = 49151;
    uchar request[MaxDatagram];
    uchar response[MaxDatagram];
  };
