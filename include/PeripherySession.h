#pragma once

#include "Networking.h"

class PeripherySession {
  public:
    PeripherySession(uint32_t id, struct sockaddr_in session_addr, bool correctlyConfigured = true);
    
    // Representation of an ML detection
    struct Detection {
        uint8_t label = 0;
        double x = 0;
        double y = 0;
        double width = 0;
        double height = 0;
        std::vector<double> kps = {};
    };

    // Format given buffer into a Detection
    static Detection ConstructDetection(uchar *buf);

    std::vector<Detection> RunInference(cv::Mat frame);

    bool valid = false;

  private:
    struct sockaddr_in session_address;
    uint32_t sessionId = 0;
    int sock = -1;
    int timeoutfd = -1;

    // Max datagram length for image stream
    constexpr static int MaxDatagram = 32768;
    uchar request[MaxDatagram];
    uchar response[MaxDatagram];
  };
