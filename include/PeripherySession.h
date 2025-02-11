#pragma once

#include "Networking.h"

class PeripherySession {
  public:
    PeripherySession();
    
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

  private:
    struct sockaddr_in session_address;
    int sock = -1;
    int timeoutfd = -1;

    const int COMMAND_PORT = 5800;

    // Max datagram length for image stream
    static const int MaxDatagram = 32768;
    uchar request[PeripherySession::MaxDatagram];
    uchar response[32768];
  };
