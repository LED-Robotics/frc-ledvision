#pragma once

#include <arpa/inet.h>
#include <linux/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Representation of an ML detection
struct Detection {
    uint8_t label = 0;
    double x = 0;
    double y = 0;
    double width = 0;
    double height = 0;
    std::vector<double> kps = {};
};

// Return a page id for a configured network socket
int getSocket(struct sockaddr_in *server_addr);

// Find IP Address and Port of Jetson YOLOv8 runner
int getMLServer(struct sockaddr_in *server_address);

// Send datagram to provided address using provided socket
int sendReceive(int sock, struct sockaddr_in *server_addr, uchar* req_buf, int reqSize, uchar* buf, int bufSize);
// Turn current buffer into a Detection
Detection constructDetection(uchar *buf);
// Request inferencing on a frame
std::vector<Detection> remoteInference(int sock, struct sockaddr_in *server_addr, cv::Mat frame);
