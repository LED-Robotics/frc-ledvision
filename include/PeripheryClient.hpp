#pragma once

#include <arpa/inet.h>
#include <linux/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Networking.hpp"
#include "PeripherySession.hpp"

class PeripheryClient {
  public:
    PeripheryClient();

    // UDP Broadcast to find command socket
    int GetCommandSocket();

    // Get available models on ML server
    std::string GetAvailableModels();
    
    // Change active model on server
    bool SwitchModel(std::string modelName);
    
    // Create inference session
    PeripherySession CreateInferenceSession();
    
    // Check if session is alive
    bool SessionAvailable(uint32_t id);

    // Check if the client is connected
    bool GetClientConnected();

  private:
    struct sockaddr_in server_address;
    int sock = -1;
    struct pollfd fd;
    bool clientConnected = false;

    std::vector<PeripherySession> sessions;

    const int COMMAND_PORT = 5800;

    // Max datagram length for image stream
    static const int MaxDatagram = 1024;
    uchar request[PeripheryClient::MaxDatagram];
    uchar response[PeripheryClient::MaxDatagram];
  };
