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

#include "Networking.h"
#include "PeripherySession.h"

class PeripheryClient {
  public:
    PeripheryClient();

    // UDP Broadcast to find command socket
    int GetCommandSocket();

    std::string GetAvailableModels();

    bool SwitchModels(std::string modelName);

    PeripherySession* CreateInferenceSession();

  private:
    struct sockaddr_in server_address;
    int sock = -1;
    int timeoutfd = -1;
    std::vector<PeripherySession> sessions;

    const int COMMAND_PORT = 5800;
  };
