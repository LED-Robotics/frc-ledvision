#pragma once

#include <arpa/inet.h>
#include <linux/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <iostream>
#include <stdio.h>
#include <sys/types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Networking {
  // Return a page id for a configured network socket
  int GetSocket();

  // Send datagram to provided address using provided socket
  int SendReceive(int sock, struct sockaddr_in *server_addr, uchar* req_buf, int reqSize, uchar* buf, int bufSize);

  constexpr uchar UdpSignature[4] = {0x5b, 0x20, 0xc4, 0x10};
}


