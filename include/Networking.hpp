#pragma once

#include <arpa/inet.h>
#include <iostream>
#include <linux/in.h>
#include <poll.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Networking {
// Return a page id for a configured network socket
int GetSocket();

// Send datagram to provided address using provided socket
int SendReceive(int sock, struct pollfd *fd, struct sockaddr_in *server_addr,
                uchar *req_buf, int reqSize, uchar *buf, int bufSize);

constexpr uchar UdpSignature[4] = {0x5b, 0x20, 0xc4, 0x10};

constexpr uchar DiscoverSignature[2] = {0x8e, 0x96};

constexpr uchar ModelListSignature[2] = {0x87, 0x11};

constexpr uchar SelectModelSignature[2] = {0x84, 0x7a};

constexpr uchar StartSessionSignature[2] = {0x5a, 0x55};

constexpr uchar QuerySessionSignature[2] = {0x76, 0x03};

constexpr uchar EndSessionSignature[2] = {0x91, 0x6e};

constexpr uchar InferenceSignature[2] = {0xe2, 0x4d};
} // namespace Networking
