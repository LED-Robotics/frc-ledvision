#include "networking.h"

// Signature for Jetson comms
const uchar udpSignature[] = {0x5b, 0x20, 0xc4, 0x10};

// Max datagram length for image stream
const int MaxDatagram = 32768;
uchar request[MaxDatagram];
uchar response[32768];

// Return a page id for a configured network socket
int getSocket(struct sockaddr_in *server_addr) {
    int sock = -1;
    int yes = 1;
    std::cout << "Server address is " << inet_ntoa(server_addr->sin_addr) << ':' << htons(server_addr->sin_port) << std::endl;
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cout << "sock error" << std::endl;
    }
    int ret = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));
    if (ret == -1) {
        perror("setsockopt error");
        return 0;
    }
    return sock;
}

// Find IP Address and Port of Jetson YOLOv8 runner
int getMLServer(struct sockaddr_in *server_address) {
    const uchar discoverSignature[] = {0x8e, 0x96};
    uchar request[6];
    memcpy(&request, udpSignature, 4);
    memcpy(&request[4], discoverSignature, 2);
    int sock;
    int yes = 1;
    struct timeval timeout;
    timeout.tv_usec = 1000000;
    struct sockaddr_in broadcast_addr;
    struct sockaddr_in server_addr;
    socklen_t addr_len;
    int count;
    int ret;
    fd_set readfd;
    char buffer[100];
    
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("sock error");
        return -1;
    }
    ret = setsockopt(sock, SOL_SOCKET, SO_BROADCAST, (char*)&yes, sizeof(yes));
    if (ret == -1) {
        perror("setsockopt error");
        return 0;
    }

    addr_len = sizeof(struct sockaddr_in);

    memset((void*)&broadcast_addr, 0, addr_len);
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    broadcast_addr.sin_port = htons(5800);

    ret = sendto(sock, request, sizeof(request), 0, (struct sockaddr*) &broadcast_addr, addr_len);

    FD_ZERO(&readfd);
    FD_SET(sock, &readfd);
    ret = select(sock + 1, &readfd, NULL, NULL, &timeout);
    if (!ret) return 0;
    while(memcmp(buffer, request, sizeof(request))) {
        if (FD_ISSET(sock, &readfd)) {
            count = recvfrom(sock, buffer, 1024, 0, (struct sockaddr*)&server_addr, &addr_len);
            server_address->sin_family = server_addr.sin_family;
            server_address->sin_addr = server_addr.sin_addr;
            server_address->sin_port = server_addr.sin_port;
        }
    } 

    return 1;
}

// Send datagram to provided address using provided socket
int sendReceive(int sock, struct sockaddr_in *server_addr, uchar* req_buf, int reqSize, uchar* buf, int bufSize) {
    socklen_t addr_len;
    int count;
    int ret;
    fd_set readfd;
    addr_len = sizeof(struct sockaddr_in);
    struct timeval timeout;
    timeout.tv_usec = 500000;
    timeout.tv_sec = 0;

    ret = sendto(sock, req_buf, reqSize, 0, (struct sockaddr*) server_addr, addr_len);
    FD_ZERO(&readfd);
    FD_SET(sock, &readfd);
    ret = select(sock + 1, &readfd, NULL, NULL, &timeout);
    if (ret > 0) {
        if (FD_ISSET(sock, &readfd)) {
            count = recvfrom(sock, buf, bufSize, 0, (struct sockaddr*) server_addr, &addr_len);
        }
    }
    return 1;
}

// Turn current buffer into a Detection
Detection constructDetection(uchar *buf) {
    Detection det{};
    det.label = (int)buf[2];
    double *temp;
    temp = (double*)(buf+3);
    det.x = *temp;
    temp++;
    det.y = *temp;
    temp++;
    det.width = *temp;
    temp++;
    det.height = *temp;
    temp++;

    uchar* kpsTemp = (uchar*)temp;
    uchar kpsLenHigh = *kpsTemp;
    kpsTemp++;
    uchar kpsLenLow = *kpsTemp;
    unsigned int kpsLen = (kpsLenHigh << 8) + kpsLenLow;
    kpsTemp++;
    temp = (double*)kpsTemp;

    for(int i = 0; i < kpsLen; i += 8) {
      det.kps.push_back(*temp);
      temp++;
    }

    return det;
}

// Request inferencing on a frame
std::vector<Detection> remoteInference(int sock, struct sockaddr_in *server_addr, cv::Mat frame) {
    // Create message header buffer
    const uchar inferSignature[] = {0xe2, 0x4d};
    uchar header[sizeof(udpSignature) + sizeof(inferSignature)];
    memcpy(&header[0], udpSignature, sizeof(udpSignature));
    memcpy(&header[sizeof(udpSignature)], inferSignature, sizeof(inferSignature));
    memcpy(request, header, sizeof(header));


    // Chunk our frame into manageable pieces 
    const int MaxChunk = MaxDatagram - sizeof(header) - 1;  // extra config byte after header
    std::vector<uchar> frameVec;
    cv::imencode(".jpg", frame, frameVec);
    // CHUNK CHUNK CHUNK CHUNK
    const int vectorSize = frameVec.size();
    uchar* rawVector = frameVec.data();
    int totalChunks = ceil((double)vectorSize / (double)MaxChunk);
    for(int i = 0; i < totalChunks; i++) {
        int offset = (i * MaxChunk);
        bool lastChunk = offset + MaxChunk >= vectorSize;
        int size = lastChunk ? vectorSize - offset : MaxChunk;
        memcpy(request + sizeof(header) + 1, rawVector + offset, size);
        request[sizeof(header)] = lastChunk;
        sendReceive(sock, server_addr, request, sizeof(header) + 1 + size, response, sizeof(response));
    }

    if(!memcmp(header, response, sizeof(header))) {
        
        uchar sizeHigh = response[sizeof(header)];
        uchar sizeLow = response[sizeof(header) + 1];
        unsigned int size = (sizeHigh << 8) + sizeLow;
        if(size && size < sizeof(response)) {   // Valid data is present
            uchar detectionsHigh = response[sizeof(header)+2];
            uchar detectionsLow = response[sizeof(header) + 3];
            unsigned int totalDetections = (detectionsHigh << 8) + detectionsLow;
            /*std::cout << "Detections: " << totalDetections << std::endl;*/
            if(totalDetections) {
              

              uchar *start = response + sizeof(header) + 4;
              std::vector<Detection> detections;
              for(int i = 0; i < totalDetections; i++) {
                auto current = constructDetection(start);
                detections.push_back(current);

                uchar lenHigh = start[0];
                uchar lenLow = start[1];
                unsigned int len = (lenHigh << 8) + lenLow;
                start += len;
                /*std::cout << "Detection packet length: " << len << std::endl;*/
                /*std::cout << "Index: " << i << std::endl;*/
                /*std::cout << "Label: " << (int)current.label << std::endl;*/
                /*std::cout << "X: " << current.x << std::endl;*/
                /*std::cout << "Y: " << current.y << std::endl;*/
                /*std::cout << "Width: " << current.width << std::endl;*/
                /*std::cout << "Height: " << current.height << std::endl;*/
                /*std::cout << std::endl;*/
              }
              /*std::cout << "Detections created: " << detections.size() << std::endl;*/
              /*std::cout << std::endl;*/
              /*std::cout << std::endl;*/
              return detections;
            }
        }
    }
    return {};
}
