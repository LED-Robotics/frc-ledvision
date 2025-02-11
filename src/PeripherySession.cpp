#include "PeripherySession.h"

PeripherySession::PeripherySession() {
  
}

// Turn current buffer into a Detection
PeripherySession::Detection PeripherySession::ConstructDetection(uchar *buf) {
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
std::vector<PeripherySession::Detection> PeripherySession::RunInference(cv::Mat frame) {
    // Create message header buffer
    const uchar inferSignature[] = {0xe2, 0x4d};
    uchar header[sizeof(Networking::UdpSignature) + sizeof(inferSignature)];
    memcpy(&header[0], Networking::UdpSignature, sizeof(Networking::UdpSignature));
    memcpy(&header[sizeof(Networking::UdpSignature)], inferSignature, sizeof(inferSignature));
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
        Networking::SendReceive(sock, &session_address, request, sizeof(header) + 1 + size, response, sizeof(response));
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
                auto current = ConstructDetection(start);
                detections.push_back(current);

                uchar lenHigh = start[0];
                uchar lenLow = start[1];
                unsigned int len = (lenHigh << 8) + lenLow;
                start += len;
              }
              return detections;
            }
        }
    }
    return {};
}
