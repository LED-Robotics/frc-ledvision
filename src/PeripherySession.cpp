#include "PeripherySession.hpp"

using namespace Networking;

PeripherySession::PeripherySession(uint32_t id, struct sockaddr_in session_addr, bool correctlyConfigured) {
  sessionId = id;
  session_address = session_addr;
  sock = GetSocket();
  valid = correctlyConfigured;
  fd.fd = sock;
  fd.events = POLLIN;
}

// Turn current buffer into a Detection
// PeripherySession::Detection PeripherySession::ConstructDetection(uchar *buf) {
//     Detection det{};
//     det.label = (int)buf[2];
//     double *temp;
//     temp = (double*)(buf+3);
//     det.x = *temp;
//     temp++;
//     det.y = *temp;
//     temp++;
//     det.width = *temp;
//     temp++;
//     det.height = *temp;
//     temp++;
//
//     uchar* kpsTemp = (uchar*)temp;
//     uchar kpsLenHigh = *kpsTemp;
//     kpsTemp++;
//     uchar kpsLenLow = *kpsTemp;
//     unsigned int kpsLen = (kpsLenHigh << 8) + kpsLenLow;
//     kpsTemp++;
//     temp = (double*)kpsTemp;
//
//     for(int i = 0; i < kpsLen; i += 8) {
//       det.kps.push_back(*temp);
//       temp++;
//     }
//
//     return det;
// }

uchar* PeripherySession::FillDetectObject(det::DetectObject *det, uchar *buf) {
  det->label = (int)buf[2];
  double *temp;
  temp = (double*)(buf+3);
  det->rect.x = *temp;
  temp++;
  det->rect.y = *temp;
  temp++;
  det->rect.width = *temp;
  temp++;
  det->rect.height = *temp;
  temp++;
  return (uchar*)temp;
}

uchar* PeripherySession::FillKeypoints(std::vector<float> &kps, uchar *buf) {
  uchar kpsLenHigh = *buf;
  buf++;
  uchar kpsLenLow = *buf;
  unsigned int kpsLen = (kpsLenHigh << 8) + kpsLenLow;
  buf++;
  float *temp = (float*)buf;

  for(int i = 0; i < kpsLen; i += 8) {
    kps.push_back(*temp);
    temp++;
  }
  return (uchar*)temp;
}

det::DetectObject PeripherySession::ConstructDetectObject(uchar *buf) {
  det::DetectObject det{};
  FillDetectObject(&det, buf);
  return det;
}

det::PoseObject PeripherySession::ConstructPoseObject(uchar *buf) {
  det::PoseObject det{};
  uchar *temp = FillDetectObject((det::DetectObject*)&det, buf);
  FillKeypoints(det.kps, temp);
  return det;
}

uint32_t PeripherySession::GetID() {
  return sessionId;
}

// Request inferencing on a frame
// std::vector<PeripherySession::Detection> PeripherySession::RunInference(cv::Mat frame) {
//     // Create message header buffer
//     size_t headerSize = sizeof(UdpSignature) + sizeof(InferenceSignature) + 4;
//     uchar header[headerSize];
//     memcpy(&header[0], UdpSignature, sizeof(UdpSignature));
//     memcpy(&header[sizeof(UdpSignature)], InferenceSignature, sizeof(InferenceSignature));
//     memcpy(&header[sizeof(UdpSignature) + sizeof(InferenceSignature)], &sessionId, 4);
//     memcpy(request, header, headerSize);
//     /*std::cout << "Session ID: " << (int)sessionId << std::endl;*/
//
//     // Chunk our frame into manageable pieces 
//     const int MaxChunk = MaxDatagram - sizeof(header) - 1;  // extra config byte after header
//     std::vector<uchar> frameVec;
//     cv::imencode(".jpg", frame, frameVec);
//     // CHUNK CHUNK CHUNK CHUNK
//     const int vectorSize = frameVec.size();
//     uchar* rawVector = frameVec.data();
//     int totalChunks = ceil((double)vectorSize / (double)MaxChunk);
//     int result = 0;
//     for(int i = 0; i < totalChunks; i++) {
//         int offset = (i * MaxChunk);
//         bool lastChunk = offset + MaxChunk >= vectorSize;
//         int size = lastChunk ? vectorSize - offset : MaxChunk;
//         memcpy(request + sizeof(header) + 1, rawVector + offset, size);
//         request[sizeof(header)] = lastChunk;
//         result = SendReceive(sock, &fd, &session_address, request, sizeof(header) + 1 + size, response, sizeof(response));
//     }
//
//     if(result && !memcmp(header, response, sizeof(header))) {
//
//         uchar sizeHigh = response[sizeof(header)];
//         uchar sizeLow = response[sizeof(header) + 1];
//         unsigned int size = (sizeHigh << 8) + sizeLow;
//         if(size && size < sizeof(response)) {   // Valid data is present
//             uchar detectionsHigh = response[sizeof(header)+2];
//             uchar detectionsLow = response[sizeof(header) + 3];
//             unsigned int totalDetections = (detectionsHigh << 8) + detectionsLow;
//             /*std::cout << "Detections: " << totalDetections << std::endl;*/
//             if(totalDetections) {
//
//
//               uchar *start = response + sizeof(header) + 4;
//               std::vector<Detection> detections;
//               for(int i = 0; i < totalDetections; i++) {
//                 auto current = ConstructDetection(start);
//                 detections.push_back(current);
//
//                 uchar lenHigh = start[0];
//                 uchar lenLow = start[1];
//                 unsigned int len = (lenHigh << 8) + lenLow;
//                 start += len;
//               }
//               return detections;
//             }
//         }
//     }
//     return {};
// }
