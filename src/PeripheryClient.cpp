#include "PeripheryClient.h"

using namespace Networking;

PeripheryClient::PeripheryClient() {
  sock = GetSocket();
  fd.fd = sock;
  fd.events = POLLIN;
}

// Find IP Address and Port of Periphery server
int PeripheryClient::GetCommandSocket() {
    uchar request[6];
    memcpy(&request, UdpSignature, 4);
    memcpy(&request[4], DiscoverSignature, 2);
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
    
    ret = setsockopt(sock, SOL_SOCKET, SO_BROADCAST, (char*)&yes, sizeof(yes));
    if (ret == -1) {
      perror("setsockopt error");
      return 0;
    }

    addr_len = sizeof(struct sockaddr_in);

    memset((void*)&broadcast_addr, 0, addr_len);
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    broadcast_addr.sin_port = htons(COMMAND_PORT);

    ret = sendto(sock, request, sizeof(request), 0, (struct sockaddr*) &broadcast_addr, addr_len);

    /*FD_ZERO(&readfd);*/
    /*FD_SET(sock, &readfd);*/
    /*ret = select(sock + 1, &readfd, NULL, NULL, &timeout);*/
    ret = poll(&fd, 1, 1000);
    if (!ret) return 0;
    while(memcmp(buffer, request, sizeof(request))) {
      /*if (FD_ISSET(sock, &readfd)) {*/
        clientConnected = true;
        count = recvfrom(sock, buffer, 1024, 0, (struct sockaddr*)&server_addr, &addr_len);
        server_address.sin_family = server_addr.sin_family;
        server_address.sin_addr = server_addr.sin_addr;
        server_address.sin_port = server_addr.sin_port;
      /*}*/
    } 

    std::cout << "Server address is " << inet_ntoa(server_addr.sin_addr) << ':' << htons(server_addr.sin_port) << std::endl;
    ret = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));
    if (ret == -1) {
      perror("setsockopt error");
      return 0;
    }

    return 1;
}

std::string PeripheryClient::GetAvailableModels() {
  // Create message header buffer
  uchar header[sizeof(UdpSignature) + sizeof(ModelListSignature)];
  size_t headerSize = sizeof(header);
  memcpy(&header[0], UdpSignature, sizeof(UdpSignature));
  memcpy(&header[sizeof(UdpSignature)], ModelListSignature, sizeof(ModelListSignature));
  memcpy(request, header, headerSize);

  int bytes = SendReceive(sock, &fd, &server_address, request, headerSize, response, sizeof(response));
  if(!bytes) clientConnected = false;

  if(!memcmp(header, response, headerSize)) {
      
    uchar sizeHigh = response[headerSize];
    uchar sizeLow = response[headerSize + 1];
    unsigned int size = (sizeHigh << 8) + sizeLow;

    if(size && size < sizeof(response)) {   // Valid data is present
      std::string models = std::string{(char*)response + headerSize + 2, size};
      return models;
    }
  }
  return "NONE";
}

bool PeripheryClient::SwitchModel(std::string modelName) {
  // Create message header buffer
  uchar header[sizeof(UdpSignature) + sizeof(SelectModelSignature)];
  size_t headerSize = sizeof(header);
  memcpy(&header[0], UdpSignature, sizeof(UdpSignature));
  memcpy(&header[sizeof(UdpSignature)], SelectModelSignature, sizeof(SelectModelSignature));
  memcpy(request, header, headerSize);

  int payloadSize = modelName.length();
  char nameBuf[payloadSize];
  strcpy(nameBuf, modelName.c_str());
  memcpy(request + headerSize, nameBuf, payloadSize);

  int bytes = SendReceive(sock, &fd, &server_address, request, headerSize + payloadSize, response, sizeof(response));
  if(!bytes) clientConnected = false;

  if(!memcmp(header, response, headerSize)) {
    bool success = response[headerSize];
    return success;
  }
  return false;
}


PeripherySession PeripheryClient::CreateInferenceSession() {
  // Create message header buffer
  uchar header[sizeof(UdpSignature) + sizeof(StartSessionSignature)];
  size_t headerSize = sizeof(header);
  memcpy(&header[0], UdpSignature, sizeof(UdpSignature));
  memcpy(&header[sizeof(UdpSignature)], StartSessionSignature, sizeof(StartSessionSignature));
  memcpy(request, header, headerSize);

  int bytes = SendReceive(sock, &fd, &server_address, request, headerSize, response, sizeof(response));
  if(!bytes) clientConnected = false;

  struct sockaddr_in session_addr;
  if(!memcmp(header, response, headerSize)) {
    uint32_t address;
    memcpy(&address, response + headerSize, 4);

    uchar portHigh = response[headerSize+4];
    uchar portLow = response[headerSize + 5];
    unsigned int port = (portHigh << 8) + portLow;
    struct sockaddr_in session_addr;
    session_addr.sin_family = AF_INET;
    session_addr.sin_addr.s_addr = server_address.sin_addr.s_addr;
    session_addr.sin_port = htons(port);
    std::cout << "Session address is " << inet_ntoa(session_addr.sin_addr) << ':' << htons(session_addr.sin_port) << std::endl;
    uint32_t id;
    memcpy(&id, response + headerSize + 6, 4);
    return PeripherySession{id, session_addr};
  }
  return PeripherySession{0, session_addr, false};
  /*return false;*/
}

bool PeripheryClient::SessionAvailable(uint32_t id) {
  // Create message header buffer
  uchar header[sizeof(UdpSignature) + sizeof(QuerySessionSignature)];
  size_t headerSize = sizeof(header);
  memcpy(&header[0], UdpSignature, sizeof(UdpSignature));
  memcpy(&header[sizeof(UdpSignature)], QuerySessionSignature, sizeof(QuerySessionSignature));
  memcpy(request, header, headerSize);

  memcpy(request + headerSize, &id, 4);

  int bytes = SendReceive(sock, &fd, &server_address, request, headerSize + 4, response, sizeof(response));
  if(!bytes) clientConnected = false;

  if(!memcmp(header, response, headerSize)) {
    bool alive = response[headerSize];
    return alive;
  }
  return false;
}

bool PeripheryClient::GetClientConnected() {
  return clientConnected;
}
