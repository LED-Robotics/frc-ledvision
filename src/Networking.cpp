#include "Networking.hpp"

// Return a page id for a configured network socket
int Networking::GetSocket() {
    int sock = -1;
    int yes = 1;
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

// Send datagram to provided address using provided socket
int Networking::SendReceive(int sock, struct pollfd *fd, struct sockaddr_in *server_addr, uchar* req_buf, int reqSize, uchar* buf, int bufSize) {
    socklen_t addr_len;
    int count;
    int ret;
    addr_len = sizeof(struct sockaddr_in);

    ret = sendto(sock, req_buf, reqSize, 0, (struct sockaddr*) server_addr, addr_len);
    
    ret = poll(fd, 1, 500);
    if (ret > 0) {
            count = recvfrom(sock, buf, bufSize, 0, (struct sockaddr*) server_addr, &addr_len);
            return count;
    }
    return 0;
}
