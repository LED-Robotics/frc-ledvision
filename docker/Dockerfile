# Use Periphery image as base
FROM periphery:latest

USER root

RUN apt update && apt upgrade -y

RUN apt install protobuf-compiler openjdk-21-jdk -y

RUN usermod --login runner --move-home -d /home/runner periphery
RUN groupadd runner
RUN chown -R runner:runner /home/runner
RUN adduser runner video
RUN adduser runner sudo
RUN echo "runner:runner" | chpasswd

WORKDIR /home/runner

USER runner

RUN echo 'export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-arm64"' > /home/runner/.bashrc

RUN mkdir /home/runner/deps

WORKDIR /home/runner/deps

COPY ./opencv /home/runner/deps/opencv

USER root

RUN chown -R runner:runner ./opencv

USER runner

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64

RUN mkdir ./opencv/build

WORKDIR opencv/build/

RUN cmake ..

RUN cmake --build . --target opencv_java opencv_java_jar --parallel 4

COPY ./allwpilib /home/runner/deps/allwpilib

WORKDIR /home/runner/deps/allwpilib

USER root

RUN chown -R runner:runner /home/runner/deps/allwpilib

USER runner

RUN mkdir build-cmake

WORKDIR ./build-cmake

RUN cmake -DWITH_JAVA=ON -DWITH_JAVA_SOURCE=OFF -DWITH_WPILIB=ON -DWITH_WPIUNITS=ON -DWITH_EXAMPLES=OFF -DWITH_GUI=OFF -DWITH_TESTS=OFF -DWITH_SIMULATION_MODULES=OFF -DOPENCV_JAVA_INSTALL_DIR=/home/runner/deps/opencv/build/bin/ ..

RUN cmake --build . --parallel 4

USER root

RUN cmake --build . --target install

USER runner

COPY ./json /home/runner/deps/json

USER root

RUN chown -R runner:runner /home/runner/deps/json

WORKDIR /home/runner/deps/json/

RUN cmake .

RUN make -j4

USER root

RUN cmake --build . --target install

USER runner

WORKDIR /home/runner

# Install NodeJS
ENV NODE_VERSION=22.7.0
ENV NVM_DIR=/home/runner/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"
RUN node --version
RUN npm --version
