# Use Periphery image as base
FROM ledvision-dev:latest

RUN git clone https://github.com/LED-Robotics/frc-ledvision.git

USER root

RUN chown -R runner:runner /home/runner/frc-ledvision

USER runner

WORKDIR /home/runner/frc-ledvision/

RUN git switch local-ml

RUN mkdir build

WORKDIR ./build

RUN cmake ..

RUN make -j4
