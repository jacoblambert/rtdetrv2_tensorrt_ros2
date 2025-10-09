FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ARG ROS_DISTRO=humble

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    ROS_DISTRO=${ROS_DISTRO}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Locale setup matches the previous workflow
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Keep the ROS 2 repository steps exactly as before
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository universe && apt-get update && apt-get install -y curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble base and vision packages
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-vision-opencv \
    ros-humble-cv-bridge \
    ros-humble-rosbag2-storage-mcap
    
# Tooling for building the workspace
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-argcomplete
# TensorRT repository is already trusted in CUDA images; add only the list entry
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install pinned TensorRT components to avoid undesired CUDA upgrades
ARG TENSORRT_VERSION=10.13.3.9-1+cuda12.9

RUN apt-get update && apt-get install -y \
    libnvinfer10=${TENSORRT_VERSION} \
    libnvinfer-headers-dev=${TENSORRT_VERSION} \
    libnvinfer-dev=${TENSORRT_VERSION} \
    libnvinfer-vc-plugin10=${TENSORRT_VERSION} \
    python3-libnvinfer-lean=${TENSORRT_VERSION} \
    python3-libnvinfer-dispatch=${TENSORRT_VERSION} \
    libnvinfer-lean-dev=${TENSORRT_VERSION} \
    libnvinfer-dispatch-dev=${TENSORRT_VERSION} \
    libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-vc-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-plugin10=${TENSORRT_VERSION} \
    libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-plugin-dev=${TENSORRT_VERSION} \
    libnvonnxparsers10=${TENSORRT_VERSION} \
    libnvonnxparsers-dev=${TENSORRT_VERSION} \
    python3-libnvinfer=${TENSORRT_VERSION} \
    python3-libnvinfer-dev=${TENSORRT_VERSION} \
    libnvinfer-vc-plugin10=${TENSORRT_VERSION} \
    libnvinfer-lean10=${TENSORRT_VERSION} \
    libnvinfer-dispatch10=${TENSORRT_VERSION} \
    libnvinfer-bin=${TENSORRT_VERSION} \
    libnvinfer-lean-dev=${TENSORRT_VERSION} \
    libnvinfer-vc-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-dispatch-dev=${TENSORRT_VERSION} \
    libnvinfer-samples=${TENSORRT_VERSION} \
    libnvinfer-win-builder-resource10=${TENSORRT_VERSION} \
    libnvinfer-headers-python-plugin-dev=${TENSORRT_VERSION} \
    tensorrt=${TENSORRT_VERSION}

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# COPY . /workspace/

CMD ["/bin/bash"]
