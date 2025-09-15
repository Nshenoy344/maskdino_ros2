FROM nvcr.io/nvidia/pytorch:23.10-py3

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y 

# Install necessary tools and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    build-essential \
    software-properties-common \
    locales\
    && rm -rf /var/lib/apt/lists/*

# Setup locale
RUN locale-gen en_US en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN add-apt-repository universe -y

RUN apt-get update 

# Set up the ROS2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    tee /etc/apt/sources.list.d/ros2.list

# Install ROS2 HUMBLE
RUN apt-get update &&  apt-get upgrade -y && apt-get install -y \
    ros-humble-desktop \
    ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*


# Initialize rosdep
RUN rosdep init && \
    rosdep update

RUN pip3 install empy 

# Create the ros2 workspace
RUN mkdir -p /root/ros2_ws/src

# # Initialize the ros2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd /root/ros2_ws/src "

COPY maskdino_ros_pkg/ /root/ros2_ws/src/maskdino_ros_pkg

WORKDIR /root/ros2_ws

# # Build the ros2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd /root/ros2_ws && colcon build"

# # Source the ros2 workspace setup script
RUN echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc



WORKDIR /root/ros2_ws/src/maskdino

# install maskDINO
COPY maskdino_ros_pkg/maskdino_ros_pkg/maskdino/MaskDINO /root/ros2_ws/src/maskdino/MaskDINO

RUN python3 -m pip install -r MaskDINO/requirements.txt && \
    python3 -m pip install opencv-python==4.7.0.68 fiftyone
RUN python3 -m pip install pillow 
RUN python3 -m pip install torchvision==0.18.0

# install detectron2
COPY maskdino_ros_pkg/maskdino_ros_pkg/maskdino/detectron2 /root/ros2_ws/src/maskdino/detectron2

RUN python3 -m pip install -e detectron2

RUN apt purge -y  libhwloc-dev libhwloc-plugins 

# startup script to install MSDeformAttn pixel decoder
COPY docker/nvidia_entrypoint.sh /opt/nvidia/nvidia_entrypoint.sh

# copy custom MaskDINO files
COPY configs/ /root/ros2_ws/src/maskdino/MaskDINO/configs/

WORKDIR /root/ros2_ws


RUN apt update && apt install -y \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-tf2-ros \
    ros-humble-tf2 

