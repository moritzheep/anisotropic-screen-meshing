FROM nvcr.io/nvidia/pytorch:23.03-py3
#FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install Dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
# For nvdiffrast
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \
# For cmake
    libssl-dev \
# For pmp
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libboost-all-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
# Misc
    wget \
    build-essential \
    python3

# Set some variables for nvdiffrast
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

RUN pip install --upgrade pip
RUN pip install ninja imageio imageio-ffmpeg

# Install CMake
WORKDIR /cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.3/cmake-3.30.3.tar.gz && tar zxvf cmake-3.30.3.tar.gz

WORKDIR /cmake/cmake-3.30.3
RUN ./bootstrap -- -DCMAKE_BUILD_TYPE:STRING=Release && make -j$(nproc) && make install

# Copy the repo
WORKDIR /anisotropic-screen-meshing
COPY ./CMakeLists.txt /anisotropic-screen-meshing/CMakeLists.txt

WORKDIR /anisotropic-screen-meshing/external
COPY ./external /anisotropic-screen-meshing/external

WORKDIR /anisotropic-screen-meshing/src
COPY ./src /anisotropic-screen-meshing/src

# WORKDIR /
# RUN git clone --recursive https://github.com/moritzheep/anisotropic-screen-meshing.git

# Enable OpenEXR support
ENV OPENCV_IO_ENABLE_OPENEXR=1

# Build
WORKDIR /anisotropic-screen-meshing/build
RUN cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` .. && make -j$(nproc)

WORKDIR /mnt

ENTRYPOINT ["/anisotropic-screen-meshing/build/src/main"]