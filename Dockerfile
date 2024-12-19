# Start from an NVIDIA CUDA base image with CUDA 11.3.1
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set timezone to UTC
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add PPA for GCC-11 and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common build-essential cmake && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get install -y --no-install-recommends \
        gcc-11 g++-11 git ca-certificates libgl1 ffmpeg libsndfile1 libboost-all-dev libx11-dev libxcb1 \
        libxcb-xinerama0 libx11-6 libxrender1 libxtst6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure Universe repository is enabled and install libx11-dev
RUN add-apt-repository universe && \
    apt-get update && \
    apt-get install -y libx11-dev && \
    apt-get clean

# Add PPA for Python 3.10 and install Python dependencies
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN python3.10 -m ensurepip --upgrade && \
    python3.10 -m pip install "pip<24.1" && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory and copy requirements.txt before the app
WORKDIR /app
COPY requirements.txt /app/

# Install Python dependencies
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Install additional Python packages
RUN python3.10 -m pip install git+https://github.com/elliottzheng/batch-face.git@master
RUN python3.10 -m pip install --no-cache-dir moviepy boto3 cog

# Install onnxruntime-gpu for CUDA 12.2
RUN python3.10 -m pip install onnxruntime-gpu==1.16.0

RUN apt-get purge -y build-essential cmake gcc-11 g++-11 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the app
COPY . /app
RUN chmod 777 /app

# Set PYTHONUNBUFFERED to ensure logs are not buffered
ENV PYTHONUNBUFFERED=1

# Set the command to run the app
#CMD ["python", "app.py", "demo.mp4"]