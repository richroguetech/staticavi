# Start from an NVIDIA CUDA base image with CUDA 11.3.1
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Set timezone environment variable
ENV TZ=UTC \
    CC=/usr/bin/gcc-11 \
    CXX=/usr/bin/g++-11 \
    PYTHONUNBUFFERED=1

# Combine commands to reduce layers and cleanup apt cache to reduce image size
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y software-properties-common git ca-certificates libgl1 --no-install-recommends && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --fix-missing gcc-11 g++-11 python3.9 python3.9-venv python3.9-dev ffmpeg libsndfile1 --no-install-recommends && \
    python3.9 -m ensurepip --upgrade && \
    python3.9 -m pip install --upgrade pip && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container to /app
# WORKDIR /app

# Copy the requirements.txt file from the host machine's current directory
# to the working directory in the container
COPY requirements.txt /

# Use pip to install the Python dependencies from requirements.txt
# and install additional Python packages in a single RUN command
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt && \
    pip install git+https://github.com/elliottzheng/batch-face.git@master && \
    python3.9 -m pip install --no-cache-dir fairseq moviepy boto3 cog numpy==1.23.4 opencv-python==4.6.0.66 && \
    python3.9 -m pip install --extra-index-url https://download.pytorch.org/whl/cu110 torch==1.12.1+cu116 torchvision==0.13.1+cu116

# Copy the rest of the current directory contents into the container at /app
#COPY . /app

# Change permissions of the app directory
#RUN chmod 777 /app

# Set the default command to run when the container starts
#CMD ["python", "app.py", "demo.mp4"]
