# Start from an NVIDIA CUDA base image with CUDA 11.3.1
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Install software properties common (allows add-apt-repository)
RUN apt-get update && \
    apt-get install -y software-properties-common --no-install-recommends \
	git \
	ca-certificates \
    libgl1

# Add Ubuntu Toolchain Test PPA for newer gcc versions
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

# Install gcc-11 and g++-11 and other essential build tools
RUN apt-get update && \
    apt-get install -y --fix-missing gcc-11 g++-11

# Set environment variables to use gcc-11 and g++-11 as the default compilers
ENV CC /usr/bin/gcc-11
ENV CXX /usr/bin/g++-11

# Add deadsnakes PPA for newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.9 and the development headers
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-venv python3.9-dev

# install deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libsndfile1

# Ensure that pip is installed and up to date
# Upgrade pip and set Python 3.9 as the default Python version
# Upgrade pip and set Python 3.9 as the default Python version
RUN python3.9 -m ensurepip --upgrade && \
    python3.9 -m pip install --upgrade pip && \
    ln -s /usr/bin/python3.9 /usr/bin/python

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from the host machine's current directory
# to the working directory in the container
COPY requirements.txt /app/

# Use pip to install the Python dependencies from requirements.txt
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt

RUN pip install git+https://github.com/elliottzheng/batch-face.git@master

# Install onnxruntime-gpu and moviepy separately to ensure clear error messages
RUN python3.9 -m pip install --no-cache-dir fairseq moviepy boto3 cog

# Install numpy, opencv-python, torch, and torchvision with CUDA 11.6 support
RUN python3.9 -m pip install numpy==1.23.4 opencv-python==4.6.0.66
RUN python3.9 -m pip install --extra-index-url https://download.pytorch.org/whl/cu110 torch==1.12.1+cu116 torchvision==0.13.1+cu116
# Copy the rest of the current directory contents into the container at /app
COPY . /app

RUN chmod 777 /app

ENV PYTHONUNBUFFERED=1

# Set the default command to run when the container starts
#CMD ["python3.9", "app.py", "e0781268-f002-450f-8269-3e1d85902262/78a7b386-c630-45b4-b4b8-504610b99b2a"]

CMD ["python", "app.py", "demo.mp4"]