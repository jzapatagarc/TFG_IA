# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
# Ensures that the Python output is sent straight to the terminal without being buffered
ENV PYTHONUNBUFFERED=1  
# Avoids prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive  

# Install system dependencies
RUN apt-get update && apt-get install -y \
# Python package installer
    python3-pip \ 
# Python development files for building Python C extensions 
    python3-dev \  
# Essential libraries for building software
    build-essential \ 
# OpenGL utility library 
    libglu1-mesa \  
# Simplified Wrapper and Interface Generator
    swig \  
# Cleans up the local repository of retrieved package files
    && apt-get clean \  
# Removes the list of available packages to save space
    && rm -rf /var/lib/apt/lists/*  

# Copy the requirements file into the container
COPY requirements.txt /tmp/

# Install Python dependencies
# Upgrade pip to the latest version
RUN pip3 install --upgrade pip \
    && pip3 install -r /tmp/requirements.txt  

# Install additional dependencies for gym[box2d]
RUN pip3 install gym[box2d]

# Create and set the working directory inside the container
# Install the dependencies from requirements.txt
# Create a directory for the application
RUN mkdir /app
# Set this directory as the working directory  
WORKDIR /app  

# Copy the current directory contents into the container at /app
COPY . /app

# Define volumes
VOLUME /app/local_models
VOLUME /app/global_models

# Set this directory as the working directory
CMD ["python3", "agent.py"]

