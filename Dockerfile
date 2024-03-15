FROM python:3.8-slim-buster
FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04
# EXPOSE 7017 

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	git \
	python3 \
	python3-pip \
    && rm -rf /var/lib/apt/lists/*
# Fixes libGL bug: https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the requirements file

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip


# # Copy the rest of the application code
COPY . .

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# # TODO fix error on docker build
# RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37

# RUN pip install cmake --upgrade

# # RUN make python-install

