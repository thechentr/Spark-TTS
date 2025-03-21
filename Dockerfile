FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel


RUN apt update && apt install -y \
    git \
    nano \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
