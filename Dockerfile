FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install \
        torch \
        numpy \
        faster-whisper

COPY app/ ./app/

CMD ["/opt/venv/bin/python", "app/main.py"]