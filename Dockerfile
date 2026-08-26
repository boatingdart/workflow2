FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
        torch \
        numpy \
        faster-whisper

COPY app/ ./app/

CMD ["python", "app/main.py"]
