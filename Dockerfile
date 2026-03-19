FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git python3-pip cmake gcc curl iputils-ping

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY source /app/storage_violation
COPY weights /app/weights
COPY main.py /app/main.py

CMD ["python3", "main.py"]