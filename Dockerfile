FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python-headless==4.8.1.78 \
    ultralytics==8.0.196 \
    transformers==4.35.0 \
    sentence-transformers==2.2.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000