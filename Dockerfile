# Base slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies only once
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install CPU-only PyTorch + minimal heavy packages
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu torchvision==0.16.0+cpu \
    && pip install --no-cache-dir opencv-python-headless==4.8.1.78 ultralytics==8.0.196 \
    && pip install --no-cache-dir transformers==4.35.0 sentence-transformers==2.2.2

# Install the rest of your lightweight requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]