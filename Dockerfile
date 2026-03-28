FROM python:3.11-slim

WORKDIR /app

# Install ALL required system libraries in one layer
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first to prevent version conflicts
RUN pip install --no-cache-dir numpy==1.24.4

# Install opencv headless
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]