# Dockerfile snippet
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]