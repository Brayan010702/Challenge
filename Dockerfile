# syntax=docker/dockerfile:1.2
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY challenge/ ./challenge/

# Expose port
EXPOSE 8080

# Run with Cloud Run's expected port
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]