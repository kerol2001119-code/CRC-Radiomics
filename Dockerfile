# CRC Radiomics Docker Image

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for catboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - run the main classification pipeline
CMD ["python", "classification_pipeline.py"]
