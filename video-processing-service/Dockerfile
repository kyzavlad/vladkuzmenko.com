FROM python:3.10-slim

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Video Processing Service with Subtitle Generation and B-Roll Insertion"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-subtitle.txt requirements-broll.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-subtitle.txt && \
    pip install --no-cache-dir -r requirements-broll.txt

# Download NLTK data
RUN python -m nltk.downloader punkt wordnet stopwords

# Download SpaCy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/video-processing /app/outputs /app/temp

# Expose port
EXPOSE 8000

# Set entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 