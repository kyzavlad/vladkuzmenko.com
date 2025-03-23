FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY app/clip_generation/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /app/app

# Create directories
RUN mkdir -p /app/data/output /app/data/temp

# Set environment variables
ENV CGS_OUTPUT_DIR=/app/data/output
ENV CGS_TEMP_DIR=/app/data/temp
ENV CGS_FFMPEG_PATH=/usr/bin/ffmpeg
ENV CGS_WORKER_COUNT=4
ENV CGS_API_PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.clip_generation.api.api:app", "--host", "0.0.0.0", "--port", "8000"] 