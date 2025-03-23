FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app/

# Create storage directory
RUN mkdir -p /app/storage/transcriptions

# Create user and set permissions
RUN useradd -m -s /bin/bash app_user \
    && chown -R app_user:app_user /app/storage

# Switch to non-root user
USER app_user

# Expose API port
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 