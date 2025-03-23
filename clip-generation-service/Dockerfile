# Use Python 3.8 slim image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage/uploads /app/storage/outputs

# Set permissions
RUN chmod -R 755 /app/storage

# Expose port
EXPOSE 8000

# Create non-root user
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 