# Clip Generation Service API Documentation

## Overview

The Clip Generation Service provides a RESTful API for video clip generation, processing, and management. This documentation covers all available endpoints, request/response formats, and usage examples.

## Base URL

```
https://api.clip-generation-service.com/v1
```

## Authentication

All API requests require authentication using JWT tokens. Include the token in the Authorization header:

```bash
Authorization: Bearer <your_jwt_token>
```

## Rate Limiting

- Standard tier: 100 requests per minute
- Pro tier: 1000 requests per minute
- Enterprise tier: Custom limits

## Endpoints

### Jobs

#### Create Job

```http
POST /jobs
```

Request body:
```json
{
  "video_url": "https://example.com/video.mp4",
  "start_time": "00:00:10",
  "end_time": "00:00:30",
  "output_format": "mp4",
  "quality": "high",
  "enhancement_options": {
    "stabilization": true,
    "color_correction": true,
    "audio_enhancement": true
  }
}
```

Response:
```json
{
  "job_id": "job_123",
  "status": "pending",
  "created_at": "2024-03-20T10:00:00Z",
  "estimated_duration": 120
}
```

#### Get Job Status

```http
GET /jobs/{job_id}
```

Response:
```json
{
  "job_id": "job_123",
  "status": "processing",
  "progress": 45,
  "created_at": "2024-03-20T10:00:00Z",
  "updated_at": "2024-03-20T10:01:00Z",
  "output_url": "https://storage.example.com/output.mp4",
  "error": null
}
```

### Quality Control

#### Analyze Job Quality

```http
POST /quality/analyze/{job_id}
```

Response:
```json
{
  "job_id": "job_123",
  "video_metrics": {
    "resolution": "1920x1080",
    "frame_rate": 30,
    "bitrate": "5000kbps",
    "color_accuracy": 0.95,
    "motion_smoothness": 0.92,
    "artifact_level": 0.05
  },
  "audio_metrics": {
    "loudness": -14.0,
    "dynamic_range": 12.0,
    "frequency_response": 0.98,
    "distortion": 0.02,
    "noise_level": 0.03
  },
  "overall_score": 0.92,
  "timestamp": "2024-03-20T10:02:00Z"
}
```

### A/B Testing

#### Create A/B Test

```http
POST /quality/ab-test/{job_id}
```

Request body:
```json
{
  "variants": [
    {
      "name": "control",
      "parameters": {
        "quality": "high",
        "enhancement_level": "standard"
      }
    },
    {
      "name": "treatment",
      "parameters": {
        "quality": "high",
        "enhancement_level": "aggressive"
      }
    }
  ],
  "metrics": ["quality_score", "processing_time", "file_size"]
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid video URL format",
    "details": {
      "field": "video_url",
      "reason": "URL must be HTTPS"
    }
  }
}
```

### Common Error Codes

- `INVALID_REQUEST`: Bad request parameters
- `UNAUTHORIZED`: Invalid or missing authentication
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error

## Webhooks

### Webhook Events

- `job.created`
- `job.processing`
- `job.completed`
- `job.failed`
- `quality.analysis.completed`
- `ab_test.completed`

### Webhook Payload Format

```json
{
  "event": "job.completed",
  "timestamp": "2024-03-20T10:05:00Z",
  "data": {
    "job_id": "job_123",
    "status": "completed",
    "output_url": "https://storage.example.com/output.mp4",
    "processing_time": 120
  }
}
```

### Webhook Security

Webhooks are signed using HMAC SHA-256. Include the signature in the `X-Webhook-Signature` header:

```bash
X-Webhook-Signature: sha256=<signature>
```

## SDK Examples

### Python

```python
from clip_generation import ClipGenerationClient

client = ClipGenerationClient(api_key="your_api_key")

# Create a job
job = client.create_job(
    video_url="https://example.com/video.mp4",
    start_time="00:00:10",
    end_time="00:00:30",
    output_format="mp4",
    quality="high"
)

# Get job status
status = client.get_job_status(job.job_id)

# Analyze quality
quality = client.analyze_quality(job.job_id)
```

### JavaScript

```javascript
const { ClipGenerationClient } = require('clip-generation');

const client = new ClipGenerationClient({
  apiKey: 'your_api_key'
});

// Create a job
const job = await client.createJob({
  videoUrl: 'https://example.com/video.mp4',
  startTime: '00:00:10',
  endTime: '00:00:30',
  outputFormat: 'mp4',
  quality: 'high'
});

// Get job status
const status = await client.getJobStatus(job.jobId);

// Analyze quality
const quality = await client.analyzeQuality(job.jobId);
```

## Best Practices

1. **Error Handling**
   - Implement exponential backoff for retries
   - Handle rate limiting gracefully
   - Log all API interactions

2. **Performance**
   - Use webhooks for status updates
   - Implement caching for frequently accessed data
   - Batch operations when possible

3. **Security**
   - Rotate API keys regularly
   - Validate webhook signatures
   - Use HTTPS for all requests

4. **Monitoring**
   - Track API usage and errors
   - Monitor response times
   - Set up alerts for error rates 