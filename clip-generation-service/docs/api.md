# API Documentation

## Authentication

All API endpoints require authentication using either JWT tokens or API keys.

### JWT Authentication
Include the JWT token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

### API Key Authentication
Include the API key in the X-API-Key header:
```
X-API-Key: <your_api_key>
```

## Endpoints

### Jobs

#### Create Job
```http
POST /jobs
Content-Type: application/json

{
    "job_type": "VIDEO_EDIT",
    "input_data": {
        "input_path": "path/to/video.mp4",
        "target_duration": 30.0,
        "target_width": 1080,
        "target_height": 1920,
        "target_lufs": -14.0
    }
}
```

Response:
```json
{
    "job_id": "job_123",
    "status": "PENDING",
    "progress": 0.0,
    "created_at": "2024-03-21T10:00:00Z"
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
    "status": "PROCESSING",
    "progress": 45.5,
    "input_data": {
        "input_path": "path/to/video.mp4",
        "target_duration": 30.0
    },
    "output_data": {
        "output_path": "path/to/output.mp4",
        "duration": 30.0
    },
    "error_message": null,
    "created_at": "2024-03-21T10:00:00Z",
    "updated_at": "2024-03-21T10:01:00Z",
    "completed_at": null
}
```

#### List User Jobs
```http
GET /jobs?status=PROCESSING&limit=10&offset=0
```

Response:
```json
{
    "total": 100,
    "offset": 0,
    "limit": 10,
    "jobs": [
        {
            "job_id": "job_123",
            "status": "PROCESSING",
            "progress": 45.5,
            "created_at": "2024-03-21T10:00:00Z"
        }
    ]
}
```

#### Cancel Job
```http
DELETE /jobs/{job_id}
```

Response:
```json
{
    "job_id": "job_123",
    "status": "CANCELLED",
    "progress": 45.5,
    "updated_at": "2024-03-21T10:02:00Z"
}
```

### WebSocket API

Connect to the WebSocket endpoint:
```
ws://localhost:8000/ws/{user_id}
```

#### Message Types

1. Job Update
```json
{
    "type": "JOB_UPDATE",
    "data": {
        "job_id": "job_123",
        "status": "PROCESSING",
        "progress": 45.5
    },
    "timestamp": "2024-03-21T10:01:00Z"
}
```

2. Job Completed
```json
{
    "type": "JOB_COMPLETED",
    "data": {
        "job_id": "job_123",
        "output_data": {
            "output_path": "path/to/output.mp4",
            "duration": 30.0
        }
    },
    "timestamp": "2024-03-21T10:02:00Z"
}
```

3. Job Error
```json
{
    "type": "JOB_ERROR",
    "data": {
        "job_id": "job_123",
        "error": "Processing failed"
    },
    "timestamp": "2024-03-21T10:02:00Z"
}
```

4. Ping/Pong
```json
{
    "type": "PING",
    "data": {},
    "timestamp": "2024-03-21T10:00:00Z"
}
```

```json
{
    "type": "PONG",
    "data": {},
    "timestamp": "2024-03-21T10:00:00Z"
}
```

## Job Types

### Video Edit
Process and optimize a video file.

Input Data:
```json
{
    "input_path": "path/to/video.mp4",
    "target_duration": 30.0,
    "target_width": 1080,
    "target_height": 1920,
    "target_lufs": -14.0
}
```

Output Data:
```json
{
    "output_path": "path/to/output.mp4",
    "duration": 30.0,
    "width": 1080,
    "height": 1920,
    "file_size": 10485760
}
```

### Avatar Create
Create a new avatar from an image or video.

Input Data:
```json
{
    "input_path": "path/to/image.jpg",
    "avatar_type": "realistic",
    "style": {
        "style": "casual",
        "age": 25,
        "gender": "female"
    }
}
```

Output Data:
```json
{
    "output_path": "path/to/avatar.glb",
    "avatar_id": "avatar_123",
    "model_version": "1.0.0"
}
```

### Avatar Generate
Generate a video using an avatar.

Input Data:
```json
{
    "avatar_id": "avatar_123",
    "script": "Hello, world!",
    "voice_id": "voice_1",
    "background": {
        "type": "color",
        "value": "#ffffff"
    }
}
```

Output Data:
```json
{
    "output_path": "path/to/output.mp4",
    "duration": 5.0,
    "width": 1080,
    "height": 1920,
    "file_size": 5242880
}
```

### Video Translate
Translate a video to another language.

Input Data:
```json
{
    "input_path": "path/to/video.mp4",
    "target_language": "es",
    "voice_id": "voice_1",
    "subtitle_options": {
        "enabled": true,
        "position": "bottom"
    }
}
```

Output Data:
```json
{
    "output_path": "path/to/output.mp4",
    "duration": 30.0,
    "width": 1080,
    "height": 1920,
    "file_size": 10485760,
    "subtitle_path": "path/to/subtitles.srt"
}
```

## Error Responses

### 400 Bad Request
```json
{
    "error": "Bad Request",
    "message": "Invalid input data",
    "details": {
        "field": "target_duration",
        "error": "Value must be between 5 and 60 seconds"
    }
}
```

### 401 Unauthorized
```json
{
    "error": "Unauthorized",
    "message": "Invalid or missing authentication token"
}
```

### 403 Forbidden
```json
{
    "error": "Forbidden",
    "message": "Insufficient permissions"
}
```

### 404 Not Found
```json
{
    "error": "Not Found",
    "message": "Job not found"
}
```

### 429 Too Many Requests
```json
{
    "error": "Too Many Requests",
    "message": "Rate limit exceeded",
    "retry_after": 60
}
```

### 500 Internal Server Error
```json
{
    "error": "Internal Server Error",
    "message": "An unexpected error occurred"
}
``` 