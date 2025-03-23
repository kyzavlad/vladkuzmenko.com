# Clip Generation Service

A FastAPI-based service for processing video clips, creating avatars, and generating video content.

## Features

- Video editing and optimization
- Avatar creation and generation
- Video translation
- Real-time job status updates via WebSocket
- Job queue management
- Comprehensive logging and monitoring

## Prerequisites

- Python 3.8+
- FFmpeg
- Redis (optional, for caching)
- PostgreSQL (or SQLite for development)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/clip-generation-service.git
cd clip-generation-service
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

The service can be configured through environment variables or a configuration file. See `.env.example` for available options.

## Database Setup

1. Initialize the database:
```bash
python -m src.database.init_db
```

2. Run migrations:
```bash
alembic upgrade head
```

## Running the Service

1. Start the service:
```bash
uvicorn src.main:app --reload
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## API Endpoints

### Video Processing

- `POST /jobs` - Create a new job
- `GET /jobs/{job_id}` - Get job status
- `GET /jobs` - List user's jobs
- `DELETE /jobs/{job_id}` - Cancel a job

### WebSocket

- `WS /ws/{user_id}` - WebSocket endpoint for real-time updates

## Job Types

### Video Edit
```json
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

### Avatar Create
```json
{
    "job_type": "AVATAR_CREATE",
    "input_data": {
        "input_path": "path/to/image.jpg",
        "avatar_type": "realistic",
        "style": {
            "style": "casual"
        }
    }
}
```

### Avatar Generate
```json
{
    "job_type": "AVATAR_GENERATE",
    "input_data": {
        "avatar_id": "avatar_1",
        "script": "Hello, world!",
        "voice_id": "voice_1"
    }
}
```

### Video Translate
```json
{
    "job_type": "VIDEO_TRANSLATE",
    "input_data": {
        "input_path": "path/to/video.mp4",
        "target_language": "es",
        "voice_id": "voice_1"
    }
}
```

## Testing

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src tests/
```

## Monitoring

The service includes Prometheus metrics and health checks:

- `/metrics` - Prometheus metrics endpoint
- `/health` - Health check endpoint

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 