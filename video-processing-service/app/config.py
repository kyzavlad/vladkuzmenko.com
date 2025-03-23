"""
Configuration settings for the application.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile

class Settings:
    """Application settings."""
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    
    # Temporary directory for file storage
    TEMP_DIRECTORY: str = os.environ.get("TEMP_DIRECTORY", os.path.join(tempfile.gettempdir(), "video_processing"))
    
    # Default video path for testing (when no video is uploaded)
    DEFAULT_VIDEO_PATH: str = os.environ.get("DEFAULT_VIDEO_PATH", "samples/sample_video.mp4")
    
    # Path to FFmpeg and FFprobe executables
    FFMPEG_PATH: str = os.environ.get("FFMPEG_PATH", "ffmpeg")
    FFPROBE_PATH: str = os.environ.get("FFPROBE_PATH", "ffprobe")
    
    # Subtitle service configuration
    SUBTITLE_CONFIG: Dict[str, Any] = {
        "default_format": "srt",
        "ffmpeg_path": FFMPEG_PATH,
        "ffprobe_path": FFPROBE_PATH,
        "use_lite_positioning": os.environ.get("USE_LITE_POSITIONING", "false").lower() == "true",
        "positioning_config": {
            "enable_face_detection": True,
            "enable_object_detection": True,
            "enable_text_detection": True
        },
        "language_config": {
            "auto_detect_language": True,
            "default_language": "en"
        }
    }
    
    # Maximum file size for uploads (in bytes)
    MAX_UPLOAD_SIZE: int = int(os.environ.get("MAX_UPLOAD_SIZE", 1024 * 1024 * 100))  # 100 MB default
    
    # File retention period (in seconds)
    FILE_RETENTION_PERIOD: int = int(os.environ.get("FILE_RETENTION_PERIOD", 3600))  # 1 hour default
    
    # Output file cleanup schedule (cron format)
    CLEANUP_SCHEDULE: str = os.environ.get("CLEANUP_SCHEDULE", "0 */1 * * *")  # Every hour

settings = Settings() 