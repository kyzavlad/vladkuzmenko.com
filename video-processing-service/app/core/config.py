import os
import secrets
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseSettings, PostgresDsn, validator, AnyHttpUrl


class Settings(BaseSettings):
    API_PREFIX: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # CORS configuration
    CORS_ORIGINS: List[str] = ["*"]
    
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    
    # Database configuration
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "video_processing")
    DATABASE_URI: Optional[PostgresDsn] = None

    @validator("DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    # Authentication and service communication
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
    SERVICE_API_KEY: str = os.getenv("SERVICE_API_KEY", "development-api-key")
    
    # Video storage configuration
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # "local", "s3"
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))  # 500 MB
    ALLOWED_VIDEO_FORMATS: List[str] = ["mp4", "mov", "avi", "mkv", "webm"]
    
    # Local storage settings
    LOCAL_STORAGE_PATH: str = os.getenv("LOCAL_STORAGE_PATH", "/tmp/video-processing")
    
    # S3 storage settings (if STORAGE_TYPE is "s3")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")
    S3_REGION: Optional[str] = os.getenv("S3_REGION")
    S3_ACCESS_KEY: Optional[str] = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY: Optional[str] = os.getenv("S3_SECRET_KEY")
    
    # Transcription service settings
    TRANSCRIPTION_SERVICE_URL: str = os.getenv("TRANSCRIPTION_SERVICE_URL", "http://localhost:8002")
    
    # Processing settings
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
    JOB_TIMEOUT_SECONDS: int = int(os.getenv("JOB_TIMEOUT_SECONDS", "3600"))  # 1 hour
    
    # Worker queue settings (if using a message queue)
    RABBITMQ_HOST: str = os.getenv("RABBITMQ_HOST", "localhost")
    RABBITMQ_PORT: str = os.getenv("RABBITMQ_PORT", "5672")
    RABBITMQ_USER: str = os.getenv("RABBITMQ_USER", "guest")
    RABBITMQ_PASSWORD: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    RABBITMQ_QUEUE: str = os.getenv("RABBITMQ_QUEUE", "video_processing_jobs")
    
    # Avatar settings
    AVATAR_GENERATION_ENABLED: bool = os.getenv("AVATAR_GENERATION_ENABLED", "False").lower() == "true"
    AVATAR_SERVICE_URL: Optional[str] = os.getenv("AVATAR_SERVICE_URL")
    
    class Config:
        case_sensitive = True


settings = Settings()

# Create local storage directory if needed
if settings.STORAGE_TYPE == "local":
    os.makedirs(settings.LOCAL_STORAGE_PATH, exist_ok=True) 