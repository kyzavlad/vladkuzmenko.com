import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Transcription Service"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Authentication service integration
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8000")
    SERVICE_API_KEY: str = os.getenv("SERVICE_API_KEY", "")
    
    # Video processing service integration
    VIDEO_PROCESSING_SERVICE_URL: str = os.getenv("VIDEO_PROCESSING_SERVICE_URL", "http://video-processing-service:8000")
    
    # RabbitMQ settings for asynchronous processing
    RABBITMQ_HOST: str = os.getenv("RABBITMQ_HOST", "rabbitmq")
    RABBITMQ_PORT: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    RABBITMQ_USER: str = os.getenv("RABBITMQ_USER", "guest")
    RABBITMQ_PASSWORD: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    RABBITMQ_QUEUE: str = os.getenv("RABBITMQ_QUEUE", "transcription_tasks")
    
    # OpenAI API configuration for Whisper
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")
    
    # Transcription settings
    MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "100"))  # 100 MB default max size
    SUPPORTED_AUDIO_FORMATS: List[str] = ["mp3", "mp4", "wav", "m4a", "webm", "mpga", "mpeg"]
    SUPPORTED_LANGUAGES: List[str] = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar"]
    DEFAULT_LANGUAGE: str = "en"
    WORD_LEVEL_TIMESTAMPS: bool = True
    TIMESTAMPS_GRANULARITY: str = "word"  # "word" or "segment"
    
    # Storage settings
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # Options: local, s3
    LOCAL_STORAGE_PATH: str = os.getenv("LOCAL_STORAGE_PATH", "storage/transcriptions")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")
    S3_ACCESS_KEY: Optional[str] = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY: Optional[str] = os.getenv("S3_SECRET_KEY")
    S3_REGION: Optional[str] = os.getenv("S3_REGION", "us-east-1")
    
    # Database configuration
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "postgres")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "transcription")
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
    # Performance settings
    MAX_CONCURRENT_TRANSCRIPTIONS: int = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "5"))
    TRANSCRIPTION_TIMEOUT_SECONDS: int = int(os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", "1800"))  # 30 minutes
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.SQLALCHEMY_DATABASE_URI = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
        
        # Validate OpenAI API key
        if not self.OPENAI_API_KEY and os.getenv("ENVIRONMENT", "development") == "production":
            print("WARNING: OPENAI_API_KEY is not set in production environment")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 