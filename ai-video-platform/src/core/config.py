from typing import List
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Video Platform"
    
    # CORS Settings
    CORS_ORIGINS: List[AnyHttpUrl] = []
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Security Settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database Settings
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: str | None = None
    
    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: str | None, values: dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"
    
    # MongoDB Settings
    MONGODB_URL: str
    MONGODB_DB: str
    
    # Redis Settings
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str | None = None
    
    # Storage Settings
    S3_ENDPOINT: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_BUCKET: str
    S3_REGION: str
    
    # Celery Settings
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # Monitoring Settings
    PROMETHEUS_MULTIPROC_DIR: str = "/tmp"
    ELASTICSEARCH_URL: str
    LOG_LEVEL: str = "INFO"
    
    # Video Processing Settings
    MAX_VIDEO_SIZE_MB: int = 500
    ALLOWED_VIDEO_FORMATS: List[str] = ["mp4", "mov", "avi", "mkv"]
    MAX_CONCURRENT_PROCESSING: int = 10
    GPU_MEMORY_LIMIT: int = 16  # GB
    
    # Token Settings
    FREE_TOKENS_PER_USER: int = 60
    TOKEN_EXPIRY_DAYS: int = 30
    
    # Billing Settings
    STRIPE_SECRET_KEY: str
    STRIPE_WEBHOOK_SECRET: str
    
    # Feature Flags
    ENABLE_GPU_PROCESSING: bool = True
    ENABLE_AVATAR_GENERATION: bool = True
    ENABLE_VIDEO_TRANSLATION: bool = True
    ENABLE_QUALITY_ANALYSIS: bool = True
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 