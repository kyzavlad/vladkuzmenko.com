from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import enum

class LogLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DatabaseType(str, enum.Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MONGODB = "mongodb"

class CacheType(str, enum.Enum):
    REDIS = "redis"
    MEMCACHED = "memcached"
    LOCAL = "local"

class StorageType(str, enum.Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"

class DatabaseConfig(BaseModel):
    type: DatabaseType
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: str
    pool_size: int = Field(default=5, ge=1, le=20)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout: int = Field(default=30, ge=1)
    pool_recycle: int = Field(default=1800, ge=1)
    echo: bool = False

class CacheConfig(BaseModel):
    type: CacheType
    host: Optional[str] = None
    port: Optional[int] = None
    password: Optional[str] = None
    database: Optional[int] = None
    ttl: int = Field(default=3600, ge=1)
    max_size: int = Field(default=1000, ge=1)

class StorageConfig(BaseModel):
    type: StorageType
    bucket: str
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    use_ssl: bool = True
    max_retries: int = Field(default=3, ge=1)
    timeout: int = Field(default=30, ge=1)

class LoggingConfig(BaseModel):
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_size: int = Field(default=10485760, ge=1)  # 10MB
    backup_count: int = Field(default=5, ge=1)
    syslog: bool = False
    syslog_host: Optional[str] = None
    syslog_port: Optional[int] = None
    syslog_facility: Optional[str] = None

class SecurityConfig(BaseModel):
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=1)
    refresh_token_expire_days: int = Field(default=7, ge=1)
    password_hash_algorithm: str = "bcrypt"
    password_salt_rounds: int = Field(default=12, ge=1)
    cors_origins: List[str] = []
    cors_allow_credentials: bool = True
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = Field(default=60, ge=1)
    rate_limit_burst_size: int = Field(default=100, ge=1)
    api_key_enabled: bool = True
    api_key_expire_days: int = Field(default=365, ge=1)
    api_key_max_keys_per_user: int = Field(default=5, ge=1)

class ProcessingConfig(BaseModel):
    max_concurrent_jobs: int = Field(default=10, ge=1)
    job_timeout_seconds: int = Field(default=3600, ge=1)
    max_file_size_mb: int = Field(default=500, ge=1)
    allowed_video_formats: List[str] = ["mp4", "mov", "avi", "mkv"]
    allowed_audio_formats: List[str] = ["mp3", "wav", "ogg", "m4a"]
    allowed_image_formats: List[str] = ["jpg", "jpeg", "png", "gif"]
    temp_directory: str = "/tmp"
    output_directory: str = "output"
    cleanup_temp_files: bool = True
    cleanup_interval_hours: int = Field(default=24, ge=1)

class NotificationConfig(BaseModel):
    email_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    smtp_from_email: Optional[str] = None
    smtp_from_name: Optional[str] = None
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    webhook_timeout_seconds: int = Field(default=5, ge=1)
    webhook_max_retries: int = Field(default=3, ge=1)

class MonitoringConfig(BaseModel):
    metrics_enabled: bool = True
    metrics_port: int = Field(default=9090, ge=1)
    health_check_enabled: bool = True
    health_check_interval_seconds: int = Field(default=30, ge=1)
    tracing_enabled: bool = False
    tracing_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    tracing_exporter_url: Optional[str] = None

class AppConfig(BaseModel):
    app_name: str = "Clip Generation Service"
    app_version: str = "1.0.0"
    app_description: str = "AI-powered video clip generation service"
    debug: bool = False
    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1)
    workers: int = Field(default=4, ge=1)
    reload: bool = False
    database: DatabaseConfig
    cache: Optional[CacheConfig] = None
    storage: StorageConfig
    logging: LoggingConfig
    security: SecurityConfig
    processing: ProcessingConfig
    notification: NotificationConfig
    monitoring: MonitoringConfig
    custom_settings: Dict[str, Any] = {}

    @validator("environment")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("environment must be one of: development, staging, production")
        return v

    @validator("workers")
    def validate_workers(cls, v):
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        if v > max_workers:
            raise ValueError(f"workers cannot exceed CPU count ({max_workers})")
        return v 