from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "clip-generation-service"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    BACKLOG: int = 2048

    # Database Settings
    DB_TYPE: str = "sqlite"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "clip_generation"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_URL: Optional[str] = None

    # Storage Settings
    STORAGE_DIR: str = "./storage"
    UPLOAD_DIR: str = "./storage/uploads"
    OUTPUT_DIR: str = "./storage/outputs"
    MAX_UPLOAD_SIZE: str = "500MB"

    # Processing Settings
    MAX_CONCURRENT_JOBS: int = 5
    JOB_TIMEOUT: int = 3600
    CLEANUP_INTERVAL: int = 86400

    # Security Settings
    SECRET_KEY: str
    API_KEY_HEADER: str = "X-API-Key"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = 3600
    ALLOWED_ORIGINS: str = "http://localhost:3000,https://yourdomain.com"

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    USE_REDIS: bool = False

    # Monitoring Settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    ENABLE_HEALTH_CHECKS: bool = True

    # WebSocket Settings
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_PING_TIMEOUT: int = 10

    # FFmpeg Settings
    FFMPEG_THREADS: int = 4
    FFMPEG_PRESET: str = "medium"
    FFMPEG_CRF: int = 23

    # Avatar Generation Settings
    AVATAR_MODEL_PATH: str = "./models/avatar"
    AVATAR_CACHE_SIZE: int = 100
    AVATAR_MAX_DURATION: int = 300

    # Video Translation Settings
    TRANSLATION_API_KEY: Optional[str] = None
    TTS_API_KEY: Optional[str] = None
    MAX_TRANSLATION_LENGTH: int = 5000

    # Token Settings
    FREE_TOKENS: int = 60
    TOKENS_PER_SECOND: int = 1
    LOW_BALANCE_THRESHOLD: int = 10
    TOKEN_EXPIRY_DAYS: int = 365

    # Stripe Settings
    STRIPE_SECRET_KEY: str
    STRIPE_WEBHOOK_SECRET: str
    STRIPE_PUBLISHABLE_KEY: str
    STRIPE_CURRENCY: str = "USD"
    STRIPE_PAYMENT_METHODS: str = "card"
    STRIPE_TAX_RATE_ID: Optional[str] = None
    STRIPE_INVOICE_PREFIX: str = "CLIP-"
    STRIPE_SUBSCRIPTION_PLAN_ID: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 