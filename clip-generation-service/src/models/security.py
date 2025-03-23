from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """User role options."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class TokenType(str, Enum):
    """Token type options."""
    ACCESS = "access"
    REFRESH = "refresh"
    API = "api"

class SecurityLevel(str, Enum):
    """Security level options."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RateLimitConfig(BaseModel):
    """Rate limit configuration model."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10

class ApiKeyConfig(BaseModel):
    """API key configuration model."""
    key: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    permissions: List[str] = []
    rate_limit: RateLimitConfig = RateLimitConfig()

class SecurityConfig(BaseModel):
    """Security configuration model."""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_hash_algorithm: str = "bcrypt"
    password_salt_rounds: int = 12
    csrf_token_expire_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    require_2fa: bool = False
    allowed_origins: List[str] = []
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: List[str] = ["*"]
    expose_headers: List[str] = []
    max_age: int = 600
    credentials: bool = True

class SecurityAuditLog(BaseModel):
    """Security audit log model."""
    timestamp: datetime
    user_id: str
    action: str
    ip_address: str
    user_agent: str
    details: Optional[Dict] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM

class SecurityMetrics(BaseModel):
    """Security metrics model."""
    total_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    rate_limited_requests: int = 0
    suspicious_activities: int = 0
    security_incidents: int = 0
    last_incident: Optional[datetime] = None
    metrics_timestamp: datetime = Field(default_factory=datetime.utcnow) 