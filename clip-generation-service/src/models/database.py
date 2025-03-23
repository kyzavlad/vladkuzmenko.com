from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class JobStatus(str, enum.Enum):
    """Job status options."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(str, enum.Enum):
    """Job type options."""
    VIDEO_EDIT = "video_edit"
    AVATAR_CREATE = "avatar_create"
    AVATAR_GENERATE = "avatar_generate"
    VIDEO_TRANSLATE = "video_translate"

class User(Base):
    """User database model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    jobs = relationship("Job", back_populates="user")
    api_keys = relationship("ApiKey", back_populates="user")

class Job(Base):
    """Job database model."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    job_type = Column(SQLEnum(JobType))
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    progress = Column(Float, default=0.0)
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="jobs")
    logs = relationship("JobLog", back_populates="job")

class JobLog(Base):
    """Job log database model."""
    __tablename__ = "job_logs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String)
    message = Column(String)
    details = Column(JSON, nullable=True)
    
    job = relationship("Job", back_populates="logs")

class ApiKey(Base):
    """API key database model."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    key = Column(String, unique=True, index=True)
    name = Column(String)
    description = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    permissions = Column(JSON, default=list)
    
    user = relationship("User", back_populates="api_keys")

class SecurityAudit(Base):
    """Security audit database model."""
    __tablename__ = "security_audits"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    details = Column(JSON, nullable=True)
    security_level = Column(String) 