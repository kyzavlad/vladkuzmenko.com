from sqlalchemy import Boolean, Column, Integer, String, Float, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
import enum

from .base import Base

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(str, enum.Enum):
    VIDEO_EDIT = "video_edit"
    AVATAR_GENERATE = "avatar_generate"
    VIDEO_TRANSLATE = "video_translate"
    QUALITY_ANALYSIS = "quality_analysis"

class Job(Base):
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    type = Column(Enum(JobType), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    progress = Column(Float, default=0.0)
    
    # Input parameters
    input_url = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    
    # Output data
    output_url = Column(String)
    result_data = Column(JSON)
    error_message = Column(String)
    
    # Processing metrics
    processing_time = Column(Float)  # in seconds
    token_cost = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="jobs")
    quality_metrics = relationship("QualityMetrics", back_populates="job", uselist=False)
    analytics = relationship("JobAnalytics", back_populates="job", uselist=False) 