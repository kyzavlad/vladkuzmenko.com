from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
from sqlalchemy.orm import relationship

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing" 
    GENERATING_SUBTITLES = "generating_subtitles"
    ENHANCING = "enhancing"
    GENERATING_AVATAR = "generating_avatar"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # File info
    original_filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)  # In bytes
    duration = Column(Float, nullable=True)  # In seconds
    
    # Video properties
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    
    # Processing info
    status = Column(String, nullable=False, default=ProcessingStatus.PENDING.value)
    progress = Column(Float, nullable=False, default=0.0)  # 0-100 percent
    error_message = Column(Text, nullable=True)
    
    # Processing options
    transcription_enabled = Column(Boolean, default=True)
    subtitles_enabled = Column(Boolean, default=True) 
    enhancement_enabled = Column(Boolean, default=False)
    avatar_enabled = Column(Boolean, default=False)
    
    # Processing result references
    transcription_id = Column(String, nullable=True)
    subtitle_path = Column(String, nullable=True)
    enhanced_video_path = Column(String, nullable=True)
    avatar_video_path = Column(String, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)  # Store additional processing metadata
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Features applied
    has_transcription = Column(Boolean, default=False)
    has_subtitles = Column(Boolean, default=False)
    has_pause_detection = Column(Boolean, default=False)
    has_b_roll = Column(Boolean, default=False)
    has_background_music = Column(Boolean, default=False)
    has_sound_effects = Column(Boolean, default=False)
    has_enhancement = Column(Boolean, default=False)
    has_avatar = Column(Boolean, default=False)

    # Processing tasks
    tasks = relationship("ProcessingTask", back_populates="video", cascade="all, delete-orphan")

class ProcessingTask(Base):
    __tablename__ = "processing_tasks"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    task_type = Column(String, nullable=False)  # transcription, enhancement, subtitle, avatar, etc.
    status = Column(String, nullable=False, default=ProcessingStatus.PENDING.value)
    progress = Column(Float, nullable=False, default=0.0)  # 0-100 percent
    result = Column(JSON, nullable=True)  # Store the result data
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="tasks") 