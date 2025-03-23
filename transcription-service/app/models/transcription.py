from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
from typing import Dict, List, Any, Optional

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class Transcription(Base):
    """Database model for storing transcription data and metadata"""
    __tablename__ = "transcriptions"
    
    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    
    # Identifying information
    video_id = Column(String, index=True, nullable=False)  # ID from Video Processing Service
    user_id = Column(String, index=True, nullable=False)   # ID from Authentication Service
    
    # Source media info
    media_url = Column(String, nullable=False)  # URL or path to the audio/video file
    media_duration = Column(Float, nullable=True)  # Duration in seconds
    media_format = Column(String, nullable=True)  # File format (mp3, mp4, etc.)
    
    # Transcription metadata
    language = Column(String, nullable=False)  # Language code (en, es, etc.)
    model = Column(String, nullable=True)  # Model used for transcription (whisper-1, etc.)
    word_timestamps = Column(Boolean, default=True)  # Whether word-level timestamps were generated
    
    # Processing status
    status = Column(String, index=True, default="pending")  # pending, processing, completed, failed
    error = Column(Text, nullable=True)  # Error message if transcription failed
    
    # Progress tracking
    progress = Column(Float, default=0.0)  # Progress from 0 to 1 (0% to 100%)
    
    # Results 
    full_text = Column(Text, nullable=True)  # Complete transcription text
    segments = Column(JSON, nullable=True)   # List of timestamped segments
    words = Column(JSON, nullable=True)      # List of timestamped words if enabled
    confidence = Column(Float, nullable=True)  # Overall confidence score
    
    # Output file paths
    json_path = Column(String, nullable=True)  # Path to JSON transcription file
    srt_path = Column(String, nullable=True)   # Path to SRT subtitle file
    vtt_path = Column(String, nullable=True)   # Path to VTT subtitle file
    txt_path = Column(String, nullable=True)   # Path to plain text file
    
    # Options and features
    diarization = Column(Boolean, default=False)  # Whether speaker diarization was applied
    speaker_count = Column(Integer, nullable=True)  # Number of speakers identified
    profanity_filter = Column(Boolean, default=False)  # Whether profanity was filtered
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime, nullable=True)  # When processing started
    completed_at = Column(DateTime, nullable=True)  # When processing completed

class TranscriptionJob(Base):
    """Database model for tracking transcription job status"""
    __tablename__ = "transcription_jobs"
    
    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    transcription_id = Column(String, ForeignKey("transcriptions.id"), index=True, nullable=False)
    
    # Job status and tracking
    status = Column(String, index=True, default="queued")  # queued, in_progress, completed, failed
    worker_id = Column(String, nullable=True)  # ID of worker processing this job
    priority = Column(Integer, default=0)  # Priority level (higher is more important)
    
    # Job parameters
    parameters = Column(JSON, nullable=True)  # Additional parameters for the job
    result = Column(JSON, nullable=True)      # Final results or metrics
    error = Column(Text, nullable=True)       # Error information if failed
    
    # Timing information
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    queue_time = Column(Float, nullable=True)  # Time spent in queue in seconds
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    
    # Retry information
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    next_retry_at = Column(DateTime, nullable=True) 