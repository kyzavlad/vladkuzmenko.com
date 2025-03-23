from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

from app.core.config import settings

class TranscriptionRequest(BaseModel):
    """Schema for requesting a new transcription"""
    video_id: str = Field(..., description="ID of the video to transcribe")
    user_id: str = Field(..., description="ID of the user requesting the transcription")
    media_url: str = Field(..., description="URL or path to the audio/video file")
    language: Optional[str] = Field(
        settings.DEFAULT_LANGUAGE, 
        description="Language code (e.g., 'en', 'es')"
    )
    word_timestamps: Optional[bool] = Field(
        settings.WORD_LEVEL_TIMESTAMPS,
        description="Whether to include word-level timestamps"
    )
    diarization: Optional[bool] = Field(
        False,
        description="Whether to apply speaker diarization"
    )
    prompt: Optional[str] = Field(
        None,
        description="Optional prompt to guide the transcription"
    )
    profanity_filter: Optional[bool] = Field(
        False,
        description="Whether to filter profanity in the transcription"
    )
    
    @validator('language')
    def validate_language(cls, v):
        if v and v not in settings.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {v}. Must be one of {settings.SUPPORTED_LANGUAGES}")
        return v
    
    @validator('media_url')
    def validate_media_url(cls, v):
        # If it's a local file, check if it exists
        if os.path.isfile(v):
            # Validate file extension
            ext = os.path.splitext(v)[1].lower().lstrip('.')
            if ext not in settings.SUPPORTED_AUDIO_FORMATS:
                raise ValueError(
                    f"Unsupported file format: {ext}. Must be one of {settings.SUPPORTED_AUDIO_FORMATS}"
                )
        return v

class TranscriptionUpdate(BaseModel):
    """Schema for updating an existing transcription"""
    status: Optional[str] = None
    progress: Optional[float] = None
    error: Optional[str] = None
    full_text: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    words: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    json_path: Optional[str] = None
    srt_path: Optional[str] = None
    vtt_path: Optional[str] = None
    txt_path: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ['pending', 'processing', 'completed', 'failed']:
            raise ValueError(f"Invalid status: {v}. Must be one of ['pending', 'processing', 'completed', 'failed']")
        return v
    
    @validator('progress')
    def validate_progress(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError(f"Progress must be between 0 and 1, got {v}")
        return v

class TranscriptionResponse(BaseModel):
    """Schema for transcription response"""
    id: str
    video_id: str
    user_id: str
    media_url: str
    language: str
    status: str
    progress: float
    error: Optional[str] = None
    
    # Results (only included when available)
    full_text: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    words: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    
    # Output file paths
    json_path: Optional[str] = None
    srt_path: Optional[str] = None
    vtt_path: Optional[str] = None
    txt_path: Optional[str] = None
    
    # Metadata
    media_duration: Optional[float] = None
    media_format: Optional[str] = None
    word_timestamps: bool
    diarization: bool
    profanity_filter: bool
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class TranscriptionJobResponse(BaseModel):
    """Schema for transcription job response"""
    id: str
    transcription_id: str
    status: str
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    queue_time: Optional[float] = None
    processing_time: Optional[float] = None
    retry_count: int
    
    class Config:
        orm_mode = True 