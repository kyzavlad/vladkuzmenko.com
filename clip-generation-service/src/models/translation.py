from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from enum import Enum

class TranslationQuality(str, Enum):
    """Translation quality options."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TranslationRequest(BaseModel):
    """Video translation request model."""
    source_language: str
    target_language: str
    preserve_voice_tone: bool = True
    generate_subtitles: bool = True
    quality: TranslationQuality = TranslationQuality.HIGH
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    pitch: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    volume: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    background_noise: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    metadata: Optional[Dict] = None

class TranslationResponse(BaseModel):
    """Video translation response model."""
    job_id: str
    status: str
    estimated_completion_time: int
    preview_url: Optional[str] = None
    result_url: Optional[str] = None
    subtitles_url: Optional[str] = None
    duration: Optional[float] = None
    size: Optional[int] = None
    format: Optional[str] = None
    resolution: Optional[str] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None

class TranslationStatusResponse(BaseModel):
    """Translation processing status response model."""
    status: str
    progress: float
    result_url: Optional[str] = None
    subtitles_url: Optional[str] = None
    duration: Optional[float] = None
    size: Optional[int] = None
    format: Optional[str] = None
    resolution: Optional[str] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None 