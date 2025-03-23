from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from enum import Enum

class AvatarStyle(str, Enum):
    """Avatar style options."""
    REALISTIC = "realistic"
    ANIMATED = "animated"
    CUSTOM = "custom"

class AvatarCreateRequest(BaseModel):
    """Avatar creation request model."""
    avatar_name: str
    style: AvatarStyle = AvatarStyle.REALISTIC
    voice_sample_url: Optional[str] = None
    voice_sample_text: Optional[str] = None
    voice_sample_language: Optional[str] = None
    voice_sample_emotion: Optional[str] = None
    voice_sample_speed: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    voice_sample_pitch: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    voice_sample_volume: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    voice_sample_quality: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    voice_sample_duration: Optional[float] = None
    voice_sample_format: Optional[str] = None
    voice_sample_size: Optional[int] = None
    voice_sample_channels: Optional[int] = None
    voice_sample_sample_rate: Optional[int] = None
    voice_sample_bit_depth: Optional[int] = None
    voice_sample_bit_rate: Optional[int] = None
    voice_sample_codec: Optional[str] = None
    voice_sample_metadata: Optional[Dict] = None

class AvatarCreateResponse(BaseModel):
    """Avatar creation response model."""
    avatar_id: str
    status: str
    estimated_completion_time: int
    preview_url: Optional[str] = None
    voice_sample_url: Optional[str] = None
    voice_sample_text: Optional[str] = None
    voice_sample_language: Optional[str] = None
    voice_sample_emotion: Optional[str] = None
    voice_sample_speed: Optional[float] = None
    voice_sample_pitch: Optional[float] = None
    voice_sample_volume: Optional[float] = None
    voice_sample_quality: Optional[float] = None
    voice_sample_duration: Optional[float] = None
    voice_sample_format: Optional[str] = None
    voice_sample_size: Optional[int] = None
    voice_sample_channels: Optional[int] = None
    voice_sample_sample_rate: Optional[int] = None
    voice_sample_bit_depth: Optional[int] = None
    voice_sample_bit_rate: Optional[int] = None
    voice_sample_codec: Optional[str] = None
    voice_sample_metadata: Optional[Dict] = None

class AvatarGenerateRequest(BaseModel):
    """Avatar video generation request model."""
    avatar_id: str
    script: str
    language: str = "en"
    emotion: Optional[str] = None
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    pitch: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    volume: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    quality: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    background: Optional[str] = None
    resolution: Optional[str] = "1080p"
    format: Optional[str] = "mp4"
    duration: Optional[float] = None
    metadata: Optional[Dict] = None

class AvatarGenerateResponse(BaseModel):
    """Avatar video generation response model."""
    job_id: str
    status: str
    estimated_completion_time: int
    preview_url: Optional[str] = None
    result_url: Optional[str] = None
    duration: Optional[float] = None
    size: Optional[int] = None
    format: Optional[str] = None
    resolution: Optional[str] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None 