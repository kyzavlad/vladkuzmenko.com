from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from enum import Enum

class SubtitleStyle(str, Enum):
    """Subtitle style options."""
    STANDARD = "standard"
    MINIMAL = "minimal"
    CUSTOM = "custom"

class OutputFormat(str, Enum):
    """Output format options."""
    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"

class Resolution(str, Enum):
    """Resolution options."""
    ORIGINAL = "original"
    HD = "1080p"
    SD = "720p"
    LOW = "480p"

class VideoEditRequest(BaseModel):
    """Video editing request model."""
    enable_subtitles: bool = True
    subtitle_style: SubtitleStyle = SubtitleStyle.STANDARD
    enable_b_roll: bool = True
    b_roll_intensity: float = Field(0.5, ge=0.0, le=1.0)
    enable_music: bool = True
    music_genre: str = "auto"
    music_intensity: float = Field(0.4, ge=0.0, le=1.0)
    enable_sound_effects: bool = True
    enable_enhancement: bool = True
    noise_reduction_level: float = Field(0.7, ge=0.0, le=1.0)
    pause_threshold: float = Field(0.8, ge=0.0, le=1.0)
    output_format: OutputFormat = OutputFormat.MP4
    resolution: Resolution = Resolution.ORIGINAL

class VideoEditResponse(BaseModel):
    """Video editing response model."""
    job_id: str
    estimated_completion_time: int

class VideoStatusResponse(BaseModel):
    """Video processing status response model."""
    status: str
    progress: float
    result_url: Optional[str] = None
    transcript: Optional[Dict] = None
    title_suggestions: Optional[List[str]] = None
    description_suggestions: Optional[Dict] = None
    improvement_recommendations: Optional[List[str]] = None
    error: Optional[str] = None 