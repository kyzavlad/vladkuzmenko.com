from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class ProcessingOptions(BaseModel):
    """
    Video processing options
    """
    transcription: bool = True
    subtitles: bool = True
    enhancement: bool = False
    avatar: bool = False


class VideoCreate(BaseModel):
    """
    Schema for creating a new video
    """
    user_id: str
    title: str
    description: Optional[str] = None
    original_filename: str
    content_type: str
    file_size: int
    transcription_enabled: bool = True
    subtitles_enabled: bool = True
    enhancement_enabled: bool = False
    avatar_enabled: bool = False


class TaskInfo(BaseModel):
    """
    Information about a processing task
    """
    id: str
    task_type: str
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class VideoResponse(BaseModel):
    """
    Basic video information response
    """
    id: str
    title: str
    description: Optional[str] = None
    status: str
    created_at: datetime
    progress: float = 0.0
    message: Optional[str] = None


class VideoList(BaseModel):
    """
    Response for listing videos
    """
    videos: List[VideoResponse]
    total: int


class VideoDetail(BaseModel):
    """
    Detailed video information response
    """
    id: str
    title: str
    description: Optional[str] = None
    original_filename: str
    content_type: str
    file_size: int
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    
    status: str
    progress: float
    error_message: Optional[str] = None
    
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    transcription_enabled: bool
    subtitles_enabled: bool
    enhancement_enabled: bool
    avatar_enabled: bool
    
    transcription_id: Optional[str] = None
    
    original_url: Optional[str] = None
    enhanced_url: Optional[str] = None
    avatar_url: Optional[str] = None
    subtitle_url: Optional[str] = None
    
    tasks: List[Dict[str, Any]] = []


class VideoStatusUpdate(BaseModel):
    """
    Schema for updating video status
    """
    status: str
    progress: Optional[float] = None
    error_message: Optional[str] = None

class VideoProcessingOptions(BaseModel):
    # Transcription options
    transcription: bool = False
    transcription_language: Optional[str] = "en"
    
    # Pause detection options
    pause_detection: bool = False
    min_pause_duration: float = 0.75  # Default minimum pause duration in seconds
    
    # Subtitle options
    subtitles: bool = False
    subtitle_format: str = "srt"  # Options: srt, ass, vtt
    subtitle_font: str = "Arial"
    subtitle_font_size: int = 24
    subtitle_font_color: str = "white"
    
    # B-roll suggestions
    b_roll_suggestions: bool = False
    b_roll_keywords: Optional[List[str]] = None
    
    # Background music
    background_music: bool = False
    music_genre: Optional[str] = None
    music_tempo: Optional[str] = None  # slow, medium, fast
    
    # Sound effects
    sound_effects: bool = False
    
    # Video enhancement
    enhancement: bool = False
    enhance_resolution: bool = False
    target_resolution: Optional[str] = None  # 720p, 1080p, 4K
    noise_reduction: bool = False
    color_correction: bool = False
    
    # Avatar generation
    avatar: bool = False
    avatar_style: Optional[str] = None
    
    @validator('subtitle_format')
    def validate_subtitle_format(cls, v):
        allowed_formats = ['srt', 'ass', 'vtt']
        if v not in allowed_formats:
            raise ValueError(f"Subtitle format must be one of {allowed_formats}")
        return v
    
    @validator('music_tempo')
    def validate_music_tempo(cls, v):
        if v is not None:
            allowed_tempos = ['slow', 'medium', 'fast']
            if v not in allowed_tempos:
                raise ValueError(f"Music tempo must be one of {allowed_tempos}")
        return v 