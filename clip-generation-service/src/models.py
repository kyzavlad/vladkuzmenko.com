from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ContentType(str, Enum):
    AUTO = "auto"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    TUTORIAL = "tutorial"
    VLOG = "vlog"
    MUSIC = "music"
    SPORTS = "sports"

class OutputFormat(str, Enum):
    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"

class Platform(str, Enum):
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"

class CaptionStyle(str, Enum):
    DEFAULT = "default"
    MINIMAL = "minimal"
    BOLD = "bold"
    COLORFUL = "colorful"

class ClipStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Clip(BaseModel):
    clip_id: str
    start_time: float
    end_time: float
    duration: float
    preview_url: str
    download_url: str
    thumbnail_url: str
    engagement_score: float = Field(ge=0, le=100)
    tags: List[str] = []
    transcript_segment: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None

class ClipGenerationRequest(BaseModel):
    min_clip_duration: int = Field(default=5, ge=5, le=60)
    max_clip_duration: int = Field(default=60, ge=5, le=120)
    max_clips: int = Field(default=10, ge=1, le=50)
    focus_on_faces: bool = True
    remove_silences: bool = True
    highlight_interesting: bool = True
    target_aspect_ratio: str = "9:16"
    content_type: ContentType = ContentType.AUTO
    output_format: OutputFormat = OutputFormat.MP4

class ClipUpdateRequest(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class BatchExportRequest(BaseModel):
    clip_ids: List[str]
    platform: Platform
    include_captions: bool = True
    caption_style: CaptionStyle = CaptionStyle.DEFAULT
    add_watermark: bool = False
    watermark_image: Optional[str] = None

class JobStatus(BaseModel):
    status: ClipStatus
    progress: float = Field(ge=0, le=100)
    clips_generated: int = 0
    clips: List[Clip] = []
    error_message: Optional[str] = None 