"""
Clip Generation Service - Data Models

This module defines the data models used throughout the Clip Generation Service.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import time


class TaskStatus(str, Enum):
    """Enumeration of possible task statuses."""
    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClipFormat(str, Enum):
    """Enumeration of supported clip formats."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"
    WEBM = "webm"
    GIF = "gif"


class QualityPreset(str, Enum):
    """Enumeration of quality presets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class ClipTask:
    """
    Represents a clip generation task.
    """
    task_id: str
    source_video: str
    start_time: float
    end_time: float
    output_name: str
    output_path: str
    format: str
    quality: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = TaskStatus.CREATED
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "task_id": self.task_id,
            "source_video": self.source_video,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "output_name": self.output_name,
            "output_path": self.output_path,
            "format": self.format,
            "quality": self.quality,
            "params": self.params,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "callback_url": self.callback_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClipTask':
        """Create task from dictionary."""
        return cls(**data)


@dataclass
class FFmpegPreset:
    """
    Represents an FFmpeg encoding preset configuration.
    """
    name: str
    video_codec: str
    audio_codec: str
    video_bitrate: str
    audio_bitrate: str
    additional_params: Dict[str, str] = field(default_factory=dict)
    
    def to_ffmpeg_args(self) -> List[str]:
        """Convert preset to FFmpeg command-line arguments."""
        args = []
        
        # Video codec
        args.extend(["-c:v", self.video_codec])
        
        # Audio codec
        args.extend(["-c:a", self.audio_codec])
        
        # Bitrates
        if self.video_bitrate:
            args.extend(["-b:v", self.video_bitrate])
        
        if self.audio_bitrate:
            args.extend(["-b:a", self.audio_bitrate])
        
        # Additional parameters
        for param, value in self.additional_params.items():
            args.extend([f"-{param}", value])
        
        return args


# Default FFmpeg presets for different quality levels
DEFAULT_PRESETS = {
    QualityPreset.LOW: FFmpegPreset(
        name="low",
        video_codec="libx264",
        audio_codec="aac",
        video_bitrate="800k",
        audio_bitrate="96k",
        additional_params={"preset": "fast", "crf": "28"}
    ),
    QualityPreset.MEDIUM: FFmpegPreset(
        name="medium",
        video_codec="libx264",
        audio_codec="aac",
        video_bitrate="1500k",
        audio_bitrate="128k",
        additional_params={"preset": "medium", "crf": "23"}
    ),
    QualityPreset.HIGH: FFmpegPreset(
        name="high",
        video_codec="libx264",
        audio_codec="aac",
        video_bitrate="3000k",
        audio_bitrate="192k",
        additional_params={"preset": "slow", "crf": "20"}
    ),
    QualityPreset.ULTRA: FFmpegPreset(
        name="ultra",
        video_codec="libx264",
        audio_codec="aac",
        video_bitrate="6000k",
        audio_bitrate="320k",
        additional_params={"preset": "veryslow", "crf": "18"}
    )
}


@dataclass
class VideoInfo:
    """
    Represents information about a video file.
    """
    path: str
    format: str
    duration: float
    size: int  # in bytes
    bitrate: int  # in bits per second
    video_streams: List[Dict[str, Any]] = field(default_factory=list)
    audio_streams: List[Dict[str, Any]] = field(default_factory=list)
    subtitle_streams: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert video info to dictionary format."""
        return {
            "path": self.path,
            "format": self.format,
            "duration": self.duration,
            "size": self.size,
            "bitrate": self.bitrate,
            "video_streams": self.video_streams,
            "audio_streams": self.audio_streams,
            "subtitle_streams": self.subtitle_streams,
            "formatted_duration": self.formatted_duration,
            "formatted_size": self.formatted_size,
            "formatted_bitrate": self.formatted_bitrate
        }
    
    @property
    def formatted_duration(self) -> str:
        """Format duration as HH:MM:SS."""
        hours, remainder = divmod(self.duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    @property
    def formatted_size(self) -> str:
        """Format size in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if self.size < 1024:
                return f"{self.size:.2f} {unit}"
            self.size /= 1024
        return f"{self.size:.2f} PB"
    
    @property
    def formatted_bitrate(self) -> str:
        """Format bitrate in human-readable form."""
        if self.bitrate < 1000:
            return f"{self.bitrate} bps"
        elif self.bitrate < 1000000:
            return f"{self.bitrate/1000:.2f} Kbps"
        else:
            return f"{self.bitrate/1000000:.2f} Mbps" 