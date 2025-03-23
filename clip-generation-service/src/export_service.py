from typing import List, Optional
import os
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from PIL import Image
import numpy as np
from .models import Platform, CaptionStyle, Clip

class ExportService:
    def __init__(self):
        self.platform_settings = {
            Platform.TIKTOK: {
                "max_duration": 180,
                "resolution": (1080, 1920),
                "bitrate": "2M"
            },
            Platform.INSTAGRAM: {
                "max_duration": 60,
                "resolution": (1080, 1920),
                "bitrate": "3.5M"
            },
            Platform.YOUTUBE: {
                "max_duration": 60 * 60,  # 1 hour
                "resolution": (1920, 1080),
                "bitrate": "8M"
            }
        }
        
        self.caption_styles = {
            CaptionStyle.DEFAULT: {
                "font": "Arial",
                "fontsize": 30,
                "color": "white",
                "stroke_color": "black",
                "stroke_width": 2
            },
            CaptionStyle.MINIMAL: {
                "font": "Helvetica",
                "fontsize": 24,
                "color": "white",
                "stroke_color": None,
                "stroke_width": 0
            },
            CaptionStyle.BOLD: {
                "font": "Arial-Bold",
                "fontsize": 36,
                "color": "white",
                "stroke_color": "black",
                "stroke_width": 3
            },
            CaptionStyle.COLORFUL: {
                "font": "Arial",
                "fontsize": 32,
                "color": "yellow",
                "stroke_color": "blue",
                "stroke_width": 2
            }
        }
    
    def _add_captions(
        self,
        video: VideoFileClip,
        text: str,
        style: CaptionStyle
    ) -> VideoFileClip:
        """Add captions to video with specified style."""
        style_settings = self.caption_styles[style]
        
        text_clip = TextClip(
            text,
            font=style_settings["font"],
            fontsize=style_settings["fontsize"],
            color=style_settings["color"],
            stroke_color=style_settings["stroke_color"],
            stroke_width=style_settings["stroke_width"]
        )
        
        # Position at bottom center
        text_clip = text_clip.set_position(("center", "bottom"))
        text_clip = text_clip.set_duration(video.duration)
        
        return CompositeVideoClip([video, text_clip])
    
    def _add_watermark(
        self,
        video: VideoFileClip,
        watermark_path: str,
        position: tuple = ("right", "bottom"),
        opacity: float = 0.7
    ) -> VideoFileClip:
        """Add watermark to video."""
        # Load and resize watermark
        watermark = Image.open(watermark_path)
        max_size = min(video.w // 4, video.h // 4)
        watermark.thumbnail((max_size, max_size))
        
        # Convert to numpy array
        watermark_array = np.array(watermark)
        if watermark_array.shape[2] == 4:  # If RGBA
            # Apply opacity
            watermark_array[..., 3] = watermark_array[..., 3] * opacity
        
        watermark_clip = VideoFileClip(watermark_array).set_duration(video.duration)
        watermark_clip = watermark_clip.set_position(position)
        
        return CompositeVideoClip([video, watermark_clip])
    
    def _optimize_for_platform(
        self,
        video: VideoFileClip,
        platform: Platform
    ) -> VideoFileClip:
        """Apply platform-specific optimizations."""
        settings = self.platform_settings[platform]
        
        # Resize to platform resolution
        video = video.resize(settings["resolution"])
        
        # Trim if exceeds max duration
        if video.duration > settings["max_duration"]:
            video = video.subclip(0, settings["max_duration"])
        
        return video
    
    def export_clip(
        self,
        clip: Clip,
        platform: Platform,
        output_dir: str,
        include_captions: bool = True,
        caption_style: CaptionStyle = CaptionStyle.DEFAULT,
        add_watermark: bool = False,
        watermark_path: Optional[str] = None
    ) -> str:
        """
        Export a single clip with platform-specific optimizations.
        
        Args:
            clip (Clip): Clip to export
            platform (Platform): Target platform
            output_dir (str): Output directory
            include_captions (bool): Whether to add captions
            caption_style (CaptionStyle): Caption style
            add_watermark (bool): Whether to add watermark
            watermark_path (Optional[str]): Path to watermark image
            
        Returns:
            str: Path to exported video
        """
        # Load video
        video = VideoFileClip(clip.download_url)
        
        # Apply platform optimizations
        video = self._optimize_for_platform(video, platform)
        
        # Add captions if requested
        if include_captions and clip.transcript_segment:
            video = self._add_captions(video, clip.transcript_segment, caption_style)
        
        # Add watermark if requested
        if add_watermark and watermark_path:
            video = self._add_watermark(video, watermark_path)
        
        # Export with platform settings
        settings = self.platform_settings[platform]
        output_path = os.path.join(
            output_dir,
            f"{clip.clip_id}_{platform.value}.mp4"
        )
        
        video.write_videofile(
            output_path,
            codec="libx264",
            bitrate=settings["bitrate"],
            audio_codec="aac",
            audio_bitrate="192k"
        )
        
        return output_path
    
    def batch_export(
        self,
        clips: List[Clip],
        platform: Platform,
        output_dir: str,
        include_captions: bool = True,
        caption_style: CaptionStyle = CaptionStyle.DEFAULT,
        add_watermark: bool = False,
        watermark_path: Optional[str] = None
    ) -> List[str]:
        """
        Export multiple clips with platform-specific optimizations.
        
        Args:
            clips (List[Clip]): Clips to export
            platform (Platform): Target platform
            output_dir (str): Output directory
            include_captions (bool): Whether to add captions
            caption_style (CaptionStyle): Caption style
            add_watermark (bool): Whether to add watermark
            watermark_path (Optional[str]): Path to watermark image
            
        Returns:
            List[str]: Paths to exported videos
        """
        exported_paths = []
        
        for clip in clips:
            try:
                path = self.export_clip(
                    clip,
                    platform,
                    output_dir,
                    include_captions,
                    caption_style,
                    add_watermark,
                    watermark_path
                )
                exported_paths.append(path)
            except Exception as e:
                print(f"Error exporting clip {clip.clip_id}: {str(e)}")
                continue
        
        return exported_paths 