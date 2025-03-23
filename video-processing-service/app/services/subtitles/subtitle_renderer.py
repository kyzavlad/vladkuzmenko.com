import logging
import asyncio
import os
import json
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

from .subtitle_generator import SubtitleFormat, SubtitleStyle, SubtitleGenerator


class RenderQuality(str, Enum):
    """Video render quality options."""
    LOW = "low"           # Fast rendering, lower quality
    MEDIUM = "medium"     # Balanced quality/speed
    HIGH = "high"         # High quality, slower
    ORIGINAL = "original" # Maintain original video quality


class SubtitleRenderer:
    """
    Renders subtitles onto videos using FFmpeg.
    
    Features:
    - Burn subtitles directly into video files
    - Generate preview images with subtitles
    - Multiple quality settings for output videos
    - Configurable subtitle rendering options
    """
    
    def __init__(
        self,
        subtitle_generator: Optional[SubtitleGenerator] = None,
        default_quality: RenderQuality = RenderQuality.MEDIUM,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the subtitle renderer.
        
        Args:
            subtitle_generator: SubtitleGenerator instance for generating subtitle files
            default_quality: Default render quality
            ffmpeg_path: Path to FFmpeg executable
            ffprobe_path: Path to FFprobe executable
            config: Additional configuration options
        """
        self.subtitle_generator = subtitle_generator or SubtitleGenerator()
        self.default_quality = default_quality
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.config = config or {}
        
        # Temporary directory for intermediate files
        self.temp_dir = self.config.get("temp_dir", tempfile.gettempdir())
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Quality presets for FFmpeg
        self.quality_presets = {
            RenderQuality.LOW: {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "video_bitrate": "1000k",
                "audio_bitrate": "128k",
                "preset": "ultrafast",
                "scale": "-1:720"  # 720p
            },
            RenderQuality.MEDIUM: {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "video_bitrate": "2500k",
                "audio_bitrate": "192k",
                "preset": "medium",
                "scale": "-1:1080"  # 1080p
            },
            RenderQuality.HIGH: {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "video_bitrate": "5000k",
                "audio_bitrate": "256k",
                "preset": "slow",
                "scale": "-1:1080"  # 1080p, higher bitrate
            },
            RenderQuality.ORIGINAL: {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "preset": "medium",
                # No bitrate or scale settings - will maintain original
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def render_video_with_subtitles(
        self,
        video_path: str,
        transcript: Dict[str, Any],
        output_path: str,
        style_name: Optional[str] = None,
        custom_style: Optional[SubtitleStyle] = None,
        quality: Optional[RenderQuality] = None,
        background_blur: float = 0.0,
        force_text_color: bool = False,
        show_progress_callback: Optional[callable] = None
    ) -> str:
        """
        Render video with burned-in subtitles.
        
        Args:
            video_path: Path to input video file
            transcript: Transcript with timing information
            output_path: Path to save the output video
            style_name: Name of subtitle style to use
            custom_style: Custom style overrides
            quality: Render quality setting
            background_blur: Amount of background blur to apply behind text (0.0-1.0)
            force_text_color: Whether to force text color even with background
            show_progress_callback: Callback function to report progress
            
        Returns:
            Path to the rendered video file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        quality = quality or self.default_quality
        
        # Get video information
        video_info = await self._get_video_info(video_path)
        
        # Create a temporary subtitle file in ASS format
        subtitle_path = os.path.join(
            self.temp_dir,
            f"{os.path.basename(video_path)}_subtitles.ass"
        )
        
        # Generate subtitle file
        await self.subtitle_generator.generate_subtitles(
            transcript=transcript,
            output_path=subtitle_path,
            format=SubtitleFormat.ASS,
            style_name=style_name,
            custom_style=custom_style
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            # Build FFmpeg command for rendering
            cmd = await self._build_render_command(
                video_path=video_path,
                subtitle_path=subtitle_path,
                output_path=output_path,
                quality=quality,
                video_info=video_info,
                background_blur=background_blur,
                force_text_color=force_text_color
            )
            
            self.logger.info(f"Rendering video with subtitles: {' '.join(cmd)}")
            
            # Execute the FFmpeg command with progress reporting
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Process output and track progress
            if show_progress_callback:
                await self._track_progress(process, video_info, show_progress_callback)
            else:
                await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"FFmpeg process returned non-zero exit code: {process.returncode}")
            
            self.logger.info(f"Successfully rendered video with subtitles: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error rendering video with subtitles: {str(e)}")
            raise
        finally:
            # Clean up temporary subtitle file
            if os.path.exists(subtitle_path):
                os.remove(subtitle_path)
    
    async def generate_preview_image(
        self,
        video_path: str,
        transcript: Dict[str, Any],
        output_path: str,
        time_position: float = 0.0,
        style_name: Optional[str] = None,
        custom_style: Optional[SubtitleStyle] = None,
        width: int = 1280
    ) -> str:
        """
        Generate a preview image from the video with subtitle overlay.
        
        Args:
            video_path: Path to input video file
            transcript: Transcript with timing information
            output_path: Path to save the preview image
            time_position: Time position in the video to capture (seconds)
            style_name: Name of subtitle style to use
            custom_style: Custom style overrides
            width: Width of the output image
            
        Returns:
            Path to the generated preview image
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video information
        video_info = await self._get_video_info(video_path)
        
        # Ensure time_position is within the video duration
        duration = float(video_info.get("duration", 0))
        if time_position > duration:
            time_position = duration / 2  # Use middle of video if position is beyond duration
        
        # Find the subtitle text that should appear at this time
        subtitle_text = self._get_subtitle_at_time(transcript, time_position)
        
        if not subtitle_text:
            # If no subtitle at this time, find a frame that has subtitle text
            for segment in transcript.get("segments", []):
                if segment.get("text", "").strip():
                    time_position = (segment.get("start", 0) + segment.get("end", 0)) / 2
                    subtitle_text = segment.get("text", "").strip()
                    break
        
        # Create a temporary subtitle file in ASS format
        subtitle_path = os.path.join(
            self.temp_dir,
            f"{os.path.basename(video_path)}_preview_subtitles.ass"
        )
        
        # Create a single-subtitle transcript for this frame
        frame_transcript = {
            "segments": [
                {
                    "start": time_position,
                    "end": time_position + 5,  # 5 seconds duration
                    "text": subtitle_text
                }
            ]
        }
        
        # Generate subtitle file
        await self.subtitle_generator.generate_subtitles(
            transcript=frame_transcript,
            output_path=subtitle_path,
            format=SubtitleFormat.ASS,
            style_name=style_name,
            custom_style=custom_style
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            # Format time position for FFmpeg
            time_str = self._format_time_position(time_position)
            
            # Build FFmpeg command for extracting frame with subtitle
            cmd = [
                self.ffmpeg_path,
                "-ss", time_str,
                "-i", video_path,
                "-vf", f"scale={width}:-1,subtitles='{subtitle_path}'",
                "-frames:v", "1",
                "-y",
                output_path
            ]
            
            self.logger.info(f"Generating preview image: {' '.join(cmd)}")
            
            # Execute the FFmpeg command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"FFmpeg error: {stderr.decode()}")
                raise Exception(f"Failed to generate preview image: {stderr.decode()}")
            
            self.logger.info(f"Successfully generated preview image: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating preview image: {str(e)}")
            raise
        finally:
            # Clean up temporary subtitle file
            if os.path.exists(subtitle_path):
                os.remove(subtitle_path)
    
    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about the video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            self.logger.error(f"FFprobe error: {stderr.decode()}")
            raise Exception(f"Failed to get video information: {stderr.decode()}")
        
        info = json.loads(stdout.decode())
        
        # Extract relevant information
        video_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        
        if not video_streams:
            raise ValueError("No video stream found in the input file")
        
        # Get the first video stream
        video_stream = video_streams[0]
        
        # Determine width and height
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        
        # Determine duration from format or video stream
        duration = info.get("format", {}).get("duration")
        if duration is None:
            duration = video_stream.get("duration")
        
        if duration is not None:
            duration = float(duration)
        else:
            duration = 0.0
        
        return {
            "width": width,
            "height": height,
            "duration": duration,
            "has_audio": len(audio_streams) > 0,
            "video_codec": video_stream.get("codec_name"),
            "frame_rate": self._parse_frame_rate(video_stream.get("r_frame_rate", "0/1")),
            "bit_rate": info.get("format", {}).get("bit_rate")
        }
    
    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """Parse frame rate string (e.g. '24000/1001') to float."""
        try:
            if "/" in frame_rate_str:
                num, den = frame_rate_str.split("/")
                return float(num) / float(den)
            else:
                return float(frame_rate_str)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    async def _build_render_command(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
        quality: RenderQuality,
        video_info: Dict[str, Any],
        background_blur: float = 0.0,
        force_text_color: bool = False
    ) -> List[str]:
        """
        Build FFmpeg command for rendering video with subtitles.
        
        Args:
            video_path: Path to input video
            subtitle_path: Path to subtitle file
            output_path: Path for output video
            quality: Render quality setting
            video_info: Video information dictionary
            background_blur: Amount of background blur behind text
            force_text_color: Whether to force text color
            
        Returns:
            FFmpeg command as list of strings
        """
        # Get quality preset
        preset = self.quality_presets.get(quality, self.quality_presets[RenderQuality.MEDIUM])
        
        # Start building the command
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-y"  # Overwrite output file if exists
        ]
        
        # Filter chain for video
        filters = []
        
        # Scaling if needed
        if quality != RenderQuality.ORIGINAL and "scale" in preset:
            filters.append(f"scale={preset['scale']}")
        
        # Add subtitles filter
        subtitle_filter = f"subtitles='{subtitle_path}'"
        
        # Apply background blur if requested
        if background_blur > 0:
            blur_amount = min(background_blur, 1.0) * 20  # Scale 0-1 to 0-20 for blur strength
            subtitle_filter = f"subtitles='{subtitle_path}':force_style='BackColour=&H00000000,BorderStyle=4,Shadow=0'"
            
        # Add the subtitle filter to the chain
        filters.append(subtitle_filter)
        
        # Apply filters if any
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        
        # Video codec and quality settings
        cmd.extend(["-c:v", preset.get("video_codec", "libx264")])
        
        if "preset" in preset:
            cmd.extend(["-preset", preset["preset"]])
        
        if "video_bitrate" in preset:
            cmd.extend(["-b:v", preset["video_bitrate"]])
        
        # Audio settings if video has audio
        if video_info.get("has_audio", False):
            cmd.extend([
                "-c:a", preset.get("audio_codec", "aac")
            ])
            
            if "audio_bitrate" in preset:
                cmd.extend(["-b:a", preset["audio_bitrate"]])
        
        # Add output path
        cmd.append(output_path)
        
        return cmd
    
    async def _track_progress(
        self,
        process: asyncio.subprocess.Process,
        video_info: Dict[str, Any],
        progress_callback: callable
    ) -> None:
        """
        Track FFmpeg progress and report via callback.
        
        Args:
            process: Subprocess instance
            video_info: Video information
            progress_callback: Callback function for progress updates
        """
        duration = video_info.get("duration", 0)
        
        while True:
            line = await process.stderr.readline()
            if not line:
                break
                
            line = line.decode().strip()
            
            # Parse FFmpeg progress information
            if "time=" in line:
                time_parts = line.split("time=")[1].split()[0].split(":")
                if len(time_parts) == 3:
                    hours, minutes, seconds = time_parts
                    seconds = float(seconds)
                    processed_time = (
                        int(hours) * 3600 + 
                        int(minutes) * 60 + 
                        seconds
                    )
                    
                    # Calculate progress percentage
                    if duration > 0:
                        progress = min(100, processed_time / duration * 100)
                    else:
                        progress = 0
                        
                    # Call progress callback
                    progress_callback(progress)
        
        # Wait for process to complete
        await process.wait()
    
    def _get_subtitle_at_time(self, transcript: Dict[str, Any], time_position: float) -> str:
        """
        Get subtitle text at a specific time position.
        
        Args:
            transcript: Transcript with timing information
            time_position: Time position in seconds
            
        Returns:
            Subtitle text at the specified time, or empty string if none
        """
        for segment in transcript.get("segments", []):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            if start_time <= time_position <= end_time:
                return segment.get("text", "").strip()
        
        return ""
    
    def _format_time_position(self, seconds: float) -> str:
        """
        Format time position for FFmpeg (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}" 