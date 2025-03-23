"""
Clip Generator Module

This module provides functionality for generating video clips from source videos
using various parameters and transformations.
"""

import os
import time
import logging
import subprocess
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClipGenerator:
    """
    Handles the actual video processing for clip generation.
    
    This class encapsulates the video processing logic for extracting clips
    from source videos with various transformations and parameters.
    """
    
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        temp_dir: str = "temp/clips",
        default_codec: str = "h264",
        default_preset: str = "medium"
    ):
        """
        Initialize the clip generator.
        
        Args:
            ffmpeg_path: Path to the ffmpeg executable
            temp_dir: Directory for storing temporary files
            default_codec: Default video codec to use
            default_preset: Default encoding preset to use
        """
        self.ffmpeg_path = ffmpeg_path
        self.temp_dir = temp_dir
        self.default_codec = default_codec
        self.default_preset = default_preset
        
        # Ensure temp directory exists
        os.makedirs(temp_dir, exist_ok=True)
        
        # Quality presets mapping
        self.quality_presets = {
            "low": {"crf": 28, "preset": "veryfast", "audio_bitrate": "96k"},
            "medium": {"crf": 23, "preset": "medium", "audio_bitrate": "128k"},
            "high": {"crf": 18, "preset": "slow", "audio_bitrate": "192k"},
            "ultra": {"crf": 14, "preset": "slower", "audio_bitrate": "256k"}
        }
        
        logger.info(f"ClipGenerator initialized")
        logger.info(f"FFMPEG path: {ffmpeg_path}")
    
    def generate_clip(
        self,
        source_video: str,
        output_path: str,
        start_time: float,
        end_time: float,
        quality: str = "medium",
        format: Optional[str] = None,
        codec: Optional[str] = None,
        resize: Optional[Tuple[int, int]] = None,
        audio: bool = True,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a video clip from the source video.
        
        Args:
            source_video: Path to the source video
            output_path: Path for the output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            quality: Quality preset (low, medium, high, ultra)
            format: Output format (e.g., mp4, avi, mov)
            codec: Video codec to use
            resize: Optional tuple of (width, height) for resizing
            audio: Whether to include audio in the output
            additional_params: Additional parameters for ffmpeg
            
        Returns:
            Path to the generated clip
            
        Raises:
            FileNotFoundError: If the source video doesn't exist
            RuntimeError: If clip generation fails
        """
        # Validate source video
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        # Get format from output path if not specified
        if format is None:
            format = os.path.splitext(output_path)[1].lstrip(".")
            if not format:
                format = "mp4"  # Default to mp4 if no extension
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get quality settings
        quality_settings = self.quality_presets.get(quality, self.quality_presets["medium"])
        
        # Set codec
        codec = codec or self.default_codec
        
        # Build ffmpeg command
        command = [
            self.ffmpeg_path,
            "-y",  # Overwrite output file if it exists
            "-ss", str(start_time),  # Start time
            "-i", source_video,  # Input file
            "-t", str(end_time - start_time),  # Duration
            "-c:v", codec,  # Video codec
            "-crf", str(quality_settings["crf"]),  # Quality
            "-preset", quality_settings.get("preset", self.default_preset)  # Encoding speed/compression tradeoff
        ]
        
        # Add resize parameter if specified
        if resize:
            width, height = resize
            command.extend(["-vf", f"scale={width}:{height}"])
        
        # Handle audio
        if audio:
            command.extend([
                "-c:a", "aac",
                "-b:a", quality_settings.get("audio_bitrate", "128k")
            ])
        else:
            command.extend(["-an"])  # No audio
        
        # Add format if not mp4 (default)
        if format != "mp4":
            command.extend(["-f", format])
        
        # Add output path
        command.append(output_path)
        
        # Log the command
        logger.info(f"Executing command: {' '.join(command)}")
        
        try:
            # Execute ffmpeg command
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file not generated: {output_path}")
            
            logger.info(f"Successfully generated clip: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFMPEG error: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to generate clip: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating clip: {str(e)}")
            raise
    
    def extract_frame(
        self,
        source_video: str,
        output_path: str,
        timestamp: float,
        format: str = "jpg",
        quality: int = 90,
        resize: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Extract a single frame from a video at the specified timestamp.
        
        Args:
            source_video: Path to the source video
            output_path: Path for the output frame image
            timestamp: Time in seconds for the frame to extract
            format: Output image format (jpg, png, etc.)
            quality: Image quality (0-100) for lossy formats
            resize: Optional tuple of (width, height) for resizing
            
        Returns:
            Path to the extracted frame
            
        Raises:
            FileNotFoundError: If the source video doesn't exist
            RuntimeError: If frame extraction fails
        """
        # Validate source video
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Build ffmpeg command
        command = [
            self.ffmpeg_path,
            "-y",  # Overwrite output file if it exists
            "-ss", str(timestamp),  # Timestamp
            "-i", source_video,  # Input file
            "-frames:v", "1"  # Extract one frame
        ]
        
        # Add resize parameter if specified
        if resize:
            width, height = resize
            command.extend(["-vf", f"scale={width}:{height}"])
        
        # Add quality parameter for jpeg
        if format.lower() in ["jpg", "jpeg"]:
            command.extend(["-q:v", str(int(quality / 10))])  # Convert 0-100 to ffmpeg's 1-10 scale
        
        # Add output path
        command.append(output_path)
        
        # Log the command
        logger.info(f"Executing frame extraction command: {' '.join(command)}")
        
        try:
            # Execute ffmpeg command
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output frame not generated: {output_path}")
            
            logger.info(f"Successfully extracted frame: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFMPEG error: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to extract frame: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error extracting frame: {str(e)}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information
            
        Raises:
            FileNotFoundError: If the video doesn't exist
            RuntimeError: If information extraction fails
        """
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Build ffprobe command
        command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            # Execute ffprobe command
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Parse JSON output
            import json
            info = json.loads(result.stdout)
            
            # Extract relevant information
            video_info = {
                "format": info.get("format", {}).get("format_name", "unknown"),
                "duration": float(info.get("format", {}).get("duration", 0)),
                "size": int(info.get("format", {}).get("size", 0)),
                "bitrate": int(info.get("format", {}).get("bit_rate", 0)),
                "streams": []
            }
            
            # Extract stream information
            for stream in info.get("streams", []):
                stream_type = stream.get("codec_type")
                if stream_type == "video":
                    video_info["streams"].append({
                        "type": "video",
                        "codec": stream.get("codec_name"),
                        "width": stream.get("width"),
                        "height": stream.get("height"),
                        "fps": eval(stream.get("r_frame_rate", "0/1")),  # Convert fraction to float
                        "bitrate": int(stream.get("bit_rate", 0))
                    })
                elif stream_type == "audio":
                    video_info["streams"].append({
                        "type": "audio",
                        "codec": stream.get("codec_name"),
                        "channels": stream.get("channels"),
                        "sample_rate": stream.get("sample_rate"),
                        "bitrate": int(stream.get("bit_rate", 0))
                    })
            
            return video_info
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFPROBE error: {e.stderr if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to get video info: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise 