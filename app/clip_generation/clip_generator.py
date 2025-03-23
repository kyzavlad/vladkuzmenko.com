"""
Clip Generator Module

Handles the extraction of clips from source videos using FFmpeg.
"""

import os
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClipGenerator:
    """
    Handles the extraction of clips from source videos.
    
    This class provides functionality to extract segments from videos,
    apply basic filters, and generate output files.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None
    ):
        """
        Initialize the clip generator.
        
        Args:
            output_dir: Directory for output files
            temp_dir: Directory for temporary files
            ffmpeg_path: Path to ffmpeg binary
            ffprobe_path: Path to ffprobe binary
        """
        # Set up directories
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "clip_generation"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up FFmpeg paths
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        self.ffprobe_path = ffprobe_path or "ffprobe"
        
        logger.info(f"Initialized ClipGenerator")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Temporary directory: {self.temp_dir}")
        logger.info(f"FFmpeg path: {self.ffmpeg_path}")
        logger.info(f"FFprobe path: {self.ffprobe_path}")
    
    def extract_clip(
        self,
        source_video: str,
        output_file: str,
        start_time: Union[float, str],
        end_time: Union[float, str],
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        preset: str = "medium",
        crf: int = 23,
        additional_options: Optional[List[str]] = None
    ) -> bool:
        """
        Extract a clip from a source video.
        
        Args:
            source_video: Path to source video
            output_file: Path to output file
            start_time: Start time in seconds or HH:MM:SS.mmm format
            end_time: End time in seconds or HH:MM:SS.mmm format
            video_codec: Video codec to use
            audio_codec: Audio codec to use
            preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            crf: Constant Rate Factor (0-51, lower is better quality)
            additional_options: Additional FFmpeg options
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(source_video):
            logger.error(f"Source video not found: {source_video}")
            return False
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format start and end times if they're numeric
        if isinstance(start_time, (int, float)):
            start_time = str(start_time)
        
        if isinstance(end_time, (int, float)):
            end_time = str(end_time)
        
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", source_video,
            "-ss", start_time,
            "-to", end_time,
            "-c:v", video_codec,
            "-c:a", audio_codec,
            "-preset", preset,
            "-crf", str(crf),
            "-y"  # Overwrite output file if it exists
        ]
        
        # Add any additional options
        if additional_options:
            cmd.extend(additional_options)
        
        # Add output file
        cmd.append(output_file)
        
        logger.info(f"Extracting clip: {start_time} to {end_time}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            # Run FFmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr}")
                return False
            
            if not os.path.exists(output_file):
                logger.error(f"Output file not created: {output_file}")
                return False
            
            logger.info(f"Clip extracted successfully: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting clip: {str(e)}")
            return False
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a video file using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if an error occurred
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Build FFprobe command to get JSON output
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            # Run FFprobe
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFprobe error: {stderr}")
                return None
            
            # Parse JSON output
            import json
            video_info = json.loads(stdout)
            
            # Extract key information
            result = {
                "format": video_info.get("format", {}),
                "duration": float(video_info.get("format", {}).get("duration", 0)),
                "size": int(video_info.get("format", {}).get("size", 0)),
                "bit_rate": int(video_info.get("format", {}).get("bit_rate", 0)),
                "streams": []
            }
            
            # Extract stream information
            for stream in video_info.get("streams", []):
                stream_type = stream.get("codec_type")
                
                if stream_type == "video":
                    result["streams"].append({
                        "type": "video",
                        "codec": stream.get("codec_name"),
                        "width": stream.get("width"),
                        "height": stream.get("height"),
                        "fps": eval(stream.get("r_frame_rate", "0/1")),  # Convert fraction to float
                        "bit_rate": int(stream.get("bit_rate", 0)),
                        "duration": float(stream.get("duration", 0))
                    })
                elif stream_type == "audio":
                    result["streams"].append({
                        "type": "audio",
                        "codec": stream.get("codec_name"),
                        "channels": stream.get("channels"),
                        "sample_rate": int(stream.get("sample_rate", 0)),
                        "bit_rate": int(stream.get("bit_rate", 0)),
                        "duration": float(stream.get("duration", 0))
                    })
            
            logger.debug(f"Video info retrieved: {video_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return None
    
    def combine_clips(
        self,
        input_clips: List[str],
        output_file: str,
        transition_duration: float = 0.5,
        crossfade_audio: bool = True,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        additional_options: Optional[List[str]] = None
    ) -> bool:
        """
        Combine multiple clips into a single video with transitions.
        
        Args:
            input_clips: List of paths to input clip files
            output_file: Path to output file
            transition_duration: Duration of transition between clips in seconds
            crossfade_audio: Whether to crossfade audio between clips
            video_codec: Video codec to use
            audio_codec: Audio codec to use
            additional_options: Additional FFmpeg options
            
        Returns:
            True if successful, False otherwise
        """
        if not input_clips:
            logger.error("No input clips provided")
            return False
        
        # Create temporary file list
        list_file = self.temp_dir / f"clip_list_{os.path.basename(output_file)}.txt"
        
        with open(list_file, "w") as f:
            for clip in input_clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")
        
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file)
        ]
        
        # Add transition filter if requested
        if transition_duration > 0 and len(input_clips) > 1:
            # This is a simplified approach - complex transitions would require more complex filter chains
            cmd.extend([
                "-filter_complex", f"xfade=transition=fade:duration={transition_duration}"
            ])
        
        # Add audio crossfade if requested
        if crossfade_audio and len(input_clips) > 1:
            # Note: This is simplified and may need to be adjusted based on actual requirements
            cmd.extend([
                "-filter_complex", f"acrossfade=d={transition_duration}"
            ])
        
        # Add codec options
        cmd.extend([
            "-c:v", video_codec,
            "-c:a", audio_codec,
            "-y"  # Overwrite output file if it exists
        ])
        
        # Add any additional options
        if additional_options:
            cmd.extend(additional_options)
        
        # Add output file
        cmd.append(output_file)
        
        logger.info(f"Combining {len(input_clips)} clips with {transition_duration}s transitions")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            # Run FFmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            # Clean up temporary file
            if os.path.exists(list_file):
                os.unlink(list_file)
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr}")
                return False
            
            if not os.path.exists(output_file):
                logger.error(f"Output file not created: {output_file}")
                return False
            
            logger.info(f"Clips combined successfully: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error combining clips: {str(e)}")
            return False 