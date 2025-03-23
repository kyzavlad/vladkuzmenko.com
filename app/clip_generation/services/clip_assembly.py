"""
Clip Assembly and Optimization Module

This module provides functionality for advanced clip generation, including
smart assembly, vertical format optimization, and final processing pipeline.
"""

import os
import math
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClipAssemblyConfig:
    """Configuration for clip assembly and optimization."""
    # General settings
    ffmpeg_path: str = "ffmpeg"
    temp_dir: str = "temp/assembly"
    output_dir: str = "output/clips"
    
    # Smart clip generation settings
    min_clip_duration: float = 5.0
    max_clip_duration: float = 60.0
    target_min_duration: float = 15.0
    target_max_duration: float = 30.0
    enforce_absolute_max: float = 120.0  # Hard maximum
    
    # Content-aware transition settings
    crossfade_duration: float = 0.5
    fade_in_duration: float = 0.75
    fade_out_duration: float = 1.0
    
    # Vertical format settings
    enable_vertical_optimization: bool = True
    vertical_aspect_ratio: str = "9:16"
    horizontal_aspect_ratio: str = "16:9"
    smart_crop_detection: bool = True
    zoom_factor: float = 1.2
    
    # Final processing settings
    target_audio_lufs: float = -14.0
    enable_color_optimization: bool = True
    bitrate_target: str = "2M"
    enable_audio_normalization: bool = True


class ClipAssemblyOptimizer:
    """
    Handles advanced clip generation with optimization.
    
    This class provides functionality for:
    1. Smart clip generation with optimal duration and transitions
    2. Vertical format optimization for mobile viewing
    3. Final processing with audio normalization and platform-specific optimizations
    """
    
    def __init__(self, config: Optional[ClipAssemblyConfig] = None):
        """
        Initialize the clip assembly optimizer.
        
        Args:
            config: Configuration for clip assembly and optimization
        """
        self.config = config or ClipAssemblyConfig()
        
        # Ensure directories exist
        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ClipAssemblyOptimizer initialized")
        
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Build ffprobe command
        command = [
            self.config.ffmpeg_path.replace("ffmpeg", "ffprobe"),
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
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise
            
    def _detect_optimal_endpoints(
        self, 
        video_path: str, 
        base_start: float, 
        base_end: float, 
        expand_range: float = 2.0
    ) -> Tuple[float, float]:
        """
        Detect optimal start and end points for a clip.
        
        This looks for natural transitions like silence or scene changes
        within a small range around the specified start/end points.
        
        Args:
            video_path: Path to the video file
            base_start: Base start time in seconds
            base_end: Base end time in seconds
            expand_range: Range to search around base points (seconds)
            
        Returns:
            Tuple of (optimal_start, optimal_end) in seconds
        """
        # In a real implementation, this would analyze audio levels and scene changes
        # For this example, we'll use a simpler approach
        
        # Ensure we don't go below 0 for start time
        search_start = max(0, base_start - expand_range)
        search_end = base_end + expand_range
        
        # Get video info to check total duration
        try:
            video_info = self._get_video_info(video_path)
            video_duration = video_info["duration"]
            # Make sure search_end doesn't exceed video duration
            search_end = min(search_end, video_duration)
        except Exception as e:
            logger.warning(f"Error getting video duration: {str(e)}. Using original endpoints.")
            return (base_start, base_end)
        
        # Extract a temporary audio file for the search range
        temp_audio = self.temp_dir / f"temp_audio_{os.path.basename(video_path)}_{search_start}_{search_end}.wav"
        
        try:
            # Extract audio for analysis
            audio_cmd = [
                self.config.ffmpeg_path,
                "-y",
                "-ss", str(search_start),
                "-i", video_path,
                "-t", str(search_end - search_start),
                "-ac", "1",  # Mono
                "-ar", "16000",  # 16kHz
                "-vn",  # No video
                str(temp_audio)
            ]
            
            subprocess.run(audio_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Analyze audio levels to find silence
            # In a real implementation, this would use librosa or a similar library
            # For this example, we'll simulate the results
            
            # Get audio duration from the extracted file
            audio_info_cmd = [
                self.config.ffmpeg_path.replace("ffmpeg", "ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(temp_audio)
            ]
            
            audio_info = subprocess.run(
                audio_info_cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            audio_duration = float(audio_info.stdout.strip())
            
            # Simulate finding silence points
            # For a real implementation, use silencedetect filter
            silence_cmd = [
                self.config.ffmpeg_path,
                "-i", str(temp_audio),
                "-af", "silencedetect=noise=-30dB:d=0.5",
                "-f", "null",
                "-"
            ]
            
            silence_result = subprocess.run(
                silence_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Parse silence detection output
            silence_points = []
            for line in silence_result.stderr.split('\n'):
                if "silence_start" in line:
                    try:
                        silence_start = float(line.split("silence_start: ")[1].strip())
                        silence_points.append(search_start + silence_start)
                    except Exception:
                        pass
                elif "silence_end" in line:
                    try:
                        silence_end = float(line.split("silence_end: ")[1].split("|")[0].strip())
                        silence_points.append(search_start + silence_end)
                    except Exception:
                        pass
            
            # Find closest silence points to the base start and end
            optimal_start = base_start
            optimal_end = base_end
            
            if silence_points:
                # Find closest silence point to base_start
                start_candidates = [p for p in silence_points if p <= base_start]
                if start_candidates:
                    optimal_start = max(start_candidates)
                    
                # Find closest silence point to base_end
                end_candidates = [p for p in silence_points if p >= base_end]
                if end_candidates:
                    optimal_end = min(end_candidates)
            
            # Ensure minimum duration
            if optimal_end - optimal_start < self.config.min_clip_duration:
                # If too short, revert to base points
                logger.warning(f"Optimal endpoints too close: {optimal_start:.2f} - {optimal_end:.2f}, using base endpoints")
                return (base_start, base_end)
            
            logger.info(f"Original endpoints: {base_start:.2f} - {base_end:.2f}, Optimal endpoints: {optimal_start:.2f} - {optimal_end:.2f}")
            return (optimal_start, optimal_end)
            
        except Exception as e:
            logger.warning(f"Error detecting optimal endpoints: {str(e)}. Using original endpoints.")
            return (base_start, base_end)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass

    def optimize_clip_duration(
        self, 
        moments: List[Dict[str, Any]],
        target_duration: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize the duration of selected moments for clip generation.
        
        This ensures clips are within the desired duration range, and
        prioritizes higher-scored moments if adjustments are needed.
        
        Args:
            moments: List of detected moments with start_time, end_time, and combined_score
            target_duration: Optional target duration (if None, uses config settings)
            
        Returns:
            List of optimized moments
        """
        if not moments:
            return []
        
        # Sort moments by score (descending)
        sorted_moments = sorted(moments, key=lambda m: m["combined_score"], reverse=True)
        
        # Determine target duration range
        if target_duration:
            target_min = target_duration * 0.8  # Allow 20% below target
            target_max = target_duration * 1.2  # Allow 20% above target
        else:
            target_min = self.config.target_min_duration
            target_max = self.config.target_max_duration
        
        # Ensure absolute limits are respected
        min_duration = max(self.config.min_clip_duration, target_min)
        max_duration = min(self.config.max_clip_duration, target_max)
        
        # Process each moment
        optimized_moments = []
        
        for moment in sorted_moments:
            start_time = moment["start_time"]
            end_time = moment["end_time"]
            current_duration = end_time - start_time
            
            # Skip if below absolute minimum
            if current_duration < self.config.min_clip_duration:
                logger.warning(f"Skipping moment at {start_time:.2f} - too short: {current_duration:.2f}s")
                continue
                
            # Trim if above absolute maximum
            if current_duration > self.config.enforce_absolute_max:
                # Center around highest interest point
                mid_point = (start_time + end_time) / 2
                half_max = self.config.enforce_absolute_max / 2
                start_time = mid_point - half_max
                end_time = mid_point + half_max
                current_duration = end_time - start_time
                logger.info(f"Trimmed moment to maximum allowed duration: {current_duration:.2f}s")
                
            # Adjust if outside target range
            if current_duration < min_duration:
                # Extend the clip
                extra_needed = min_duration - current_duration
                start_time -= extra_needed / 2
                end_time += extra_needed / 2
                
                # Ensure start_time is not negative
                if start_time < 0:
                    start_time = 0
                    end_time = min_duration
                
                logger.info(f"Extended moment to meet minimum target: {min_duration:.2f}s")
                
            elif current_duration > max_duration:
                # Shrink the clip
                extra = current_duration - max_duration
                start_time += extra / 2
                end_time -= extra / 2
                logger.info(f"Shortened moment to meet maximum target: {max_duration:.2f}s")
            
            # Update the moment with new times
            moment["start_time"] = start_time
            moment["end_time"] = end_time
            moment["duration"] = end_time - start_time
            
            optimized_moments.append(moment)
            
        return optimized_moments
            
    def generate_smart_clip(
        self,
        source_video: str,
        output_path: str,
        start_time: float,
        end_time: float,
        optimize_endpoints: bool = True,
        vertical_format: bool = False,
        audio_normalize: bool = True
    ) -> str:
        """
        Generate a smart clip with optimized endpoints and processing.
        
        Args:
            source_video: Path to the source video
            output_path: Path for the output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            optimize_endpoints: Whether to optimize start/end points
            vertical_format: Whether to optimize for vertical viewing
            audio_normalize: Whether to normalize audio levels
            
        Returns:
            Path to the generated clip
        """
        # Validate source video
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Optimize endpoints if requested
        if optimize_endpoints:
            start_time, end_time = self._detect_optimal_endpoints(
                source_video, start_time, end_time
            )
        
        # Process based on format
        if vertical_format and self.config.enable_vertical_optimization:
            return self.generate_vertical_clip(
                source_video, 
                output_path, 
                start_time, 
                end_time, 
                audio_normalize
            )
        else:
            return self.generate_processed_clip(
                source_video, 
                output_path, 
                start_time, 
                end_time, 
                audio_normalize
            )
            
    def generate_processed_clip(
        self,
        source_video: str,
        output_path: str,
        start_time: float,
        end_time: float,
        audio_normalize: bool = True
    ) -> str:
        """
        Generate a processed clip with enhanced audio and video.
        
        Args:
            source_video: Path to the source video
            output_path: Path for the output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            audio_normalize: Whether to normalize audio levels
            
        Returns:
            Path to the generated clip
        """
        # Create temporary file for the raw clip
        temp_clip = self.temp_dir / f"temp_clip_{os.path.basename(output_path)}"
        
        try:
            # First extract the raw clip
            raw_clip_cmd = [
                self.config.ffmpeg_path,
                "-y",  # Overwrite output file if it exists
                "-ss", str(start_time),  # Start time
                "-i", source_video,  # Input file
                "-t", str(end_time - start_time),  # Duration
                "-c:v", "libx264",  # Video codec
                "-crf", "18",  # High quality for intermediate
                "-preset", "fast",  # Fast encoding for intermediate
                "-c:a", "aac",  # Audio codec
                "-b:a", "192k",  # Audio bitrate
                str(temp_clip)
            ]
            
            subprocess.run(raw_clip_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Now apply final processing
            final_cmd = [
                self.config.ffmpeg_path,
                "-y",  # Overwrite output file if it exists
                "-i", str(temp_clip)  # Input file
            ]
            
            # Handle color optimization
            if self.config.enable_color_optimization:
                # Add color processing filters
                color_filter = "eq=contrast=1.1:brightness=0.05:saturation=1.2"
                
                # Add video filter
                final_cmd.extend(["-vf", color_filter])
            
            # Handle audio normalization
            if audio_normalize and self.config.enable_audio_normalization:
                # Apply loudnorm filter for EBU R128 normalization
                audio_filter = (
                    f"loudnorm=I={self.config.target_audio_lufs}:LRA=11:TP=-1.5"
                )
                
                final_cmd.extend(["-af", audio_filter])
            
            # Add final encoding settings
            final_cmd.extend([
                "-c:v", "libx264",  # Video codec
                "-crf", "23",  # Quality (lower is better)
                "-preset", "medium",  # Encoding speed/compression tradeoff
                "-c:a", "aac",  # Audio codec
                "-b:a", "128k",  # Audio bitrate
                "-movflags", "+faststart",  # Web optimization
                output_path
            ])
            
            # Execute the final command
            subprocess.run(final_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file not generated: {output_path}")
            
            logger.info(f"Successfully generated processed clip: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFMPEG error: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to generate clip: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating clip: {str(e)}")
            raise
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_clip):
                try:
                    os.remove(temp_clip)
                except Exception:
                    pass
                    
    def generate_vertical_clip(
        self,
        source_video: str,
        output_path: str,
        start_time: float,
        end_time: float,
        audio_normalize: bool = True
    ) -> str:
        """
        Generate a vertical (9:16) optimized clip from a horizontal video.
        
        Args:
            source_video: Path to the source video
            output_path: Path for the output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            audio_normalize: Whether to normalize audio levels
            
        Returns:
            Path to the generated clip
        """
        # Create temporary file for the raw clip
        temp_clip = self.temp_dir / f"temp_clip_{os.path.basename(output_path)}"
        
        try:
            # Get video info to determine dimensions
            video_info = self._get_video_info(source_video)
            
            # Find the video stream
            video_stream = next((s for s in video_info["streams"] if s["type"] == "video"), None)
            
            if not video_stream:
                raise ValueError("No video stream found in source file")
                
            # Get original dimensions
            original_width = video_stream["width"]
            original_height = video_stream["height"]
            
            # Calculate target dimensions for 9:16 aspect ratio
            # If already vertical, just use original dimensions
            if original_height > original_width:
                target_width = original_width
                target_height = original_height
            else:
                # Convert to vertical
                # For 16:9 source to 9:16 target, we need to crop
                target_height = 16 * original_width / 9
                target_width = original_width
                
                # If target_height > original_height, we need to scale instead
                if target_height > original_height:
                    # Scale up width to match 9:16 ratio with original height
                    target_width = 9 * original_height / 16
                    target_height = original_height
            
            # First extract the raw clip
            raw_clip_cmd = [
                self.config.ffmpeg_path,
                "-y",  # Overwrite output file if it exists
                "-ss", str(start_time),  # Start time
                "-i", source_video,  # Input file
                "-t", str(end_time - start_time),  # Duration
                "-c:v", "libx264",  # Video codec
                "-crf", "18",  # High quality for intermediate
                "-preset", "fast",  # Fast encoding for intermediate
                "-c:a", "aac",  # Audio codec
                "-b:a", "192k",  # Audio bitrate
                str(temp_clip)
            ]
            
            subprocess.run(raw_clip_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Now apply vertical optimization with smart detection
            # Generate the vertical clip with appropriate cropping/scaling
            vertical_cmd = [
                self.config.ffmpeg_path,
                "-y",  # Overwrite output file if it exists
                "-i", str(temp_clip)  # Input file
            ]
            
            # Set up video filter for vertical format
            if original_height > original_width:
                # Already vertical, just apply zoom if needed
                if self.config.zoom_factor > 1.0:
                    # Apply zoompan filter
                    video_filter = (
                        f"scale={original_width}:{original_height},"
                        f"zoompan=z={self.config.zoom_factor}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                    )
                else:
                    video_filter = f"scale={original_width}:{original_height}"
            else:
                # Horizontal to vertical conversion
                if self.config.smart_crop_detection:
                    # In a real implementation, this would use face/object detection
                    # to determine the optimal crop area. Here we'll simulate it.
                    
                    # Assume the center is the most important region
                    crop_x = original_width / 2 - target_width / 2
                    
                    # Construct crop filter
                    video_filter = (
                        f"crop={target_width}:{target_height}:{crop_x}:0,"
                        f"scale={int(target_width)}:{int(target_height)}"
                    )
                else:
                    # Simple center crop
                    crop_x = original_width / 2 - target_width / 2
                    video_filter = (
                        f"crop={target_width}:{target_height}:{crop_x}:0,"
                        f"scale={int(target_width)}:{int(target_height)}"
                    )
            
            # Handle color optimization
            if self.config.enable_color_optimization:
                # Add color processing to video filter
                video_filter += ",eq=contrast=1.1:brightness=0.05:saturation=1.2"
            
            # Add video filter
            vertical_cmd.extend(["-vf", video_filter])
            
            # Handle audio normalization
            if audio_normalize and self.config.enable_audio_normalization:
                # Apply loudnorm filter for EBU R128 normalization
                audio_filter = (
                    f"loudnorm=I={self.config.target_audio_lufs}:LRA=11:TP=-1.5"
                )
                
                vertical_cmd.extend(["-af", audio_filter])
            
            # Add final encoding settings
            vertical_cmd.extend([
                "-c:v", "libx264",  # Video codec
                "-crf", "23",  # Quality (lower is better)
                "-preset", "medium",  # Encoding speed/compression tradeoff
                "-c:a", "aac",  # Audio codec
                "-b:a", "128k",  # Audio bitrate
                "-movflags", "+faststart",  # Web optimization
                output_path
            ])
            
            # Execute the final command
            subprocess.run(vertical_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file not generated: {output_path}")
            
            logger.info(f"Successfully generated vertical clip: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFMPEG error: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to generate vertical clip: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating vertical clip: {str(e)}")
            raise
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_clip):
                try:
                    os.remove(temp_clip)
                except Exception:
                    pass
    
    def assemble_multi_moment_clip(
        self,
        source_video: str,
        output_path: str,
        moments: List[Dict[str, Any]],
        vertical_format: bool = False,
        target_duration: Optional[float] = None,
        optimize_transitions: bool = True
    ) -> str:
        """
        Assemble multiple moments into a single cohesive clip.
        
        Args:
            source_video: Path to the source video
            output_path: Path for the output clip
            moments: List of detected moments with start_time and end_time
            vertical_format: Whether to optimize for vertical viewing
            target_duration: Optional target duration in seconds
            optimize_transitions: Whether to optimize transitions between segments
            
        Returns:
            Path to the assembled clip
        """
        if not moments:
            raise ValueError("No moments provided for clip assembly")
        
        # Validate source video
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Optimize durations if needed
        optimized_moments = self.optimize_clip_duration(moments, target_duration)
        
        if not optimized_moments:
            raise ValueError("No valid moments after optimization")
        
        # Sort moments by time
        sorted_moments = sorted(optimized_moments, key=lambda m: m["start_time"])
        
        # Create temp directory for segment clips
        temp_segments_dir = self.temp_dir / f"segments_{os.path.basename(output_path)}"
        os.makedirs(temp_segments_dir, exist_ok=True)
        
        segment_files = []
        segment_list_file = temp_segments_dir / "segments.txt"
        
        try:
            # Extract each segment
            for i, moment in enumerate(sorted_moments):
                start_time = moment["start_time"]
                end_time = moment["end_time"]
                
                # Create segment file
                segment_file = temp_segments_dir / f"segment_{i:03d}.mp4"
                segment_files.append(str(segment_file))
                
                # Extract segment with optimal endpoints if requested
                if optimize_transitions:
                    start_time, end_time = self._detect_optimal_endpoints(
                        source_video, start_time, end_time
                    )
                
                # Generate raw segment
                segment_cmd = [
                    self.config.ffmpeg_path,
                    "-y",  # Overwrite output file if it exists
                    "-ss", str(start_time),  # Start time
                    "-i", source_video,  # Input file
                    "-t", str(end_time - start_time),  # Duration
                    "-c:v", "libx264",  # Video codec
                    "-crf", "18",  # High quality for intermediate
                    "-preset", "fast",  # Fast encoding for intermediate
                    "-c:a", "aac",  # Audio codec
                    "-b:a", "192k",  # Audio bitrate
                ]
                
                # Apply vertical optimization if requested
                if vertical_format and self.config.enable_vertical_optimization:
                    # Get video info to determine dimensions
                    video_info = self._get_video_info(source_video)
                    
                    # Find the video stream
                    video_stream = next((s for s in video_info["streams"] if s["type"] == "video"), None)
                    
                    if video_stream:
                        # Get original dimensions
                        original_width = video_stream["width"]
                        original_height = video_stream["height"]
                        
                        # Calculate target dimensions for 9:16 aspect ratio
                        if original_height > original_width:
                            # Already vertical, just apply zoom if needed
                            if self.config.zoom_factor > 1.0:
                                # Apply zoompan filter
                                video_filter = (
                                    f"scale={original_width}:{original_height},"
                                    f"zoompan=z={self.config.zoom_factor}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                                )
                            else:
                                video_filter = f"scale={original_width}:{original_height}"
                        else:
                            # Horizontal to vertical conversion
                            # Simple center crop for segments
                            target_height = 16 * original_width / 9
                            target_width = original_width
                            
                            if target_height > original_height:
                                target_width = 9 * original_height / 16
                                target_height = original_height
                            
                            crop_x = original_width / 2 - target_width / 2
                            video_filter = (
                                f"crop={target_width}:{target_height}:{crop_x}:0,"
                                f"scale={int(target_width)}:{int(target_height)}"
                            )
                        
                        segment_cmd.extend(["-vf", video_filter])
                
                # Add output path
                segment_cmd.append(str(segment_file))
                
                # Execute segment extraction
                subprocess.run(segment_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Create segment list file for ffmpeg concat
            with open(segment_list_file, 'w') as f:
                for segment in segment_files:
                    f.write(f"file '{segment}'\n")
            
            # Assemble final clip
            assembly_cmd = [
                self.config.ffmpeg_path,
                "-y",  # Overwrite output file if it exists
                "-f", "concat",  # Concatenation
                "-safe", "0",  # Allow absolute paths
                "-i", str(segment_list_file),  # Input segments list
            ]
            
            # Handle transitions
            if optimize_transitions and len(segment_files) > 1:
                # For transitions between segments, we would ideally use complex
                # filtergraph with crossfade. For simplicity in this example:
                # Just use the simple concat demuxer
                pass
            
            # Handle final processing
            
            # Add color optimization if enabled
            if self.config.enable_color_optimization:
                assembly_cmd.extend([
                    "-vf", "eq=contrast=1.1:brightness=0.05:saturation=1.2"
                ])
            
            # Add audio normalization if enabled
            if self.config.enable_audio_normalization:
                assembly_cmd.extend([
                    "-af", f"loudnorm=I={self.config.target_audio_lufs}:LRA=11:TP=-1.5"
                ])
            
            # Add final encoding settings
            assembly_cmd.extend([
                "-c:v", "libx264",  # Video codec
                "-crf", "23",  # Quality (lower is better)
                "-preset", "medium",  # Encoding speed/compression tradeoff
                "-c:a", "aac",  # Audio codec
                "-b:a", "128k",  # Audio bitrate
                "-movflags", "+faststart",  # Web optimization
                output_path
            ])
            
            # Execute the assembly command
            subprocess.run(assembly_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file not generated: {output_path}")
            
            logger.info(f"Successfully assembled multi-moment clip: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFMPEG error: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to assemble clip: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error assembling clip: {str(e)}")
            raise
            
        finally:
            # Clean up temporary files
            for segment in segment_files:
                try:
                    if os.path.exists(segment):
                        os.remove(segment)
                except Exception:
                    pass
            
            try:
                if os.path.exists(segment_list_file):
                    os.remove(segment_list_file)
            except Exception:
                pass 