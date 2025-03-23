"""
Lightweight subtitle positioning service focused on basic video analysis.

This module provides a simplified version of the SubtitlePositioningService
with fewer dependencies and faster runtime performance, suitable for 
environments without advanced computer vision libraries.
"""

import logging
import os
import numpy as np
import cv2
import tempfile
import asyncio
import subprocess
import json
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from pathlib import Path


class ContentImportance(Enum):
    """Levels of importance for detected visual content."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ContentRegion:
    """Represents a region of interest in a video frame."""
    x: int
    y: int
    width: int
    height: int
    importance: ContentImportance
    confidence: float
    content_type: str


class SubtitlePositioningLite:
    """
    Lightweight version of subtitle positioning for optimizing subtitle placement.
    
    Features:
    - Simple face detection to avoid covering faces
    - Basic scene complexity analysis
    - Dynamic positioning between top/bottom based on content
    - Less resource-intensive than full positioning service
    """
    
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the lightweight subtitle positioning service.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable
            ffprobe_path: Path to FFprobe executable
            config: Configuration options
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.config = config or {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature detection
        self._initialize_detection()
        
        # Configure analysis parameters
        self._initialize_parameters()
        
        # Temporary directory for extracted frames
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def __del__(self):
        """Clean up temporary resources."""
        try:
            self.temp_dir.cleanup()
        except:
            pass
    
    def _initialize_detection(self):
        """Initialize simple detection capabilities."""
        self.enable_face_detection = self.config.get("enable_face_detection", True)
        self.enable_complexity_analysis = self.config.get("enable_complexity_analysis", True)
        
        # Load face detection model if enabled
        if self.enable_face_detection:
            try:
                # Use OpenCV's built-in face detector
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    self.logger.warning("Built-in face detection model not found, disabling face detection")
                    self.enable_face_detection = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize face detection: {str(e)}")
                self.enable_face_detection = False
        
        self.logger.info(f"Detection initialization complete. Face detection: {self.enable_face_detection}")
    
    def _initialize_parameters(self):
        """Initialize analysis parameters and thresholds."""
        # Sampling rate - lower means more frames analyzed (more accurate but slower)
        self.frame_sample_rate = self.config.get("frame_sample_rate", 48)  # Default: analyze every 48th frame (for 24fps = 1 frame every 2 seconds)
        
        # Minimum frames to analyze
        self.min_frames = self.config.get("min_frames", 10)
        
        # Maximum frames to analyze 
        self.max_frames = self.config.get("max_frames", 100)
        
        # Frame regions - define typical safe areas for subtitle placement
        # (normalized coordinates: 0-1)
        self.top_region = (0.0, 0.0, 1.0, 0.2)      # x, y, width, height
        self.bottom_region = (0.0, 0.8, 1.0, 0.2)
        
        # Default preference order for subtitle placement (bottom usually preferred)
        self.position_preference = self.config.get("position_preference", ["bottom", "top"])
    
    async def analyze_video_for_positioning(
        self,
        video_path: str,
        transcript: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze video content to determine optimal subtitle positioning.
        
        Args:
            video_path: Path to video file
            transcript: Transcript with timing information
            
        Returns:
            Updated transcript with position information for each segment
        """
        self.logger.info(f"Starting lightweight video analysis for subtitle positioning: {video_path}")
        
        # Verify the video file exists
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video info (duration, dimensions, etc.)
        video_info = await self._get_video_info(video_path)
        
        # Create a deep copy of the transcript to avoid modifying the original
        import copy
        processed_transcript = copy.deepcopy(transcript)
        
        try:
            # Extract key frames from the video
            key_frames = await self._extract_frames(video_path, video_info)
            self.logger.info(f"Extracted {len(key_frames)} frames for analysis")
            
            # For each segment in the transcript, determine the optimal position
            segments = processed_transcript.get("segments", [])
            
            for i, segment in enumerate(segments):
                # Get segment start and end times
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                
                # Skip segments with no text
                if not segment.get("text", "").strip():
                    continue
                
                # Find frames that fall within this segment's time range
                segment_frames = [
                    (frame_path, frame_time) for frame_path, frame_time in key_frames
                    if start_time <= frame_time <= end_time
                ]
                
                # If no extracted frames for this segment, use nearest frame
                if not segment_frames:
                    # Find the nearest frame to the segment midpoint
                    mid_time = (start_time + end_time) / 2
                    nearest_frame = min(key_frames, key=lambda frame: abs(frame[1] - mid_time))
                    segment_frames = [nearest_frame]
                
                # Analyze frames to identify important content
                frame_analysis_results = []
                for frame_path, _ in segment_frames:
                    regions = self._analyze_frame(frame_path, video_info)
                    frame_analysis_results.append(regions)
                
                # Determine optimal position for this segment based on content analysis
                position = self._determine_optimal_position(frame_analysis_results, video_info)
                
                # Add positioning data to the segment
                if "style" not in segment:
                    segment["style"] = {}
                
                segment["style"]["position"] = position
                
                # Log progress periodically
                if i % 10 == 0 or i == len(segments) - 1:
                    self.logger.debug(f"Processed positioning for {i+1}/{len(segments)} segments")
            
            self.logger.info(f"Completed subtitle positioning analysis for {len(segments)} segments")
            return processed_transcript
            
        except Exception as e:
            self.logger.error(f"Error analyzing video for subtitle positioning: {str(e)}")
            # Return original transcript in case of error
            return transcript
        finally:
            # Clean up extracted frames
            for frame_path, _ in key_frames:
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up frame file {frame_path}: {str(e)}")
    
    async def _extract_frames(
        self,
        video_path: str,
        video_info: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Extract frames from the video for analysis at regular intervals.
        
        Args:
            video_path: Path to video file
            video_info: Video information dictionary
            
        Returns:
            List of tuples containing (frame_path, timestamp)
        """
        duration = video_info.get("duration", 0)
        fps = video_info.get("fps", 30)
        
        if duration <= 0:
            self.logger.warning("Invalid video duration, using default value")
            duration = 60  # Assume 1 minute if duration unknown
        
        # Calculate the frame interval based on the sample rate
        interval = max(1, round(fps / self.frame_sample_rate))
        
        # Determine how many frames to extract
        num_frames = min(self.max_frames, max(self.min_frames, int(duration / (interval / fps))))
        
        # Create a temporary directory for frames
        frames_dir = os.path.join(self.temp_dir.name, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames using FFmpeg
        output_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
        
        cmd = [
            self.ffmpeg_path,
            "-nostdin",
            "-i", video_path,
            "-vf", f"select='not(mod(n,{interval}))',setpts=N/FRAME_RATE/TB",
            "-q:v", "3",  # Quality level (1-31, 1 is highest)
            "-frames:v", str(num_frames),
            "-y",
            output_pattern
        ]
        
        try:
            # Run the FFmpeg command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                stderr_str = stderr.decode('utf-8', errors='replace')
                self.logger.error(f"FFmpeg frame extraction failed: {stderr_str}")
                raise RuntimeError(f"Failed to extract frames: {stderr_str}")
            
            # Get the list of extracted frames with their timestamps
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
            
            # Calculate timestamp for each frame
            frame_data = []
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_dir, frame_file)
                # Calculate approximate timestamp based on frame number and interval
                timestamp = (i * interval) / fps
                frame_data.append((frame_path, timestamp))
            
            return frame_data
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            # Return an empty list in case of error
            return []
    
    def _analyze_frame(
        self,
        frame_path: str,
        video_info: Dict[str, Any]
    ) -> List[ContentRegion]:
        """
        Analyze a frame to identify important content regions.
        
        Args:
            frame_path: Path to frame image file
            video_info: Video information dictionary
            
        Returns:
            List of ContentRegion objects representing important areas
        """
        # Read the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            self.logger.warning(f"Failed to read frame: {frame_path}")
            return []
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # List to store detected content regions
        regions = []
        
        # Face detection (if enabled)
        if self.enable_face_detection:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    # Calculate importance based on size and position
                    size_ratio = (w * h) / (width * height)
                    center_dist = self._calculate_center_distance(x, y, w, h, width, height)
                    
                    # Faces are critical content that should not be covered
                    importance = ContentImportance.CRITICAL
                    confidence = 0.9 - center_dist + (size_ratio * 2)  # Higher confidence for centered, larger faces
                    
                    regions.append(ContentRegion(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        importance=importance,
                        confidence=min(1.0, max(0.5, confidence)),
                        content_type="face"
                    ))
            except Exception as e:
                self.logger.warning(f"Face detection error: {str(e)}")
        
        # Scene complexity analysis 
        if self.enable_complexity_analysis:
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply a simple edge detection
                edges = cv2.Canny(gray, 100, 200)
                
                # Analyze top half and bottom half of the frame
                top_half = edges[:height//2, :]
                bottom_half = edges[height//2:, :]
                
                # Count edge pixels in each half
                top_edge_count = np.count_nonzero(top_half)
                bottom_edge_count = np.count_nonzero(bottom_half)
                
                # Calculate complexity as percentage of edge pixels
                top_complexity = top_edge_count / (top_half.size)
                bottom_complexity = bottom_edge_count / (bottom_half.size)
                
                # Add regions based on complexity
                if top_complexity > 0.05:  # 5% threshold
                    regions.append(ContentRegion(
                        x=0,
                        y=0,
                        width=width,
                        height=height//2,
                        importance=ContentImportance.MEDIUM if top_complexity > 0.1 else ContentImportance.LOW,
                        confidence=min(0.9, top_complexity * 5),
                        content_type="complex_region_top"
                    ))
                
                if bottom_complexity > 0.05:  # 5% threshold
                    regions.append(ContentRegion(
                        x=0,
                        y=height//2,
                        width=width,
                        height=height//2,
                        importance=ContentImportance.MEDIUM if bottom_complexity > 0.1 else ContentImportance.LOW,
                        confidence=min(0.9, bottom_complexity * 5),
                        content_type="complex_region_bottom"
                    ))
            except Exception as e:
                self.logger.warning(f"Scene complexity analysis error: {str(e)}")
        
        return regions
    
    def _calculate_center_distance(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Calculate normalized distance of a region from the frame center.
        
        Args:
            x, y, w, h: Region coordinates and dimensions
            frame_width, frame_height: Frame dimensions
            
        Returns:
            Normalized distance from center (0-1, where 0 is center)
        """
        # Calculate center points
        region_center_x = x + w / 2
        region_center_y = y + h / 2
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        # Calculate normalized distance
        dx = (region_center_x - frame_center_x) / (frame_width / 2)
        dy = (region_center_y - frame_center_y) / (frame_height / 2)
        
        return min(1.0, np.sqrt(dx**2 + dy**2))
    
    def _determine_optimal_position(
        self,
        frame_analyses: List[List[ContentRegion]],
        video_info: Dict[str, Any]
    ) -> str:
        """
        Determine the optimal subtitle position based on content analysis.
        
        Args:
            frame_analyses: List of content regions detected in frames
            video_info: Video information dictionary
            
        Returns:
            Optimal position ('top' or 'bottom')
        """
        width = video_info.get("width", 1920)
        height = video_info.get("height", 1080)
        
        # Count important content in each region across frames
        top_importance = 0.0
        bottom_importance = 0.0
        
        # Normalize region coordinates
        top_region = (
            int(self.top_region[0] * width),
            int(self.top_region[1] * height),
            int(self.top_region[2] * width),
            int(self.top_region[3] * height)
        )
        
        bottom_region = (
            int(self.bottom_region[0] * width),
            int(self.bottom_region[1] * height),
            int(self.bottom_region[2] * width),
            int(self.bottom_region[3] * height)
        )
        
        # Process all frames
        for regions in frame_analyses:
            for region in regions:
                # Calculate region center
                region_center_x = region.x + region.width / 2
                region_center_y = region.y + region.height / 2
                
                # Calculate importance score based on region attributes
                importance_value = self._get_importance_value(region.importance)
                importance_score = importance_value * region.confidence
                
                # Check if the region overlaps with top or bottom areas
                # For face detection, we check if *any part* of the face is in the region
                # to ensure we don't place subtitles over parts of faces
                if region.content_type == "face":
                    # Check if the face overlaps with the top region
                    if (region.y < top_region[1] + top_region[3] and 
                        region.y + region.height > top_region[1]):
                        top_importance += importance_score
                    
                    # Check if the face overlaps with the bottom region
                    if (region.y + region.height > bottom_region[1] and 
                        region.y < bottom_region[1] + bottom_region[3]):
                        bottom_importance += importance_score
                else:
                    # For other content, just check the center point
                    if self._point_in_region(region_center_x, region_center_y, top_region):
                        top_importance += importance_score
                    
                    if self._point_in_region(region_center_x, region_center_y, bottom_region):
                        bottom_importance += importance_score
        
        # Normalize importance scores based on number of frames
        num_frames = len(frame_analyses)
        if num_frames > 0:
            top_importance /= num_frames
            bottom_importance /= num_frames
        
        # Apply preference order if the difference is small
        threshold = 0.5
        if abs(top_importance - bottom_importance) < threshold:
            # Use the preference order
            return self.position_preference[0]
        
        # Otherwise, use the position with less important content
        return "top" if top_importance < bottom_importance else "bottom"
    
    def _get_importance_value(self, importance: ContentImportance) -> float:
        """Convert importance enum to numeric value."""
        if importance == ContentImportance.CRITICAL:
            return 10.0
        elif importance == ContentImportance.HIGH:
            return 5.0
        elif importance == ContentImportance.MEDIUM:
            return 2.0
        else:  # LOW
            return 1.0
    
    def _point_in_region(
        self,
        x: float,
        y: float,
        region: Tuple[int, int, int, int]
    ) -> bool:
        """
        Check if a point is inside a region.
        
        Args:
            x, y: Point coordinates
            region: (x, y, width, height) of the region
            
        Returns:
            True if the point is inside the region
        """
        rx, ry, rw, rh = region
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about the video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        # Use FFprobe to get video information
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            # Run the FFprobe command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                stderr_str = stderr.decode('utf-8', errors='replace')
                self.logger.error(f"FFprobe failed: {stderr_str}")
                raise RuntimeError(f"Failed to get video info: {stderr_str}")
            
            # Parse JSON output
            ffprobe_output = json.loads(stdout)
            
            # Extract relevant video information
            info = {
                "width": 1920,   # Default if not found
                "height": 1080,  # Default if not found
                "duration": 0,   # Default if not found
                "fps": 30        # Default if not found
            }
            
            # Find video stream
            for stream in ffprobe_output.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["width"] = int(stream.get("width", 1920))
                    info["height"] = int(stream.get("height", 1080))
                    
                    # Parse FPS as fraction (e.g. "24/1")
                    fps_str = stream.get("r_frame_rate", "30/1")
                    if "/" in fps_str:
                        num, den = map(int, fps_str.split("/"))
                        info["fps"] = num / den if den else 30
                    else:
                        info["fps"] = float(fps_str)
                    
                    break
            
            # Get duration from format section
            format_info = ffprobe_output.get("format", {})
            if "duration" in format_info:
                info["duration"] = float(format_info["duration"])
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            # Return default values if something goes wrong
            return {
                "width": 1920,
                "height": 1080,
                "duration": 60,
                "fps": 30
            } 