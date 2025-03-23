import logging
import os
import numpy as np
import cv2
import tempfile
import asyncio
import subprocess
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


class SubtitlePositioningService:
    """
    Analyzes video frames to determine optimal subtitle positioning.
    
    Features:
    - Face and object detection to avoid covering important content
    - Scene composition analysis for optimal subtitle placement
    - Content importance scoring based on size, position, and detection confidence
    - Dynamic positioning throughout video based on scene changes
    - Area masking to identify safe zones for subtitle placement
    """
    
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the subtitle positioning service.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable
            ffprobe_path: Path to FFprobe executable
            model_path: Path to pre-trained models for computer vision tasks
            config: Configuration options
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.model_path = model_path
        self.config = config or {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature detection models
        self._initialize_models()
        
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
    
    def _initialize_models(self):
        """Initialize computer vision models for content detection."""
        self.enable_face_detection = self.config.get("enable_face_detection", True)
        self.enable_object_detection = self.config.get("enable_object_detection", True)
        self.enable_text_detection = self.config.get("enable_text_detection", True)
        self.enable_motion_analysis = self.config.get("enable_motion_analysis", True)
        
        # Load face detection model
        if self.enable_face_detection:
            try:
                # Try to use OpenCV's built-in face detector
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    self.logger.warning("Built-in face detection model not found, disabling face detection")
                    self.enable_face_detection = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize face detection: {str(e)}")
                self.enable_face_detection = False
        
        # Initialize object detection if enabled
        if self.enable_object_detection:
            try:
                # Try to use a simple version with HOG + SVM descriptor if available
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception as e:
                self.logger.warning(f"Failed to initialize object detection: {str(e)}")
                self.enable_object_detection = False
        
        # Initialize simple edge detection for general scene complexity
        self.enable_scene_complexity_analysis = True
        
        self.logger.info(f"Model initialization complete. Face detection: {self.enable_face_detection}, "
                         f"Object detection: {self.enable_object_detection}, "
                         f"Text detection: {self.enable_text_detection}")
    
    def _initialize_parameters(self):
        """Initialize analysis parameters and thresholds."""
        # Sampling rate for frame analysis (analyze every N frames)
        self.frame_sample_rate = self.config.get("frame_sample_rate", 24)  # Default: analyze 1 frame per second for 24fps video
        
        # Number of frames to analyze per scene
        self.frames_per_scene = self.config.get("frames_per_scene", 5)
        
        # Importance thresholds
        self.face_importance = ContentImportance.CRITICAL
        self.text_importance = ContentImportance.HIGH
        self.central_object_importance = ContentImportance.HIGH
        self.peripheral_object_importance = ContentImportance.MEDIUM
        
        # Region of interest (ROI) parameters
        self.center_weight = self.config.get("center_weight", 1.5)  # Objects in center are more important
        self.size_weight = self.config.get("size_weight", 0.8)  # Larger objects are more important
        
        # Frame regions - define typical safe areas for subtitle placement
        # (normalized coordinates: 0-1)
        self.top_region = (0.0, 0.0, 1.0, 0.2)      # x, y, width, height
        self.bottom_region = (0.0, 0.8, 1.0, 0.2)
        self.center_region = (0.1, 0.3, 0.8, 0.4)
        
        # Default preference order for subtitle placement (bottom usually preferred)
        self.position_preference = self.config.get("position_preference", ["bottom", "top", "center"])
    
    async def analyze_video_for_positioning(
        self,
        video_path: str,
        transcript: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze video content to determine optimal subtitle positioning throughout the video.
        
        Args:
            video_path: Path to video file
            transcript: Transcript with timing information
            
        Returns:
            Updated transcript with position information for each segment
        """
        self.logger.info(f"Starting video analysis for subtitle positioning: {video_path}")
        
        # Get video info (duration, dimensions, etc.)
        video_info = await self._get_video_info(video_path)
        
        # Create a deep copy of the transcript to avoid modifying the original
        import copy
        processed_transcript = copy.deepcopy(transcript)
        
        try:
            # Extract key frames from the video
            key_frames = await self._extract_key_frames(video_path, video_info)
            self.logger.info(f"Extracted {len(key_frames)} key frames for analysis")
            
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
                if i % 10 == 0:
                    self.logger.debug(f"Processed positioning for {i}/{len(segments)} segments")
            
            self.logger.info(f"Completed subtitle positioning analysis for {len(segments)} segments")
            return processed_transcript
            
        except Exception as e:
            self.logger.error(f"Error analyzing video for subtitle positioning: {str(e)}")
            raise
        finally:
            # Clean up extracted frames
            for frame_path, _ in key_frames:
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up frame file {frame_path}: {str(e)}")
    
    async def _extract_key_frames(
        self,
        video_path: str,
        video_info: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Extract key frames from the video for analysis.
        
        Args:
            video_path: Path to video file
            video_info: Video information dictionary
            
        Returns:
            List of tuples containing (frame_path, timestamp)
        """
        duration = video_info.get("duration", 0)
        fps = video_info.get("fps", 30)
        
        # Calculate the frame interval based on the sample rate
        interval = max(1, round(fps / self.frame_sample_rate))
        
        # Determine how many frames to extract
        num_frames = min(200, max(20, int(duration / (interval / fps))))
        
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
            "-q:v", "3",
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
            raise
    
    def _analyze_frame(
        self,
        frame_path: str,
        video_info: Dict[str, Any]
    ) -> List[ContentRegion]:
        """
        Analyze a single frame to identify important content regions.
        
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
                    
                    # Faces are usually the most important elements
                    importance = self.face_importance
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
        
        # Object detection (if enabled)
        if self.enable_object_detection:
            try:
                # People detection
                people, weights = self.hog.detectMultiScale(
                    frame,
                    winStride=(8, 8),
                    padding=(4, 4),
                    scale=1.05
                )
                
                for i, (x, y, w, h) in enumerate(people):
                    # Get confidence from weights if available
                    confidence = float(weights[i]) if i < len(weights) else 0.6
                    
                    # Calculate importance based on size and position
                    size_ratio = (w * h) / (width * height)
                    center_dist = self._calculate_center_distance(x, y, w, h, width, height)
                    
                    # Determine importance based on position and size
                    if center_dist < 0.3 and size_ratio > 0.1:
                        importance = ContentImportance.HIGH
                    elif center_dist < 0.5:
                        importance = ContentImportance.MEDIUM
                    else:
                        importance = ContentImportance.LOW
                    
                    regions.append(ContentRegion(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        importance=importance,
                        confidence=min(1.0, max(0.5, confidence)),
                        content_type="person"
                    ))
            except Exception as e:
                self.logger.warning(f"Object detection error: {str(e)}")
        
        # Scene complexity analysis for general content importance
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Divide the frame into regions (3x3 grid) and calculate complexity for each
            region_height = height // 3
            region_width = width // 3
            
            for row in range(3):
                for col in range(3):
                    # Extract region
                    y_start = row * region_height
                    y_end = (row + 1) * region_height
                    x_start = col * region_width
                    x_end = (col + 1) * region_width
                    
                    region_edges = edges[y_start:y_end, x_start:x_end]
                    
                    # Calculate complexity (percentage of edge pixels)
                    complexity = np.count_nonzero(region_edges) / (region_height * region_width)
                    
                    # Only add complex regions that might contain important content
                    if complexity > 0.1:  # 10% threshold for edges
                        importance = ContentImportance.LOW
                        if complexity > 0.3:
                            importance = ContentImportance.MEDIUM
                        
                        # Central regions with high complexity are more important
                        if row == 1 and col == 1 and complexity > 0.2:
                            importance = ContentImportance.HIGH
                        
                        regions.append(ContentRegion(
                            x=x_start,
                            y=y_start,
                            width=region_width,
                            height=region_height,
                            importance=importance,
                            confidence=min(0.9, complexity * 2),  # Scale complexity to confidence
                            content_type="complex_region"
                        ))
        except Exception as e:
            self.logger.warning(f"Scene complexity analysis error: {str(e)}")
        
        # Text detection is complex, using a simple edge-based heuristic instead
        if self.enable_text_detection:
            try:
                # Use simple heuristics to detect potential text regions
                # (horizontal edges with specific patterns can indicate text lines)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply morphological operations to enhance text regions
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
                
                # Threshold the image
                _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # Apply closing to connect nearby text
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by aspect ratio and size to find text-like regions
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Text typically has a specific aspect ratio and size
                    aspect_ratio = w / h
                    size_ratio = (w * h) / (width * height)
                    
                    if 2 < aspect_ratio < 15 and 0.001 < size_ratio < 0.05:
                        # Likely text
                        regions.append(ContentRegion(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            importance=ContentImportance.MEDIUM,
                            confidence=0.7,
                            content_type="text"
                        ))
            except Exception as e:
                self.logger.warning(f"Text detection error: {str(e)}")
        
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
            Optimal position ('top', 'bottom', or 'center')
        """
        width = video_info.get("width", 1920)
        height = video_info.get("height", 1080)
        
        # Count important content in each region across frames
        top_importance = 0.0
        bottom_importance = 0.0
        center_importance = 0.0
        
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
        
        center_region = (
            int(self.center_region[0] * width),
            int(self.center_region[1] * height),
            int(self.center_region[2] * width),
            int(self.center_region[3] * height)
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
                
                # Check if the region overlaps with our predefined regions
                if self._point_in_region(region_center_x, region_center_y, top_region):
                    top_importance += importance_score
                
                if self._point_in_region(region_center_x, region_center_y, bottom_region):
                    bottom_importance += importance_score
                
                if self._point_in_region(region_center_x, region_center_y, center_region):
                    center_importance += importance_score
        
        # Normalize importance scores based on number of frames
        num_frames = len(frame_analyses)
        if num_frames > 0:
            top_importance /= num_frames
            bottom_importance /= num_frames
            center_importance /= num_frames
        
        # Apply preference order if all regions have similar importance
        # (using a threshold to determine "similar")
        threshold = 0.5
        min_importance = min(top_importance, bottom_importance, center_importance)
        max_importance = max(top_importance, bottom_importance, center_importance)
        
        if max_importance - min_importance < threshold:
            # Regions have similar importance, use preference order
            for position in self.position_preference:
                if position == "top":
                    return "top"
                elif position == "bottom":
                    return "bottom"
                elif position == "center":
                    return "center"
        
        # Otherwise, choose the position with the least important content
        if top_importance <= bottom_importance and top_importance <= center_importance:
            return "top"
        elif bottom_importance <= top_importance and bottom_importance <= center_importance:
            return "bottom"
        else:
            return "center"
    
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
        Get video information using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video information
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration:stream=width,height,r_frame_rate",
            "-of", "json",
            "-select_streams", "v:0",
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
            import json
            info = json.loads(stdout.decode('utf-8'))
            
            # Extract relevant information
            duration = float(info.get("format", {}).get("duration", 0))
            
            stream_info = info.get("streams", [{}])[0]
            width = int(stream_info.get("width", 1920))
            height = int(stream_info.get("height", 1080))
            
            # Parse frame rate (could be in format "num/den")
            fps = 30.0  # Default
            r_frame_rate = stream_info.get("r_frame_rate", "30/1")
            if "/" in r_frame_rate:
                num, den = map(int, r_frame_rate.split("/"))
                if den != 0:
                    fps = num / den
            
            return {
                "duration": duration,
                "width": width,
                "height": height,
                "fps": fps
            }
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            # Return default values if we can't get actual info
            return {
                "duration": 0,
                "width": 1920,
                "height": 1080,
                "fps": 30
            } 