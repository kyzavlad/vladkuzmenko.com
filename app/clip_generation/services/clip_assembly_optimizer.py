"""
Clip Assembly Optimizer Module

This module extends the basic ClipAssemblyOptimizer with advanced optimization
features like face-aware cropping for vertical videos.
"""

import os
import logging
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from app.clip_generation.services.clip_assembly import ClipAssemblyOptimizer, ClipAssemblyConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedClipAssemblyOptimizer(ClipAssemblyOptimizer):
    """
    Advanced clip assembly optimizer with face-aware cropping and other optimizations.
    
    This class extends the basic ClipAssemblyOptimizer with advanced features
    like face detection for optimal cropping in vertical videos.
    """
    
    def __init__(self, config: Optional[ClipAssemblyConfig] = None):
        """
        Initialize the advanced clip assembly optimizer.
        
        Args:
            config: Configuration for clip assembly and optimization
        """
        super().__init__(config)
        
        # Set up face detection
        self._setup_face_detection()
        
        logger.info("AdvancedClipAssemblyOptimizer initialized")

    def _setup_face_detection(self):
        """Set up face detection module if available."""
        try:
            import cv2
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self._has_face_detection = True
            logger.info("Face detection initialized")
        except ImportError:
            logger.warning("OpenCV not available, face-aware cropping will be limited")
            self._has_face_detection = False
    
    def _detect_faces_in_frame(self, frame_path: str) -> List[Dict[str, Any]]:
        """
        Detect faces in a video frame.
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            List of detected faces with position and size information
        """
        if not self._has_face_detection:
            logger.warning("Face detection not available")
            return []
        
        try:
            import cv2
            import numpy as np
            
            # Read the image
            img = cv2.imread(frame_path)
            if img is None:
                logger.warning(f"Failed to read image: {frame_path}")
                return []
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Convert to list of dictionaries with face information
            face_list = []
            for (x, y, w, h) in faces:
                face_info = {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "center_x": int(x + w/2),
                    "center_y": int(y + h/2),
                    "area": int(w * h)
                }
                face_list.append(face_info)
            
            # Sort by area (largest first)
            face_list.sort(key=lambda f: f["area"], reverse=True)
            
            logger.info(f"Detected {len(face_list)} faces in frame")
            return face_list
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def _extract_sample_frames(
        self, 
        video_path: str, 
        start_time: float, 
        end_time: float, 
        num_frames: int = 5
    ) -> List[str]:
        """
        Extract sample frames from a video segment for face detection.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            num_frames: Number of frames to extract
            
        Returns:
            List of paths to extracted frames
        """
        # Create temp directory for frames
        frames_dir = self.temp_dir / f"face_frames_{os.path.basename(video_path)}"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Calculate frame intervals
        duration = end_time - start_time
        if duration <= 0:
            logger.warning("Invalid duration for frame extraction")
            return []
        
        interval = duration / (num_frames + 1)
        frame_times = [start_time + interval * (i + 1) for i in range(num_frames)]
        
        # Extract frames
        frame_paths = []
        for i, timestamp in enumerate(frame_times):
            frame_path = frames_dir / f"frame_{i:03d}.jpg"
            
            try:
                # Use ffmpeg to extract frame
                cmd = [
                    self.config.ffmpeg_path,
                    "-y",
                    "-ss", str(timestamp),
                    "-i", video_path,
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(frame_path)
                ]
                
                subprocess.run(
                    cmd, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                if os.path.exists(frame_path):
                    frame_paths.append(str(frame_path))
                    
            except Exception as e:
                logger.error(f"Error extracting frame at {timestamp}: {str(e)}")
        
        logger.info(f"Extracted {len(frame_paths)} sample frames for face detection")
        return frame_paths
    
    def _calculate_optimal_crop_region(
        self,
        faces: List[Dict[str, Any]],
        original_width: int,
        original_height: int,
        target_width: int,
        target_height: int
    ) -> Dict[str, int]:
        """
        Calculate the optimal crop region based on detected faces.
        
        Args:
            faces: List of detected faces
            original_width: Original video width
            original_height: Original video height
            target_width: Target width after cropping
            target_height: Target height after cropping
            
        Returns:
            Dictionary with crop coordinates (x, y, width, height)
        """
        # If no faces detected, use center crop
        if not faces:
            crop_x = (original_width - target_width) // 2
            crop_y = 0  # Start from top 
            return {
                "x": crop_x,
                "y": crop_y,
                "width": target_width,
                "height": target_height
            }
        
        # Calculate weighted center of all faces
        # Weight by face area (larger faces have more weight)
        total_weight = sum(face["area"] for face in faces)
        weighted_center_x = sum(face["center_x"] * face["area"] for face in faces) / total_weight
        
        # Calculate crop region
        crop_x = int(weighted_center_x - target_width / 2)
        
        # Ensure crop region is within bounds
        crop_x = max(0, min(crop_x, original_width - target_width))
        crop_y = 0  # Start from top for vertical videos
        
        return {
            "x": crop_x,
            "y": crop_y,
            "width": target_width,
            "height": target_height
        }
    
    def generate_face_aware_vertical_clip(
        self,
        source_video: str,
        output_path: str,
        start_time: float,
        end_time: float,
        audio_normalize: bool = True
    ) -> str:
        """
        Generate a vertical clip with face-aware cropping.
        
        This method analyzes the video to detect faces and ensures they
        remain centered in the vertical crop.
        
        Args:
            source_video: Path to the source video
            output_path: Path for the output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            audio_normalize: Whether to normalize audio levels
            
        Returns:
            Path to the generated clip
        """
        # Validate source video
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
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
                logger.info("Video is already vertical, no need for face-aware cropping")
                return super().generate_vertical_clip(
                    source_video, 
                    output_path, 
                    start_time, 
                    end_time, 
                    audio_normalize
                )
            else:
                # Convert to vertical
                # For 16:9 source to 9:16 target, we need to crop
                target_height = original_height
                target_width = 9 * target_height / 16
                
                # If target_width > original_width, we need to scale instead
                if target_width > original_width:
                    target_width = original_width
                    target_height = 16 * target_width / 9
            
            # Extract sample frames for face detection
            sample_frames = self._extract_sample_frames(source_video, start_time, end_time)
            
            # Detect faces in all frames
            all_faces = []
            for frame_path in sample_frames:
                faces = self._detect_faces_in_frame(frame_path)
                all_faces.extend(faces)
            
            # Calculate optimal crop region
            crop_region = self._calculate_optimal_crop_region(
                all_faces,
                original_width,
                original_height,
                int(target_width),
                int(target_height)
            )
            
            logger.info(f"Calculated face-aware crop region: {crop_region}")
            
            # First extract the raw segment
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
            
            # Now apply face-aware cropping and processing
            vertical_cmd = [
                self.config.ffmpeg_path,
                "-y",  # Overwrite output file if it exists
                "-i", str(temp_clip)  # Input file
            ]
            
            # Set up video filter for face-aware cropping
            video_filter = (
                f"crop={crop_region['width']}:{crop_region['height']}:"
                f"{crop_region['x']}:{crop_region['y']},"
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
            
            logger.info(f"Successfully generated face-aware vertical clip: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFMPEG error: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to generate vertical clip: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating vertical clip: {str(e)}")
            raise
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_clip):
                try:
                    os.remove(temp_clip)
                except Exception:
                    pass
            
            # Clean up sample frames
            frames_dir = self.temp_dir / f"face_frames_{os.path.basename(source_video)}"
            if os.path.exists(frames_dir):
                try:
                    import shutil
                    shutil.rmtree(frames_dir)
                except Exception:
                    pass 