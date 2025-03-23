"""
Animation Framework

This module provides functionality for animating 3D face models,
including the First Order Motion Model with temporal consistency,
facial landmark tracking, micro-expression synthesis,
and person-specific gesture learning.
"""

import os
import logging
import numpy as np
import cv2
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import uuid
import tempfile
import time
from dataclasses import dataclass
import scipy.interpolate

logger = logging.getLogger(__name__)

@dataclass
class AnimationResult:
    """Results from the animation process."""
    animation_id: str
    output_path: str
    duration: float
    frame_count: int
    fps: float
    landmarks_sequence: Optional[List[Dict[str, List[float]]]] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class AnimationFramework:
    """
    Animation Framework for generating realistic facial animations and gestures.
    
    Features:
    - First Order Motion Model with temporal consistency
    - 68-point facial landmark tracking
    - Natural micro-expression synthesis
    - Gaze direction modeling and control
    - Head pose variation with natural limits
    - Emotion intensity adjustment
    - Person-specific gesture and mannerism learning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Animation Framework.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "outputs/animations")
        self.landmark_model_path = self.config.get("landmark_model_path", "models/landmark_detector.dat")
        self.motion_model_path = self.config.get("motion_model_path", "models/first_order_motion_model.pth")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Animation Framework initialized")
        self._load_models()
    
    def _load_models(self):
        """Load the required models for animation."""
        logger.info("Loading animation models...")
        
        # In a real implementation, this would load the actual models
        # For the sake of this example, we'll simulate the loading
        self.has_landmark_model = True
        self.has_motion_model = True
        
        logger.info("Animation models loaded successfully")
    
    async def animate_from_video(self, 
                          face_model_path: str, 
                          driving_video_path: str, 
                          options: Optional[Dict[str, Any]] = None) -> AnimationResult:
        """
        Animate a 3D face model using a driving video.
        
        Args:
            face_model_path: Path to the 3D face model
            driving_video_path: Path to the driving video
            options: Animation options
            
        Returns:
            AnimationResult: The result of the animation process
        """
        start_time = time.time()
        logger.info(f"Starting animation from video: {driving_video_path}")
        
        options = options or {}
        animation_id = str(uuid.uuid4())
        output_path = os.path.join(self.output_dir, f"{animation_id}.mp4")
        
        try:
            # Extract frames from the driving video
            driving_frames, fps = self._extract_frames(driving_video_path)
            frame_count = len(driving_frames)
            
            # Track landmarks in the driving video
            landmarks_sequence = self._track_landmarks(driving_frames)
            
            # Apply First Order Motion Model with temporal consistency
            animated_frames = self._apply_first_order_motion(face_model_path, driving_frames, landmarks_sequence, options)
            
            # Enhance animation with micro-expressions if enabled
            if options.get("enhance_micro_expressions", True):
                animated_frames = self._enhance_micro_expressions(animated_frames, landmarks_sequence)
            
            # Apply temporal consistency adjustments
            animated_frames = self._apply_temporal_consistency(animated_frames, options)
            
            # Save the animation
            self._save_animation(animated_frames, output_path, fps)
            
            processing_time = time.time() - start_time
            logger.info(f"Animation completed in {processing_time:.2f} seconds")
            
            return AnimationResult(
                animation_id=animation_id,
                output_path=output_path,
                duration=frame_count / fps,
                frame_count=frame_count,
                fps=fps,
                landmarks_sequence=landmarks_sequence,
                processing_time=processing_time,
                metadata={"source_video": driving_video_path}
            )
            
        except Exception as e:
            logger.error(f"Error in animation process: {str(e)}")
            raise

    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple containing a list of frames and the fps of the video
        """
        logger.info(f"Extracting frames from {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frames.append(frame)
                
            cap.release()
            logger.info(f"Extracted {len(frames)} frames at {fps} FPS")
            
            return frames, fps
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise

    def _track_landmarks(self, frames: List[np.ndarray]) -> List[Dict[str, List[float]]]:
        """
        Track 68-point facial landmarks across video frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of dictionaries containing landmarks for each frame
        """
        logger.info(f"Tracking landmarks across {len(frames)} frames")
        
        landmarks_sequence = []
        
        for i, frame in enumerate(frames):
            if i % 100 == 0:
                logger.info(f"Processing landmarks for frame {i}/{len(frames)}")
                
            # In a real implementation, this would use an actual landmark detector
            # For this example, we'll generate simulated landmarks
            landmarks = self._simulate_landmark_detection(frame)
            landmarks_sequence.append(landmarks)
            
        logger.info(f"Landmark tracking completed for {len(frames)} frames")
        return landmarks_sequence

    def _simulate_landmark_detection(self, frame: np.ndarray) -> Dict[str, List[float]]:
        """
        Simulate facial landmark detection.
        
        Args:
            frame: Video frame
            
        Returns:
            Dictionary of facial landmarks
        """
        # This is a simulation for the example
        # In a real implementation, an actual facial landmark detector would be used
        
        # Generate 68 landmark points with small random variations
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Basic face shape - simulated landmarks
        landmarks = {
            # Jaw line
            "jaw": [
                [center_x - 100 + np.random.normal(0, 2), center_y + 80 + np.random.normal(0, 1)]
                for i in range(17)
            ],
            # Right eyebrow
            "right_eyebrow": [
                [center_x - 80 + i * 10 + np.random.normal(0, 1), center_y - 50 + np.random.normal(0, 1)]
                for i in range(5)
            ],
            # Left eyebrow
            "left_eyebrow": [
                [center_x + 30 + i * 10 + np.random.normal(0, 1), center_y - 50 + np.random.normal(0, 1)]
                for i in range(5)
            ],
            # Nose bridge
            "nose_bridge": [
                [center_x + np.random.normal(0, 1), center_y - 30 + i * 10 + np.random.normal(0, 1)]
                for i in range(4)
            ],
            # Nose tip
            "nose_tip": [
                [center_x - 20 + i * 10 + np.random.normal(0, 1), center_y + np.random.normal(0, 1)]
                for i in range(5)
            ],
            # Right eye
            "right_eye": [
                [center_x - 60 + np.random.normal(0, 1), center_y - 30 + np.random.normal(0, 1)],
                [center_x - 50 + np.random.normal(0, 1), center_y - 35 + np.random.normal(0, 1)],
                [center_x - 40 + np.random.normal(0, 1), center_y - 30 + np.random.normal(0, 1)],
                [center_x - 50 + np.random.normal(0, 1), center_y - 25 + np.random.normal(0, 1)],
                [center_x - 60 + np.random.normal(0, 1), center_y - 30 + np.random.normal(0, 1)],
                [center_x - 50 + np.random.normal(0, 1), center_y - 30 + np.random.normal(0, 1)]
            ],
            # Left eye
            "left_eye": [
                [center_x + 40 + np.random.normal(0, 1), center_y - 30 + np.random.normal(0, 1)],
                [center_x + 50 + np.random.normal(0, 1), center_y - 35 + np.random.normal(0, 1)],
                [center_x + 60 + np.random.normal(0, 1), center_y - 30 + np.random.normal(0, 1)],
                [center_x + 50 + np.random.normal(0, 1), center_y - 25 + np.random.normal(0, 1)],
                [center_x + 40 + np.random.normal(0, 1), center_y - 30 + np.random.normal(0, 1)],
                [center_x + 50 + np.random.normal(0, 1), center_y - 30 + np.random.normal(0, 1)]
            ],
            # Outer lip
            "outer_lip": [
                [center_x - 30 + i * 10 + np.random.normal(0, 1), center_y + 40 + np.random.normal(0, 1)]
                for i in range(7)
            ],
            # Inner lip
            "inner_lip": [
                [center_x - 20 + i * 10 + np.random.normal(0, 1), center_y + 40 + np.random.normal(0, 1)]
                for i in range(5)
            ]
        }
        
        # Flatten to 1D for compatibility with standard landmark formats
        flattened = []
        for region in landmarks.values():
            for point in region:
                flattened.extend(point)
                
        return {"points": flattened, "regions": landmarks}

    def _apply_first_order_motion(self, 
                            face_model_path: str, 
                            driving_frames: List[np.ndarray],
                            landmarks_sequence: List[Dict[str, List[float]]],
                            options: Dict[str, Any]) -> List[np.ndarray]:
        """
        Apply First Order Motion Model with temporal consistency.
        
        Args:
            face_model_path: Path to the 3D face model
            driving_frames: List of driving video frames
            landmarks_sequence: List of landmarks for each frame
            options: Animation options
            
        Returns:
            List of animated frames
        """
        logger.info("Applying First Order Motion Model with temporal consistency")
        
        # Load the face model
        # In a real implementation, this would load the actual 3D model
        # For this example, we'll simulate the model
        
        animated_frames = []
        
        # Calculate motion fields between consecutive frames
        motion_fields = self._calculate_motion_fields(landmarks_sequence)
        
        # Apply motion fields to the face model
        for i, frame in enumerate(driving_frames):
            if i % 100 == 0:
                logger.info(f"Generating animated frame {i}/{len(driving_frames)}")
                
            # Simulate applying motion to the face model
            # In a real implementation, this would use the First Order Motion Model
            animated_frame = self._simulate_animation_frame(frame, motion_fields[min(i, len(motion_fields)-1)])
            animated_frames.append(animated_frame)
            
        logger.info(f"Generated {len(animated_frames)} animated frames")
        return animated_frames

    def _calculate_motion_fields(self, landmarks_sequence: List[Dict[str, List[float]]]) -> List[np.ndarray]:
        """
        Calculate motion fields between consecutive frames based on landmarks.
        
        Args:
            landmarks_sequence: List of landmarks for each frame
            
        Returns:
            List of motion fields
        """
        motion_fields = []
        
        for i in range(1, len(landmarks_sequence)):
            prev_landmarks = landmarks_sequence[i-1]
            curr_landmarks = landmarks_sequence[i]
            
            # Calculate the motion field between consecutive frames
            # This is a simplified simulation
            motion_field = np.zeros((100, 100, 2))  # Placeholder size
            
            # In a real implementation, this would use optical flow or dense motion prediction
            # based on the landmark differences
            
            motion_fields.append(motion_field)
            
        # Add a zero motion field for the first frame
        motion_fields.insert(0, np.zeros((100, 100, 2)))
        
        return motion_fields

    def _simulate_animation_frame(self, driving_frame: np.ndarray, motion_field: np.ndarray) -> np.ndarray:
        """
        Simulate generating an animated frame.
        
        Args:
            driving_frame: Driving video frame
            motion_field: Motion field to apply
            
        Returns:
            Animated frame
        """
        # This is a simulation for the example
        # In a real implementation, this would apply the motion field to the 3D model
        
        # For the simulation, we'll just add a small random variation to the driving frame
        animated_frame = driving_frame.copy()
        
        # Add a subtle variation to simulate animation
        noise = np.random.normal(0, 5, driving_frame.shape).astype(np.uint8)
        animated_frame = cv2.add(animated_frame, noise)
        
        return animated_frame

    def _apply_temporal_consistency(self, 
                              frames: List[np.ndarray], 
                              options: Dict[str, Any]) -> List[np.ndarray]:
        """
        Apply temporal consistency adjustments to ensure smooth animation.
        
        Args:
            frames: List of animated frames
            options: Animation options
            
        Returns:
            List of temporally consistent frames
        """
        logger.info("Applying temporal consistency adjustments")
        
        smoothness = options.get("temporal_smoothness", 0.8)
        window_size = options.get("smoothing_window", 5)
        
        # Apply temporal smoothing
        # In a real implementation, this would use more sophisticated methods
        # For this example, we'll use a simple moving average
        
        smoothed_frames = frames.copy()
        
        if len(frames) <= window_size:
            return smoothed_frames
            
        for i in range(window_size, len(frames) - window_size):
            # Simple weighted average of neighboring frames
            for w in range(-window_size, window_size + 1):
                weight = 1.0 - abs(w) / (window_size + 1)
                weight *= smoothness if w != 0 else (1.0 - smoothness)
                
                if w == 0:
                    # Current frame gets higher weight
                    smoothed_frames[i] = frames[i].copy()
                else:
                    # Add weighted contribution from neighboring frame
                    smoothed_frames[i] = cv2.addWeighted(
                        smoothed_frames[i], 1.0, 
                        frames[i + w], weight, 
                        0
                    )
                    
        logger.info("Temporal consistency applied")
        return smoothed_frames

    def _enhance_micro_expressions(self, 
                             frames: List[np.ndarray],
                             landmarks_sequence: List[Dict[str, List[float]]]) -> List[np.ndarray]:
        """
        Enhance the animation with natural micro-expressions.
        
        Args:
            frames: List of animated frames
            landmarks_sequence: List of landmarks for each frame
            
        Returns:
            List of frames enhanced with micro-expressions
        """
        logger.info("Enhancing animation with micro-expressions")
        
        # Skip if no frames or fewer than 3 frames (need at least start/middle/end)
        if not frames or len(frames) < 3:
            logger.warning("Not enough frames to add micro-expressions")
            return frames
            
        enhanced_frames = frames.copy()
        frame_count = len(frames)
        
        # Define common micro-expressions to add
        micro_expressions = {
            "eye_blink": {
                "probability": 0.2,  # probability of adding a blink
                "duration": [3, 6],  # duration range in frames
                "regions": ["left_eye", "right_eye"],
                "intensity": 0.8
            },
            "eyebrow_raise": {
                "probability": 0.15,
                "duration": [8, 15],
                "regions": ["left_eyebrow", "right_eyebrow"],
                "intensity": 0.4
            },
            "mouth_twitch": {
                "probability": 0.1,
                "duration": [4, 8],
                "regions": ["outer_lip", "inner_lip"],
                "intensity": 0.3
            },
            "nostril_flare": {
                "probability": 0.05,
                "duration": [5, 10],
                "regions": ["nose_tip"],
                "intensity": 0.2
            }
        }
        
        # Track added expressions to avoid overlaps
        added_expressions = []
        
        # Add random micro-expressions throughout the animation
        import random
        
        # Add eye blinks approximately every 2-5 seconds (assuming 30fps)
        blink_interval = random.randint(60, 150)
        last_blink = -blink_interval  # start with possibility of early blink
        
        for i in range(frame_count):
            # Randomly add eye blinks at natural intervals
            if i - last_blink >= blink_interval:
                # Only add blink if we have enough frames left
                if i + 8 < frame_count:
                    self._add_eye_blink(enhanced_frames, landmarks_sequence, i, duration=random.randint(4, 8))
                    last_blink = i
                    blink_interval = random.randint(60, 150)  # ~2-5 seconds at 30fps
            
            # Check if we should add other random micro-expressions
            for expr_name, expr_props in micro_expressions.items():
                # Skip eye blinks as they're handled separately
                if expr_name == "eye_blink":
                    continue
                    
                # Randomly decide to add this expression
                if random.random() < expr_props["probability"] / frame_count * 100:
                    # Check that we don't already have an expression at this point
                    can_add = True
                    for added in added_expressions:
                        if abs(added["start"] - i) < added["duration"]:
                            can_add = False
                            break
                    
                    if can_add and i + expr_props["duration"][1] < frame_count:
                        duration = random.randint(expr_props["duration"][0], expr_props["duration"][1])
                        self._add_micro_expression(
                            enhanced_frames, 
                            landmarks_sequence, 
                            i, 
                            expr_name, 
                            expr_props["regions"],
                            duration, 
                            expr_props["intensity"]
                        )
                        added_expressions.append({
                            "name": expr_name,
                            "start": i,
                            "duration": duration
                        })
        
        logger.info(f"Added natural micro-expressions to animation ({len(added_expressions)} expressions)")
        return enhanced_frames
    
    def _add_eye_blink(self, frames: List[np.ndarray], landmarks_sequence: List[Dict[str, List[float]]], 
                       start_frame: int, duration: int) -> None:
        """
        Add a natural eye blink to the animation.
        
        Args:
            frames: List of animation frames
            landmarks_sequence: List of landmarks for each frame
            start_frame: Starting frame for the blink
            duration: Duration of blink in frames
        """
        if duration < 3:
            duration = 3  # Minimum duration for a blink
            
        # Get the eye landmarks from the sequence
        try:
            landmarks = landmarks_sequence[start_frame]["regions"]
            left_eye = landmarks.get("left_eye", [])
            right_eye = landmarks.get("right_eye", [])
            
            if not left_eye or not right_eye:
                return  # Can't add blink without eye landmarks
                
            # Calculate eye center and size for left and right eyes
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            
            # Add blink effect to the frames
            for i in range(duration):
                # Calculate blink progress (0->1->0 for open->closed->open)
                if i < duration / 2:
                    progress = i / (duration / 2)  # 0 to 1
                else:
                    progress = 2 - (i / (duration / 2))  # 1 to 0
                    
                # Skip if we're out of frame range
                if start_frame + i >= len(frames):
                    break
                    
                # Get the frame to modify
                frame = frames[start_frame + i]
                
                # Apply blink effect by darkening and slightly closing the eye region
                self._apply_eye_closure(frame, left_eye_center, progress)
                self._apply_eye_closure(frame, right_eye_center, progress)
                
        except (IndexError, KeyError) as e:
            logger.warning(f"Error adding eye blink: {str(e)}")
    
    def _apply_eye_closure(self, frame: np.ndarray, eye_center, progress: float) -> None:
        """
        Apply eye closure effect to a specific region of the frame.
        """
        # Convert center to integer coordinates
        center_x, center_y = int(eye_center[0]), int(eye_center[1])
        
        # Define eye region size
        eye_width = 30
        eye_height = 20
        
        # Calculate region boundaries with bounds checking
        x1 = max(0, center_x - eye_width // 2)
        x2 = min(frame.shape[1] - 1, center_x + eye_width // 2)
        
        # The vertical boundaries change with progress (closing the eye)
        eye_open_height = eye_height
        eye_closed_height = max(2, int(eye_height * (1 - progress)))
        
        y1 = max(0, center_y - eye_closed_height // 2)
        y2 = min(frame.shape[0] - 1, center_y + eye_closed_height // 2)
        
        # Skip if the region is invalid
        if x1 >= x2 or y1 >= y2:
            return
            
        # Apply darkening effect proportional to closure
        if progress > 0.1:  # Only darken when eye is significantly closed
            frame[y1:y2, x1:x2] = cv2.addWeighted(
                frame[y1:y2, x1:x2], 
                1 - progress * 0.5,  # Darkening factor
                np.zeros_like(frame[y1:y2, x1:x2]), 
                0, 
                -5 * progress  # Additional darkness
            )
    
    def _add_micro_expression(self, frames: List[np.ndarray], landmarks_sequence: List[Dict[str, List[float]]], 
                            start_frame: int, expression_type: str, regions: List[str],
                            duration: int, intensity: float) -> None:
        """
        Add a micro-expression to the animation.
        
        Args:
            frames: List of animation frames
            landmarks_sequence: List of landmarks for each frame
            start_frame: Starting frame for the expression
            expression_type: Type of micro-expression to add
            regions: Facial regions affected by the expression
            duration: Duration of expression in frames
            intensity: Expression intensity (0.0-1.0)
        """
        try:
            # Get landmarks for the affected regions
            landmarks = landmarks_sequence[start_frame]["regions"]
            
            # Process each frame of the micro-expression
            for i in range(duration):
                # Calculate expression progress (0->1->0 for onset->peak->offset)
                if i < duration / 3:  # onset
                    progress = i / (duration / 3) * intensity
                elif i < 2 * duration / 3:  # hold peak
                    progress = intensity
                else:  # offset
                    progress = intensity * (1 - (i - 2 * duration / 3) / (duration / 3))
                    
                # Skip if we're out of frame range
                if start_frame + i >= len(frames):
                    break
                    
                # Get the frame to modify
                frame = frames[start_frame + i]
                
                # Apply the appropriate micro-expression effect
                if expression_type == "eyebrow_raise":
                    self._apply_eyebrow_raise(frame, landmarks, progress)
                elif expression_type == "mouth_twitch":
                    self._apply_mouth_twitch(frame, landmarks, progress)
                elif expression_type == "nostril_flare":
                    self._apply_nostril_flare(frame, landmarks, progress)
                
        except (IndexError, KeyError) as e:
            logger.warning(f"Error adding micro-expression: {str(e)}")
    
    def _apply_eyebrow_raise(self, frame: np.ndarray, landmarks: Dict, progress: float) -> None:
        """Apply eyebrow raise effect to the frame."""
        try:
            left_eyebrow = landmarks.get("left_eyebrow", [])
            right_eyebrow = landmarks.get("right_eyebrow", [])
            
            if not left_eyebrow or not right_eyebrow:
                return
                
            # Define the regions to modify for eyebrow raising
            for eyebrow in [left_eyebrow, right_eyebrow]:
                eyebrow_center = np.mean(eyebrow, axis=0)
                center_x, center_y = int(eyebrow_center[0]), int(eyebrow_center[1])
                
                # Size of the region to modify
                width = 40
                height = 30
                
                # Calculate the shift amount based on progress
                shift = int(progress * 4)  # Maximum 4 pixels shift upward
                
                # Define regions with bounds checking
                x1 = max(0, center_x - width // 2)
                x2 = min(frame.shape[1] - 1, center_x + width // 2)
                y1 = max(0, center_y - height // 2)
                y2 = min(frame.shape[0] - 1, center_y + height // 2)
                
                # Target region (shifted upward)
                target_y1 = max(0, y1 - shift)
                target_y2 = max(0, y2 - shift)
                
                # Skip if regions are invalid
                if x1 >= x2 or y1 >= y2 or target_y1 >= target_y2:
                    continue
                
                # Simple shift approach - copy the eyebrow region to a slightly higher position
                # and blend with the original frame
                if shift > 0 and y1 + shift < frame.shape[0]:
                    # Calculate the height of the regions
                    src_height = y2 - y1
                    target_height = target_y2 - target_y1
                    
                    # Use the minimum height to avoid index errors
                    height_to_use = min(src_height, target_height)
                    
                    if height_to_use > 0:
                        # Create a mask for smooth blending
                        mask = np.zeros((height_to_use, x2 - x1, 3), dtype=np.float32)
                        mask[:] = progress
                        
                        # Blend the shifted eyebrow with the original frame
                        frame[target_y1:target_y1+height_to_use, x1:x2] = cv2.addWeighted(
                            frame[target_y1:target_y1+height_to_use, x1:x2],
                            1 - progress,
                            frame[y1:y1+height_to_use, x1:x2],
                            progress,
                            0
                        )
                        
        except Exception as e:
            logger.warning(f"Error applying eyebrow raise: {str(e)}")
    
    def _apply_mouth_twitch(self, frame: np.ndarray, landmarks: Dict, progress: float) -> None:
        """Apply subtle mouth twitch effect to the frame."""
        try:
            outer_lip = landmarks.get("outer_lip", [])
            inner_lip = landmarks.get("inner_lip", [])
            
            if not outer_lip:
                return
                
            # Find mouth corners
            mouth_points = np.array(outer_lip)
            left_corner_idx = np.argmin(mouth_points[:, 0])
            right_corner_idx = np.argmax(mouth_points[:, 0])
            
            left_corner = outer_lip[left_corner_idx]
            right_corner = outer_lip[right_corner_idx]
            
            # Randomly choose which corner to twitch (or both)
            import random
            twitch_left = random.random() < 0.7
            twitch_right = random.random() < 0.3
            
            # Size of the region to modify
            width = 20
            height = 20
            
            # Apply twitch to selected corners
            if twitch_left:
                center_x, center_y = int(left_corner[0]), int(left_corner[1])
                # Calculate the shift amount based on progress (horizontal and vertical)
                shift_x = int(progress * 2)  # Max 2 pixels horizontal
                shift_y = int(progress * 1)  # Max 1 pixel vertical
                
                # Define regions with bounds checking
                x1 = max(0, center_x - width // 2)
                x2 = min(frame.shape[1] - 1, center_x + width // 2)
                y1 = max(0, center_y - height // 2)
                y2 = min(frame.shape[0] - 1, center_y + height // 2)
                
                # Apply subtle warp to the corner region
                if x1 < x2 and y1 < y2:
                    # Simple brightness adjustment to simulate muscle movement
                    frame[y1:y2, x1:x2] = cv2.addWeighted(
                        frame[y1:y2, x1:x2],
                        1,
                        np.zeros_like(frame[y1:y2, x1:x2]),
                        0,
                        progress * 3  # Slight brightening
                    )
            
            if twitch_right:
                center_x, center_y = int(right_corner[0]), int(right_corner[1])
                # Calculate the shift amount based on progress (horizontal and vertical)
                shift_x = int(progress * 2)  # Max 2 pixels horizontal
                shift_y = int(progress * 1)  # Max 1 pixel vertical
                
                # Define regions with bounds checking
                x1 = max(0, center_x - width // 2)
                x2 = min(frame.shape[1] - 1, center_x + width // 2)
                y1 = max(0, center_y - height // 2)
                y2 = min(frame.shape[0] - 1, center_y + height // 2)
                
                # Apply subtle warp to the corner region
                if x1 < x2 and y1 < y2:
                    # Simple brightness adjustment to simulate muscle movement
                    frame[y1:y2, x1:x2] = cv2.addWeighted(
                        frame[y1:y2, x1:x2],
                        1,
                        np.zeros_like(frame[y1:y2, x1:x2]),
                        0,
                        progress * 3  # Slight brightening
                    )
                    
        except Exception as e:
            logger.warning(f"Error applying mouth twitch: {str(e)}")
    
    def _apply_nostril_flare(self, frame: np.ndarray, landmarks: Dict, progress: float) -> None:
        """Apply subtle nostril flare effect to the frame."""
        try:
            nose_tip = landmarks.get("nose_tip", [])
            
            if not nose_tip:
                return
                
            # Get nostril position (approximated from nose tip)
            nose_center = np.mean(nose_tip, axis=0)
            center_x, center_y = int(nose_center[0]), int(nose_center[1])
            
            # Size of nostril regions
            width = 30
            height = 15
            
            # Calculate left and right nostril centers
            left_center_x = center_x - 12
            right_center_x = center_x + 12
            nostril_y = center_y + 5
            
            # Apply subtle expansion effect to each nostril
            for nostril_x in [left_center_x, right_center_x]:
                # Define regions with bounds checking
                x1 = max(0, int(nostril_x - width // 2))
                x2 = min(frame.shape[1] - 1, int(nostril_x + width // 2))
                y1 = max(0, int(nostril_y - height // 2))
                y2 = min(frame.shape[0] - 1, int(nostril_y + height // 2))
                
                # Skip if regions are invalid
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Apply subtle darkening to simulate nostril expansion
                frame[y1:y2, x1:x2] = cv2.addWeighted(
                    frame[y1:y2, x1:x2],
                    1 - progress * 0.3,  # Subtle darkening
                    np.zeros_like(frame[y1:y2, x1:x2]),
                    0,
                    -progress * 2  # Slight shadow effect
                )
                
        except Exception as e:
            logger.warning(f"Error applying nostril flare: {str(e)}")

    def _save_animation(self, frames: List[np.ndarray], output_path: str, fps: float) -> None:
        """
        Save the animated frames as a video.
        
        Args:
            frames: List of animated frames
            output_path: Path to save the output video
            fps: Frames per second
        """
        logger.info(f"Saving animation to {output_path}")
        
        if not frames:
            raise ValueError("No frames to save")
            
        height, width = frames[0].shape[:2]
        
        # Set up the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames to the video
        for frame in frames:
            out.write(frame)
            
        # Release the video writer
        out.release()
        
        logger.info(f"Animation saved successfully to {output_path}") 