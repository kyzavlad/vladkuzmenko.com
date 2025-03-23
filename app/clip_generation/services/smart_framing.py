"""
Smart Framing Module

This module provides intelligent video framing capabilities based on face tracking,
implementing rule of thirds composition, smooth camera movements, and multi-face framing.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from app.clip_generation.services.face_tracking import FaceBox
from app.clip_generation.services.face_tracking_manager import TrackedFace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FramingRect:
    """Represents a framing rectangle for video cropping."""
    x: float  # Left coordinate
    y: float  # Top coordinate
    width: float  # Width
    height: float  # Height
    
    @property
    def right(self) -> float:
        """Get right coordinate."""
        return self.x + self.width
    
    @property
    def bottom(self) -> float:
        """Get bottom coordinate."""
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        """Get horizontal center."""
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Get vertical center."""
        return self.y + self.height / 2
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple (x, y, width, height) with integer values."""
        return (int(self.x), int(self.y), int(self.width), int(self.height))


class SmartFraming:
    """
    Provides intelligent video framing based on face tracking.
    
    Features:
    - Rule of thirds composition
    - Speaker focus
    - Smooth camera movements
    - Multi-face composition
    - Context preservation
    """
    
    def __init__(
        self,
        target_width: int = 1280,
        target_height: int = 720,
        smoothing_factor: float = 0.8,
        padding_ratio: float = 0.1,
        rule_of_thirds: bool = True,
        preserve_context: bool = True,
        min_movement_threshold: float = 0.05,
        face_zoom_factor: float = 1.5
    ):
        """
        Initialize smart framing.
        
        Args:
            target_width: Target output width
            target_height: Target output height
            smoothing_factor: Temporal smoothing factor (0-1)
            padding_ratio: Extra padding around faces
            rule_of_thirds: Apply rule of thirds composition
            preserve_context: Preserve scene context around faces
            min_movement_threshold: Minimum threshold for camera movement
            face_zoom_factor: Zoom factor for face framing
        """
        self.target_width = target_width
        self.target_height = target_height
        self.target_aspect = target_width / target_height
        self.smoothing_factor = smoothing_factor
        self.padding_ratio = padding_ratio
        self.rule_of_thirds = rule_of_thirds
        self.preserve_context = preserve_context
        self.min_movement_threshold = min_movement_threshold
        self.face_zoom_factor = face_zoom_factor
        
        # State variables
        self.current_framing: Optional[FramingRect] = None
        self.target_framing: Optional[FramingRect] = None
        self.frame_counter = 0
        
        logger.info(f"Initialized SmartFraming with target size {target_width}x{target_height}")
    
    def frame_image(
        self,
        image: np.ndarray,
        tracked_faces: Dict[int, TrackedFace],
        speaker_id: Optional[int] = None
    ) -> Tuple[np.ndarray, FramingRect]:
        """
        Apply smart framing to an image based on tracked faces.
        
        Args:
            image: Input image to frame
            tracked_faces: Dictionary of tracked faces by ID
            speaker_id: ID of the current speaker face (optional)
            
        Returns:
            Tuple of (framed image, framing rectangle)
        """
        self.frame_counter += 1
        height, width = image.shape[:2]
        
        # Calculate optimal framing
        target_rect = self._calculate_optimal_framing(
            tracked_faces,
            speaker_id,
            frame_width=width,
            frame_height=height
        )
        
        # Apply temporal smoothing
        smoothed_rect = self._smooth_framing(target_rect)
        
        # Perform cropping and resizing
        framed_image = self._apply_framing(image, smoothed_rect)
        
        return framed_image, smoothed_rect
    
    def reset(self) -> None:
        """Reset framing state."""
        self.current_framing = None
        self.target_framing = None
        self.frame_counter = 0
        
        logger.info("Reset SmartFraming state")
    
    def _calculate_optimal_framing(
        self,
        tracked_faces: Dict[int, TrackedFace],
        speaker_id: Optional[int] = None,
        frame_width: int = 1920,
        frame_height: int = 1080
    ) -> FramingRect:
        """
        Calculate the optimal framing rectangle based on tracked faces.
        
        Args:
            tracked_faces: Dictionary of tracked faces by ID
            speaker_id: ID of the current speaker (if known)
            frame_width: Width of the input frame
            frame_height: Height of the input frame
            
        Returns:
            Optimal framing rectangle
        """
        # Default: full frame
        full_frame = FramingRect(0, 0, frame_width, frame_height)
        
        # If no faces, return full frame
        if not tracked_faces:
            return self._adjust_aspect_ratio(full_frame)
        
        # Focus on speaker if available
        if speaker_id is not None and speaker_id in tracked_faces:
            speaker_face = tracked_faces[speaker_id]
            return self._calculate_speaker_framing(
                speaker_face,
                tracked_faces,
                frame_width,
                frame_height
            )
        
        # If no speaker, handle multiple faces
        if len(tracked_faces) == 1:
            # Single face - center on it
            face = next(iter(tracked_faces.values()))
            return self._calculate_single_face_framing(face, frame_width, frame_height)
        else:
            # Multiple faces - frame them all
            return self._calculate_multi_face_framing(tracked_faces, frame_width, frame_height)
    
    def _calculate_speaker_framing(
        self,
        speaker_face: TrackedFace,
        all_faces: Dict[int, TrackedFace],
        frame_width: int,
        frame_height: int
    ) -> FramingRect:
        """
        Calculate framing focused on the speaker, while considering other faces.
        
        Args:
            speaker_face: The speaker's tracked face
            all_faces: All tracked faces
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Framing rectangle
        """
        face_box = speaker_face.box
        
        # Apply rule of thirds - position face's center at 1/3 of the frame
        target_x = frame_width / 3 if self.rule_of_thirds else frame_width / 2
        target_y = frame_height / 3 if self.rule_of_thirds else frame_height / 2
        
        # Calculate face size scale factor based on our zoom preferences
        face_width = face_box.width * self.face_zoom_factor
        face_height = face_box.height * self.face_zoom_factor
        
        # Calculate padding
        padding_w = face_width * self.padding_ratio
        padding_h = face_height * self.padding_ratio
        
        # Calculate frame width and height
        frame_w = face_width + 2 * padding_w
        frame_h = face_height + 2 * padding_h
        
        # Adjust for aspect ratio
        if frame_w / frame_h > self.target_aspect:
            # Too wide, adjust height
            frame_h = frame_w / self.target_aspect
        else:
            # Too tall, adjust width
            frame_w = frame_h * self.target_aspect
        
        # Calculate frame position
        cx, cy = face_box.center
        
        # Apply rule of thirds offset
        if self.rule_of_thirds:
            # Position face at 1/3 of the frame horizontally
            offset_x = frame_w / 3
            # If face is on the right side of the frame, position at 2/3
            if cx > frame_width / 2:
                offset_x = frame_w * 2 / 3
                
            offset_y = frame_h / 3
        else:
            # Center the face
            offset_x = frame_w / 2
            offset_y = frame_h / 2
        
        frame_x = cx - offset_x
        frame_y = cy - offset_y
        
        # Create framing rect
        rect = FramingRect(frame_x, frame_y, frame_w, frame_h)
        
        # Ensure we're not cropping too tight
        rect = self._ensure_context_preservation(rect, all_faces, frame_width, frame_height)
        
        # Ensure framing is within the image bounds
        rect = self._clip_to_frame(rect, frame_width, frame_height)
        
        return rect
    
    def _calculate_single_face_framing(
        self,
        face: TrackedFace,
        frame_width: int,
        frame_height: int
    ) -> FramingRect:
        """
        Calculate framing for a single face.
        
        Args:
            face: The face to frame
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Framing rectangle
        """
        face_box = face.box
        
        # Calculate face size with zoom factor
        face_width = face_box.width * self.face_zoom_factor
        face_height = face_box.height * self.face_zoom_factor
        
        # Calculate padding
        padding_w = face_width * self.padding_ratio
        padding_h = face_height * self.padding_ratio
        
        # Calculate frame width and height
        frame_w = face_width + 2 * padding_w
        frame_h = face_height + 2 * padding_h
        
        # Adjust for aspect ratio
        if frame_w / frame_h > self.target_aspect:
            # Too wide, adjust height
            frame_h = frame_w / self.target_aspect
        else:
            # Too tall, adjust width
            frame_w = frame_h * self.target_aspect
        
        # Calculate frame position
        cx, cy = face_box.center
        
        # Apply rule of thirds
        if self.rule_of_thirds:
            # Position face at 1/3 of the frame horizontally
            offset_x = frame_w / 3
            # If face is on the right side of the frame, position at 2/3
            if cx > frame_width / 2:
                offset_x = frame_w * 2 / 3
                
            offset_y = frame_h / 3
        else:
            # Center the face
            offset_x = frame_w / 2
            offset_y = frame_h / 2
        
        frame_x = cx - offset_x
        frame_y = cy - offset_y
        
        # Create framing rect
        rect = FramingRect(frame_x, frame_y, frame_w, frame_h)
        
        # Ensure framing is within the image bounds
        rect = self._clip_to_frame(rect, frame_width, frame_height)
        
        return rect
    
    def _calculate_multi_face_framing(
        self,
        faces: Dict[int, TrackedFace],
        frame_width: int,
        frame_height: int
    ) -> FramingRect:
        """
        Calculate framing that includes multiple faces.
        
        Args:
            faces: Dictionary of tracked faces
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Framing rectangle
        """
        if not faces:
            return FramingRect(0, 0, frame_width, frame_height)
        
        # Find the bounding box that contains all faces
        min_x = frame_width
        min_y = frame_height
        max_x = 0
        max_y = 0
        
        for face_id, face in faces.items():
            # Use track history for stable faces to prevent jitter
            if face.track_length > 5:
                box = face.box
                min_x = min(min_x, box.x1)
                min_y = min(min_y, box.y1)
                max_x = max(max_x, box.x2)
                max_y = max(max_y, box.y2)
        
        # Calculate padding
        width = max_x - min_x
        height = max_y - min_y
        
        padding_w = width * self.padding_ratio
        padding_h = height * self.padding_ratio
        
        # Calculate frame width and height
        frame_w = width + 2 * padding_w
        frame_h = height + 2 * padding_h
        
        # Adjust for aspect ratio
        if frame_w / frame_h > self.target_aspect:
            # Too wide, adjust height
            frame_h = frame_w / self.target_aspect
        else:
            # Too tall, adjust width
            frame_w = frame_h * self.target_aspect
        
        # Calculate frame position (centered on the faces)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        frame_x = center_x - frame_w / 2
        frame_y = center_y - frame_h / 2
        
        # Create framing rect
        rect = FramingRect(frame_x, frame_y, frame_w, frame_h)
        
        # Ensure framing is within the image bounds
        rect = self._clip_to_frame(rect, frame_width, frame_height)
        
        return rect
    
    def _ensure_context_preservation(
        self,
        rect: FramingRect,
        faces: Dict[int, TrackedFace],
        frame_width: int,
        frame_height: int
    ) -> FramingRect:
        """
        Adjust framing to preserve context and include important faces.
        
        Args:
            rect: Current framing rectangle
            faces: Dictionary of tracked faces
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Adjusted framing rectangle
        """
        if not self.preserve_context or len(faces) <= 1:
            return rect
        
        # Check if important faces are outside the current framing
        important_faces_outside = False
        for face_id, face in faces.items():
            # Skip unstable tracks
            if not face.is_stable():
                continue
                
            box = face.box
            
            # Check if this face is outside the current framing
            if (box.x1 < rect.x or box.x2 > rect.right or
                box.y1 < rect.y or box.y2 > rect.bottom):
                # This face is important if:
                # 1. It's stable (already checked)
                # 2. It has a high confidence
                if face.avg_confidence() > 0.7:
                    important_faces_outside = True
                    break
        
        if important_faces_outside:
            # If important faces are outside, try to include them by using multi-face framing
            multi_rect = self._calculate_multi_face_framing(faces, frame_width, frame_height)
            
            # Blend between the original rect and the multi-face rect
            # This ensures smooth transitions
            blend_factor = 0.7  # How much to favor the multi-face framing
            
            blended_rect = FramingRect(
                x=rect.x * (1 - blend_factor) + multi_rect.x * blend_factor,
                y=rect.y * (1 - blend_factor) + multi_rect.y * blend_factor,
                width=rect.width * (1 - blend_factor) + multi_rect.width * blend_factor,
                height=rect.height * (1 - blend_factor) + multi_rect.height * blend_factor
            )
            
            return blended_rect
        
        return rect
    
    def _smooth_framing(self, target_rect: FramingRect) -> FramingRect:
        """
        Apply temporal smoothing to framing for smoother camera movements.
        
        Args:
            target_rect: Target framing rectangle
            
        Returns:
            Smoothed framing rectangle
        """
        # Initialize current framing if needed
        if self.current_framing is None:
            self.current_framing = target_rect
            return target_rect
        
        # Calculate distance between current and target framing
        dx = target_rect.center_x - self.current_framing.center_x
        dy = target_rect.center_y - self.current_framing.center_y
        dw = target_rect.width - self.current_framing.width
        dh = target_rect.height - self.current_framing.height
        
        # Calculate distance as percentage of frame size
        dist_pct = np.sqrt(dx*dx + dy*dy) / max(self.current_framing.width, self.current_framing.height)
        
        # Adjust smoothing factor based on distance
        # For small movements, use more smoothing; for large movements, less smoothing
        adaptive_smoothing = self.smoothing_factor
        if dist_pct > 0.2:  # Large movement
            adaptive_smoothing = max(0.5, self.smoothing_factor - 0.2)
        
        # Don't move if the change is too small
        if (dist_pct < self.min_movement_threshold and 
            abs(dw) / self.current_framing.width < self.min_movement_threshold and
            abs(dh) / self.current_framing.height < self.min_movement_threshold):
            return self.current_framing
        
        # Apply exponential smoothing
        smoothed_rect = FramingRect(
            x=self.current_framing.x * adaptive_smoothing + target_rect.x * (1 - adaptive_smoothing),
            y=self.current_framing.y * adaptive_smoothing + target_rect.y * (1 - adaptive_smoothing),
            width=self.current_framing.width * adaptive_smoothing + target_rect.width * (1 - adaptive_smoothing),
            height=self.current_framing.height * adaptive_smoothing + target_rect.height * (1 - adaptive_smoothing)
        )
        
        # Update current framing
        self.current_framing = smoothed_rect
        
        return smoothed_rect
    
    def _clip_to_frame(
        self,
        rect: FramingRect,
        frame_width: int,
        frame_height: int
    ) -> FramingRect:
        """
        Ensure framing rectangle stays within frame boundaries.
        
        Args:
            rect: Framing rectangle
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Clipped framing rectangle
        """
        # Ensure rectangle doesn't go outside frame boundaries
        x = max(0, min(rect.x, frame_width - rect.width))
        y = max(0, min(rect.y, frame_height - rect.height))
        
        # If rectangle is larger than frame, scale it down
        width = min(rect.width, frame_width)
        height = min(rect.height, frame_height)
        
        return FramingRect(x, y, width, height)
    
    def _adjust_aspect_ratio(self, rect: FramingRect) -> FramingRect:
        """
        Adjust rectangle to match target aspect ratio.
        
        Args:
            rect: Input rectangle
            
        Returns:
            Rectangle with adjusted aspect ratio
        """
        current_aspect = rect.width / rect.height
        
        if abs(current_aspect - self.target_aspect) < 0.01:
            # Aspect ratio is already close enough
            return rect
        
        if current_aspect > self.target_aspect:
            # Too wide, adjust height
            new_height = rect.width / self.target_aspect
            y_offset = (new_height - rect.height) / 2
            return FramingRect(rect.x, rect.y - y_offset, rect.width, new_height)
        else:
            # Too tall, adjust width
            new_width = rect.height * self.target_aspect
            x_offset = (new_width - rect.width) / 2
            return FramingRect(rect.x - x_offset, rect.y, new_width, rect.height)
    
    def _apply_framing(self, image: np.ndarray, rect: FramingRect) -> np.ndarray:
        """
        Apply framing rectangle to an image.
        
        Args:
            image: Input image
            rect: Framing rectangle
            
        Returns:
            Framed and resized image
        """
        height, width = image.shape[:2]
        
        # Convert rectangle to integer coordinates
        x, y, w, h = rect.to_tuple()
        
        # Ensure coordinates are valid
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        # Crop the image
        cropped = image[y:y+h, x:x+w]
        
        # Resize to target dimensions
        resized = cv2.resize(cropped, (self.target_width, self.target_height))
        
        return resized 