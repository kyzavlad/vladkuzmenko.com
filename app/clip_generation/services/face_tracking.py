"""
Face Tracking Base Module

This module provides the base classes and data structures for face tracking,
including the FaceBox dataclass and abstract FaceDetector class.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FaceBox:
    """
    Represents a face bounding box with optional landmarks and metadata.
    
    This class is used to store face detection results, including box coordinates,
    confidence scores, facial landmarks, and other metadata.
    """
    x1: float  # Left coordinate
    y1: float  # Top coordinate
    x2: float  # Right coordinate
    y2: float  # Bottom coordinate
    confidence: Optional[float] = None  # Detection confidence
    landmarks: Optional[List[Tuple[float, float]]] = None  # Facial landmarks
    face_id: Optional[int] = None  # Face ID for tracking
    embedding: Optional[np.ndarray] = None  # Face embedding for recognition
    model: Optional[str] = None  # Detection model name
    
    @property
    def width(self) -> float:
        """Get the width of the face box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Get the height of the face box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Get the area of the face box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center coordinates of the face box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert face box to dictionary."""
        result = {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "center": self.center,
        }
        
        if self.confidence is not None:
            result["confidence"] = self.confidence
            
        if self.face_id is not None:
            result["face_id"] = self.face_id
            
        if self.model is not None:
            result["model"] = self.model
            
        # Note: we don't include embeddings or landmarks in the dict as they can be large
        result["has_landmarks"] = self.landmarks is not None
        result["has_embedding"] = self.embedding is not None
            
        return result
    
    def scale(self, scale_x: float, scale_y: float) -> "FaceBox":
        """
        Scale the face box by the given factors.
        
        Args:
            scale_x: Horizontal scale factor
            scale_y: Vertical scale factor
            
        Returns:
            Scaled face box
        """
        new_box = FaceBox(
            x1=self.x1 * scale_x,
            y1=self.y1 * scale_y,
            x2=self.x2 * scale_x,
            y2=self.y2 * scale_y,
            confidence=self.confidence,
            face_id=self.face_id,
            model=self.model
        )
        
        # Scale landmarks if present
        if self.landmarks is not None:
            new_box.landmarks = [(x * scale_x, y * scale_y) for x, y in self.landmarks]
            
        # Keep the embedding reference (no need to scale)
        new_box.embedding = self.embedding
        
        return new_box
    
    def normalize(self, frame_width: int, frame_height: int) -> "FaceBox":
        """
        Normalize box coordinates to [0, 1] range.
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Face box with normalized coordinates
        """
        return self.scale(1.0 / frame_width, 1.0 / frame_height)
    
    def denormalize(self, frame_width: int, frame_height: int) -> "FaceBox":
        """
        Convert normalized [0, 1] coordinates to absolute pixels.
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Face box with denormalized coordinates
        """
        return self.scale(frame_width, frame_height)
    
    def enlarge(self, factor: float = 1.2) -> "FaceBox":
        """
        Enlarge the face box by a factor around its center.
        
        Args:
            factor: Enlargement factor (1.0 = no change)
            
        Returns:
            Enlarged face box
        """
        cx, cy = self.center
        half_width = self.width * factor / 2
        half_height = self.height * factor / 2
        
        return FaceBox(
            x1=cx - half_width,
            y1=cy - half_height,
            x2=cx + half_width,
            y2=cy + half_height,
            confidence=self.confidence,
            landmarks=self.landmarks,
            face_id=self.face_id,
            embedding=self.embedding,
            model=self.model
        )
    
    def clip(self, min_x: float = 0, min_y: float = 0, 
             max_x: Optional[float] = None, max_y: Optional[float] = None) -> "FaceBox":
        """
        Clip face box coordinates to the given range.
        
        Args:
            min_x: Minimum x coordinate
            min_y: Minimum y coordinate
            max_x: Maximum x coordinate
            max_y: Maximum y coordinate
            
        Returns:
            Clipped face box
        """
        x1 = max(self.x1, min_x)
        y1 = max(self.y1, min_y)
        x2 = min(self.x2, max_x) if max_x is not None else self.x2
        y2 = min(self.y2, max_y) if max_y is not None else self.y2
        
        return FaceBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=self.confidence,
            landmarks=self.landmarks,
            face_id=self.face_id,
            embedding=self.embedding,
            model=self.model
        )


class FaceDetector(ABC):
    """
    Abstract base class for face detectors.
    
    This class defines the interface that all face detector implementations
    must follow, allowing for easy swapping of different detection models.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the face detector.
        
        Args:
            model_path: Path to the face detection model
            confidence_threshold: Minimum confidence score to consider a detection valid
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the face detection model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of detected face boxes
        """
        pass
    
    def __str__(self) -> str:
        """Get string representation of the face detector."""
        return f"{self.__class__.__name__}(confidence_threshold={self.confidence_threshold})" 