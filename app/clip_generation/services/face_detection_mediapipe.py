"""
MediaPipe Face Detector Module

This module provides a face detector implementation using MediaPipe Face Mesh.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

from app.clip_generation.services.face_tracking import FaceDetector, FaceBox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MediaPipeFaceDetector(FaceDetector):
    """
    Face detector using MediaPipe Face Mesh.
    
    This detector uses MediaPipe's Face Mesh model for face detection
    and provides precise facial landmarks for detailed face tracking.
    """
    
    def __init__(
        self,
        max_faces: int = 10,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_mode: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize MediaPipe Face Mesh detector.
        
        Args:
            max_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            static_mode: Whether to use static mode (no tracking)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(None, device)
        self.model_type = "mediapipe"
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_mode = static_mode
        
        # Pre-defined face mesh contours for face boundary
        self.face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the MediaPipe Face Mesh model."""
        try:
            # Importing here to avoid dependency if not using this model
            import mediapipe as mp
            
            logger.info("Loading MediaPipe Face Mesh model")
            mp_face_mesh = mp.solutions.face_mesh
            
            self.model = mp_face_mesh.FaceMesh(
                static_image_mode=self.static_mode,
                max_num_faces=self.max_faces,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            logger.info("MediaPipe Face Mesh model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading MediaPipe Face Mesh model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces in a frame using MediaPipe Face Mesh.
        
        Args:
            frame: Input frame as numpy array (BGR)
            
        Returns:
            List of detected face boxes with landmarks
        """
        if self.model is None:
            logger.error("MediaPipe Face Mesh model not loaded")
            return []
        
        try:
            # MediaPipe expects RGB images
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Process the image
            results = self.model.process(rgb_frame)
            
            face_boxes = []
            
            if results.multi_face_landmarks:
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Extract all landmarks
                    landmarks = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark])
                    
                    # Calculate bounding box from face oval landmarks
                    oval_landmarks = np.array([(face_landmarks.landmark[idx].x * w, 
                                               face_landmarks.landmark[idx].y * h) 
                                               for idx in self.face_oval_indices])
                    
                    x1, y1 = np.min(oval_landmarks, axis=0)
                    x2, y2 = np.max(oval_landmarks, axis=0)
                    
                    # Add some padding (10%)
                    width, height = x2 - x1, y2 - y1
                    x1 = max(0, x1 - width * 0.1)
                    y1 = max(0, y1 - height * 0.1)
                    x2 = min(w, x2 + width * 0.1)
                    y2 = min(h, y2 + height * 0.1)
                    
                    face_box = FaceBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=1.0,  # MediaPipe doesn't provide confidence scores
                        landmarks=landmarks,
                        model="mediapipe"
                    )
                    face_boxes.append(face_box)
            
            return face_boxes
            
        except Exception as e:
            logger.error(f"Error during MediaPipe face detection: {str(e)}")
            return []
    
    def get_facial_landmarks(self, face_box: FaceBox) -> Dict[str, np.ndarray]:
        """
        Extract facial landmarks from a face detection.
        
        Args:
            face_box: Face box with landmarks from MediaPipe
            
        Returns:
            Dictionary of facial landmarks grouped by facial features
        """
        if face_box.landmarks is None:
            return {}
        
        # MediaPipe Face Mesh landmark indices for different facial features
        facial_features = {
            "left_eye": np.array([33, 133, 157, 158, 159, 160, 161, 173, 246]),
            "right_eye": np.array([362, 263, 386, 387, 388, 390, 374, 466]),
            "left_eyebrow": np.array([70, 63, 105, 66, 107, 55, 65, 52, 53]),
            "right_eyebrow": np.array([300, 293, 334, 296, 336, 285, 295, 282, 283]),
            "nose": np.array([168, 6, 197, 195, 5, 4, 45, 220, 115, 49, 131, 134, 51, 5, 281, 363, 360]),
            "mouth": np.array([61, 185, 40, 39, 37, 0, 267, 269, 270, 409]),
            "face_oval": np.array(self.face_oval_indices)
        }
        
        # Extract landmarks for each facial feature
        landmarks_dict = {}
        for feature_name, indices in facial_features.items():
            landmarks_dict[feature_name] = face_box.landmarks[indices]
        
        return landmarks_dict 