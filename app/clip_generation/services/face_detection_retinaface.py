"""
RetinaFace Detector Module

This module provides a face detector implementation using RetinaFace.
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


class RetinaFaceDetector(FaceDetector):
    """
    Face detector using RetinaFace model.
    
    This detector uses RetinaFace for robust face detection particularly
    in challenging lighting conditions and with various face orientations.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        network: str = "resnet50",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cpu",
        scales: List[int] = [640, 1080]
    ):
        """
        Initialize RetinaFace detector.
        
        Args:
            model_path: Path to RetinaFace model weights
            network: Network backbone ('resnet50' or 'mobilenet')
            conf_threshold: Confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu' or 'cuda')
            scales: List of image scales to use
        """
        super().__init__(model_path, device)
        self.model_type = "retinaface"
        self.network = network
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.scales = scales
        
        # Default model path if not provided
        if not self.model_path:
            self.model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models",
                f"retinaface_{network}.pth"
            )
        
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the RetinaFace model."""
        try:
            # Importing here to avoid dependency if not using this model
            # This assumes you have the retinaface implementation available
            try:
                from retinaface.model import retinaface_model
                from retinaface.commons import preprocess, postprocess
                self.preprocess = preprocess
                self.postprocess = postprocess
            except ImportError:
                # Fallback to local implementation if library not available
                logger.warning("RetinaFace library not found. Using local implementation.")
                from app.clip_generation.services.retinaface_utils import load_retinaface_model
                self.model = load_retinaface_model(
                    self.model_path, 
                    self.network, 
                    self.device
                )
                return
            
            logger.info(f"Loading RetinaFace model ({self.network}) from {self.model_path}")
            
            self.model = retinaface_model(
                model_path=self.model_path,
                network=self.network,
                device=self.device
            )
            
            logger.info("RetinaFace model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading RetinaFace model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces in a frame using RetinaFace.
        
        Args:
            frame: Input frame as numpy array (BGR)
            
        Returns:
            List of detected face boxes with landmarks
        """
        if self.model is None:
            logger.error("RetinaFace model not loaded")
            return []
        
        try:
            # RetinaFace detection
            resp = self._detect_faces(frame)
            
            face_boxes = []
            
            if resp is not None:
                for face in resp:
                    box = face['facial_area']
                    x1, y1, x2, y2 = box
                    confidence = face.get('score', 0.9)  # Default if not provided
                    
                    # Extract landmarks if available
                    landmarks = None
                    if 'landmarks' in face:
                        landmarks_dict = face['landmarks']
                        # Format: [left_eye, right_eye, nose, mouth_right, mouth_left]
                        landmarks = np.array([
                            landmarks_dict['right_eye'],  # Right eye (from their perspective)
                            landmarks_dict['left_eye'],   # Left eye
                            landmarks_dict['nose'],       # Nose
                            landmarks_dict['mouth_right'],# Right mouth corner
                            landmarks_dict['mouth_left']  # Left mouth corner
                        ])
                    
                    face_box = FaceBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(confidence),
                        landmarks=landmarks,
                        model="retinaface"
                    )
                    face_boxes.append(face_box)
            
            return face_boxes
            
        except Exception as e:
            logger.error(f"Error during RetinaFace detection: {str(e)}")
            return []
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Internal method to detect faces using RetinaFace.
        
        Args:
            frame: Input frame as numpy array (BGR)
            
        Returns:
            List of face detections with facial area and landmarks
        """
        try:
            # Convert to RGB if needed
            if frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame.copy()
            
            # Implementation depends on the specific RetinaFace package
            # This is a generic implementation
            if hasattr(self.model, 'detect_faces'):
                # Using retinaface-pytorch or similar
                return self.model.detect_faces(
                    rgb_frame, 
                    threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold
                )
            else:
                # Using alternative implementation
                # Resize to optimal scale
                h, w = frame.shape[:2]
                target_size = self.scales[0]
                max_size = self.scales[1]
                
                im_size_min = min(h, w)
                im_size_max = max(h, w)
                
                scale = target_size / im_size_min
                if scale * im_size_max > max_size:
                    scale = max_size / im_size_max
                
                # Resize image
                resized = cv2.resize(
                    rgb_frame, 
                    (int(w * scale), int(h * scale))
                )
                
                # Detect faces
                faces = self.model.detect(
                    resized, 
                    self.conf_threshold, 
                    self.nms_threshold
                )
                
                # Convert back to original scale
                result = []
                if faces is not None:
                    for face in faces:
                        box = face[0:4] / scale
                        landmarks = face[5:15].reshape((5, 2)) / scale
                        confidence = face[4]
                        
                        result.append({
                            'facial_area': box.astype(int).tolist(),
                            'score': float(confidence),
                            'landmarks': {
                                'right_eye': landmarks[0].tolist(),
                                'left_eye': landmarks[1].tolist(), 
                                'nose': landmarks[2].tolist(),
                                'mouth_right': landmarks[3].tolist(),
                                'mouth_left': landmarks[4].tolist()
                            }
                        })
                
                return result
                
        except Exception as e:
            logger.error(f"Error in RetinaFace detection: {str(e)}")
            return [] 