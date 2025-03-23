"""
YOLO Face Detector Module

This module provides a face detector implementation using YOLOv8.
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


class YOLOFaceDetector(FaceDetector):
    """
    Face detector using YOLOv8 model.
    
    This detector uses the YOLOv8 model trained specifically for face detection.
    It provides fast and accurate face detection for multiple faces in a frame.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cpu",
        img_size: int = 640
    ):
        """
        Initialize YOLOv8 face detector.
        
        Args:
            model_path: Path to YOLOv8 face detection model weights
            conf_threshold: Confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu' or 'cuda')
            img_size: Input image size for the model
        """
        super().__init__(model_path, device)
        self.model_type = "yolov8"
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.img_size = img_size
        
        # Default model path if not provided
        if not self.model_path:
            self.model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models",
                "yolov8n-face.pt"
            )
        
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the YOLOv8 model."""
        try:
            # Importing here to avoid dependency if not using this model
            from ultralytics import YOLO
            
            if not os.path.exists(self.model_path):
                logger.warning(f"YOLOv8 model not found at {self.model_path}. " 
                             f"Will attempt to load from Ultralytics hub.")
            
            logger.info(f"Loading YOLOv8 face detection model from {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Set model parameters
            self.model.conf = self.conf_threshold
            self.model.iou = self.nms_threshold
            
            # Move model to specified device
            self.model.to(self.device)
            
            logger.info("YOLOv8 face detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces in a frame using YOLOv8.
        
        Args:
            frame: Input frame as numpy array (BGR)
            
        Returns:
            List of detected face boxes
        """
        if self.model is None:
            logger.error("YOLOv8 model not loaded")
            return []
        
        try:
            # YOLOv8 expects BGR images by default
            results = self.model(frame, size=self.img_size, verbose=False)[0]
            
            face_boxes = []
            
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                # Process detections
                for box in results.boxes:
                    # Extract box data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only keep face detections (class 0)
                    if class_id == 0:
                        face_box = FaceBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=confidence,
                            model="yolov8"
                        )
                        face_boxes.append(face_box)
            
            return face_boxes
            
        except Exception as e:
            logger.error(f"Error during YOLOv8 face detection: {str(e)}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[FaceBox]]:
        """
        Detect faces in a batch of frames using YOLOv8.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detected face boxes for each frame
        """
        if self.model is None:
            logger.error("YOLOv8 model not loaded")
            return [[] for _ in frames]
        
        try:
            # YOLOv8 batch inference
            batch_results = self.model(frames, size=self.img_size, verbose=False)
            
            all_face_boxes = []
            
            for results in batch_results:
                face_boxes = []
                
                if hasattr(results, 'boxes') and len(results.boxes) > 0:
                    # Process detections
                    for box in results.boxes:
                        # Extract box data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only keep face detections (class 0)
                        if class_id == 0:
                            face_box = FaceBox(
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2),
                                confidence=confidence,
                                model="yolov8"
                            )
                            face_boxes.append(face_box)
                
                all_face_boxes.append(face_boxes)
            
            return all_face_boxes
            
        except Exception as e:
            logger.error(f"Error during YOLOv8 batch face detection: {str(e)}")
            return [[] for _ in frames] 