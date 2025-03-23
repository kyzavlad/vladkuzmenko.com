"""
Ensemble Face Detector Module

This module provides a face detector implementation that combines multiple face detection models
for improved accuracy and robustness across different conditions.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum

from app.clip_generation.services.face_tracking import FaceDetector, FaceBox, DetectionModel
from app.clip_generation.services.face_detection_yolo import YOLOFaceDetector
from app.clip_generation.services.face_detection_mediapipe import MediaPipeFaceDetector
from app.clip_generation.services.face_detection_retinaface import RetinaFaceDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleMode(str, Enum):
    """Enumeration of ensemble modes."""
    CASCADE = "cascade"  # Try models in order until faces are found
    PARALLEL = "parallel"  # Run all models and merge results
    WEIGHTED = "weighted"  # Run all models and use weighted confidence


class EnsembleFaceDetector(FaceDetector):
    """
    Face detector that combines multiple detection models.
    
    This detector uses an ensemble of face detection models to provide
    more robust and accurate face detection across various conditions.
    """
    
    def __init__(
        self,
        models: List[str] = [DetectionModel.YOLO, DetectionModel.MEDIAPIPE, DetectionModel.RETINAFACE],
        mode: EnsembleMode = EnsembleMode.CASCADE,
        models_dir: str = "models",
        iou_threshold: float = 0.5,
        device: str = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    ):
        """
        Initialize ensemble face detector.
        
        Args:
            models: List of detection models to use
            mode: Ensemble mode (cascade, parallel, or weighted)
            models_dir: Directory containing model weights
            iou_threshold: IoU threshold for merging detections
            device: Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(None, device)
        self.model_type = "ensemble"
        self.models_list = models
        self.mode = mode
        self.models_dir = models_dir
        self.iou_threshold = iou_threshold
        
        # Initialize individual detectors
        self.detectors = {}
        self._init_detectors()
        
        logger.info(f"Initialized ensemble face detector with {len(self.models_list)} models in {mode} mode")
    
    def _init_detectors(self) -> None:
        """Initialize individual face detectors."""
        try:
            for model_name in self.models_list:
                if model_name == DetectionModel.YOLO:
                    logger.info("Initializing YOLO face detector")
                    model_path = os.path.join(self.models_dir, "yolov8n-face.pt")
                    self.detectors[model_name] = YOLOFaceDetector(
                        model_path=model_path if os.path.exists(model_path) else None,
                        device=self.device
                    )
                
                elif model_name == DetectionModel.MEDIAPIPE:
                    logger.info("Initializing MediaPipe face detector")
                    self.detectors[model_name] = MediaPipeFaceDetector(
                        device=self.device
                    )
                
                elif model_name == DetectionModel.RETINAFACE:
                    logger.info("Initializing RetinaFace detector")
                    model_path = os.path.join(self.models_dir, "retinaface_resnet50.pth")
                    self.detectors[model_name] = RetinaFaceDetector(
                        model_path=model_path if os.path.exists(model_path) else None,
                        device=self.device
                    )
            
            if not self.detectors:
                logger.warning("No face detectors were initialized")
                
        except Exception as e:
            logger.error(f"Error initializing detectors: {str(e)}")
            raise
    
    def load_model(self) -> None:
        """Load all models in the ensemble."""
        # Models are loaded during initialization of individual detectors
        pass
    
    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces in a frame using the ensemble of detectors.
        
        Args:
            frame: Input frame as numpy array (BGR)
            
        Returns:
            List of detected face boxes
        """
        if not self.detectors:
            logger.error("No face detectors available in ensemble")
            return []
        
        try:
            if self.mode == EnsembleMode.CASCADE:
                return self._detect_cascade(frame)
            elif self.mode == EnsembleMode.PARALLEL:
                return self._detect_parallel(frame)
            elif self.mode == EnsembleMode.WEIGHTED:
                return self._detect_weighted(frame)
            else:
                logger.error(f"Unknown ensemble mode: {self.mode}")
                return []
                
        except Exception as e:
            logger.error(f"Error during ensemble face detection: {str(e)}")
            return []
    
    def _detect_cascade(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces using cascade mode.
        
        In cascade mode, detectors are tried in order until faces are found.
        This is faster but relies on the first successful detector.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected face boxes
        """
        for model_name, detector in self.detectors.items():
            face_boxes = detector.detect(frame)
            if face_boxes:
                logger.debug(f"Found {len(face_boxes)} faces with {model_name} detector")
                return face_boxes
        
        # If no faces found with any detector
        return []
    
    def _detect_parallel(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces using parallel mode.
        
        In parallel mode, all detectors are run and results are merged
        using non-maximum suppression to remove duplicates.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected face boxes
        """
        all_faces = []
        
        # Run all detectors
        for model_name, detector in self.detectors.items():
            face_boxes = detector.detect(frame)
            all_faces.extend(face_boxes)
            logger.debug(f"Found {len(face_boxes)} faces with {model_name} detector")
        
        # Merge results using NMS
        return self._merge_detections(all_faces)
    
    def _detect_weighted(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces using weighted mode.
        
        In weighted mode, all detectors are run and results are merged
        with confidence scores weighted by detector reliability.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected face boxes
        """
        all_faces = []
        
        # Detector weights (higher is better)
        weights = {
            DetectionModel.YOLO: 1.0,
            DetectionModel.MEDIAPIPE: 0.8,
            DetectionModel.RETINAFACE: 0.9
        }
        
        # Run all detectors
        for model_name, detector in self.detectors.items():
            face_boxes = detector.detect(frame)
            
            # Apply weight to confidence scores
            weight = weights.get(model_name, 0.5)
            for face in face_boxes:
                face.confidence *= weight
            
            all_faces.extend(face_boxes)
            logger.debug(f"Found {len(face_boxes)} faces with {model_name} detector")
        
        # Merge results
        return self._merge_detections(all_faces)
    
    def _merge_detections(self, detections: List[FaceBox]) -> List[FaceBox]:
        """
        Merge multiple detections of the same face using NMS.
        
        Args:
            detections: List of all detected face boxes
            
        Returns:
            List of merged face boxes
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        # Apply NMS
        kept_detections = []
        while detections:
            # Take the highest confidence detection
            best = detections.pop(0)
            kept_detections.append(best)
            
            # Remove overlapping detections with lower confidence
            detections = [
                det for det in detections 
                if self._calculate_iou(best, det) < self.iou_threshold
            ]
        
        return kept_detections
    
    def _calculate_iou(self, box1: FaceBox, box2: FaceBox) -> float:
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU score (0-1)
        """
        # Calculate intersection area
        x_left = max(box1.x1, box2.x1)
        y_top = max(box1.y1, box2.y1)
        x_right = min(box1.x2, box2.x2)
        y_bottom = min(box1.y2, box2.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = box1.width * box1.height
        box2_area = box2.width * box2.height
        union_area = box1_area + box2_area - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area 