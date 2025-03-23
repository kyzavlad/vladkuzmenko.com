"""
Face Tracking Manager Module

This module provides the central manager for face tracking in the Clip Generation Service,
coordinating detection, recognition, and tracking of faces across video frames.
"""

import os
import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from app.clip_generation.services.face_detection_ensemble import EnsembleFaceDetector, FaceDetector
from app.clip_generation.services.face_recognition import ArcFaceRecognizer, FaceIdentity
from app.clip_generation.services.face_tracking_kalman import FaceTrackingFilter
from app.clip_generation.services.face_tracking import FaceBox
from app.clip_generation.services.face_tracking_optimizer import FaceTrackingOptimizer, SamplingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    """Represents a tracked face across multiple frames."""
    face_id: int
    box: FaceBox
    first_seen: int
    last_seen: int
    embedding: Optional[np.ndarray] = None
    identity: Optional[FaceIdentity] = None
    track_length: int = 1
    is_speaker: bool = False
    frames_as_speaker: int = 0
    confidence_history: List[float] = field(default_factory=list)
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    
    def update(self, box: FaceBox, frame_idx: int) -> None:
        """
        Update the tracked face with a new detection.
        
        Args:
            box: New face box detection
            frame_idx: Current frame index
        """
        self.box = box
        self.last_seen = frame_idx
        self.track_length += 1
        
        if box.confidence is not None:
            self.confidence_history.append(box.confidence)
        
        # Track center position
        cx = (box.x1 + box.x2) / 2
        cy = (box.y1 + box.y2) / 2
        self.position_history.append((cx, cy))
        
        # Keep history at a reasonable size
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]
        
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]
    
    def avg_confidence(self) -> float:
        """Calculate the average confidence of this tracked face."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def is_stable(self) -> bool:
        """Check if this tracked face is stable (has been tracked for a sufficient time)."""
        return self.track_length >= 5


class FaceTrackingManager:
    """
    Manager for face tracking in video clips.
    
    This class coordinates face detection, recognition, and tracking
    across video frames, providing a unified interface for face analysis.
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        detection_interval: int = 5,
        recognition_interval: int = 10,
        detection_threshold: float = 0.5,
        recognition_threshold: float = 0.6,
        max_faces: int = 10,
        device: str = "cpu",
        track_mode: str = "kalman",
        sampling_strategy: str = "uniform",
        batch_size: int = 4,
        worker_threads: int = 2,
        use_gpu: bool = None  # Auto-detect if None
    ):
        """
        Initialize the face tracking manager.
        
        Args:
            model_dir: Directory containing model files
            detection_interval: Run detection every N frames
            recognition_interval: Run recognition every N frames
            detection_threshold: Minimum confidence for detections
            recognition_threshold: Minimum similarity for recognition
            max_faces: Maximum number of faces to track
            device: Device to run inference on ('cpu' or 'cuda')
            track_mode: Tracking mode ('kalman' or 'simple')
            sampling_strategy: Frame sampling strategy ('uniform', 'adaptive', 'keyframe', 'motion')
            batch_size: Size of frame batches for batch processing
            worker_threads: Number of worker threads for parallel processing
            use_gpu: Whether to use GPU acceleration (auto-detect if None)
        """
        self.model_dir = model_dir
        self.detection_interval = detection_interval
        self.recognition_interval = recognition_interval
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.max_faces = max_faces
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            import torch
            use_gpu = torch.cuda.is_available()
        
        # Set device
        if use_gpu and device == "cpu":
            self.device = "cuda"
        else:
            self.device = device
            
        self.track_mode = track_mode
        self.frame_id = 0
        self.speaker_id = None  # Current speaker face ID
        
        # Performance optimization settings
        self.sampling_strategy = sampling_strategy
        self.batch_size = batch_size
        self.worker_threads = worker_threads
        self.use_gpu = use_gpu
        
        # Initialize components
        self.face_detector = None
        self.face_recognizer = None
        self.tracking_filter = None
        self.optimizer = None
        
        # State
        self.tracked_faces = {}  # face_id -> TrackedFace
        self.next_face_id = 0
        
        # Initialize face tracking components
        self._init_components()
        
        # Processing statistics
        self.stats = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "detection_time": [],
            "tracking_time": [],
            "recognition_time": [],
            "faces_detected": []
        }
        
        logger.info(f"Initialized FaceTrackingManager with {self.face_detector.__class__.__name__}")

    def _init_components(self) -> None:
        """Initialize face tracking components."""
        # Initialize face detector (ensemble of models)
        self.face_detector = EnsembleFaceDetector(
            mode="cascade",
            model_dir=self.model_dir,
            iou_threshold=0.5,
            device=self.device
        )
        
        # Initialize face recognizer
        self.face_recognizer = ArcFaceRecognizer(
            model_path=os.path.join(self.model_dir, "arcface_model.pth"),
            recognition_threshold=self.recognition_threshold,
            device=self.device
        )
        
        # Initialize face tracking filter
        self.tracking_filter = FaceTrackingFilter()
        
        # Initialize the performance optimizer
        self.optimizer = FaceTrackingOptimizer(
            sampling_strategy=SamplingStrategy(self.sampling_strategy),
            sampling_rate=self.detection_interval,
            use_gpu=self.use_gpu,
            batch_size=self.batch_size,
            worker_threads=self.worker_threads,
            device=self.device
        )
        
        # Start batch processing if using batch mode and worker threads > 0
        if self.worker_threads > 0:
            self.optimizer.start_batch_processing(self.face_detector)
        
        logger.info("Face tracking components initialized")

    def process_frame(self, frame: np.ndarray) -> Dict[int, TrackedFace]:
        """
        Process a frame with face tracking.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary of tracked faces
        """
        start_time = time.time()
        self.frame_id += 1
        h, w = frame.shape[:2]
        
        # Check if we should process this frame based on sampling strategy
        should_process = self.optimizer.should_process_frame(frame)
        
        if should_process:
            # Performance-optimized detection
            if self.worker_threads > 0:
                # Add to batch processing queue
                self.optimizer.add_to_batch_queue(frame, self.frame_id)
                
                # Get any available results from previous frames
                batch_results = self.optimizer.get_batch_results()
                
                for result_frame_id, face_boxes in batch_results:
                    # Update tracked faces with detection results
                    self._update_tracked_faces(face_boxes, frame)
                    
                    # Update statistics
                    self.stats["frames_processed"] += 1
                    self.stats["faces_detected"].append(len(face_boxes))
            else:
                # Optimize frame for detection
                optimized_frame = self.optimizer.optimize_frame(frame)
                
                # Detect faces in the current frame
                detection_start = time.time()
                detected_faces = self.face_detector.detect(optimized_frame)
                detection_time = time.time() - detection_start
                
                # Update tracked faces with detection results
                tracking_start = time.time()
                self._update_tracked_faces(detected_faces, frame)
                tracking_time = time.time() - tracking_start
                
                # Update statistics
                self.stats["frames_processed"] += 1
                self.stats["detection_time"].append(detection_time)
                self.stats["tracking_time"].append(tracking_time)
                self.stats["faces_detected"].append(len(detected_faces))
                
                # Update adaptive sampling rate
                self.optimizer.update_adaptive_rate(len(detected_faces), detection_time)
        else:
            # Skip detection, but still update tracking
            if self.tracking_filter and self.tracked_faces:
                for face_id in list(self.tracked_faces.keys()):
                    predicted_box = self.tracking_filter.predict(face_id)
                    if predicted_box:
                        self.tracked_faces[face_id].box = predicted_box
                        
            self.stats["frames_skipped"] += 1
        
        # Run face recognition at specified interval
        if self.frame_id % self.recognition_interval == 0 and self.face_recognizer:
            recognition_start = time.time()
            self._update_face_identities(frame)
            recognition_time = time.time() - recognition_start
            if "recognition_time" in self.stats:
                self.stats["recognition_time"].append(recognition_time)
        
        # Update speaker detection
        self._update_speaker_detection()
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Cap the stats arrays to avoid memory growth
        for key in ["detection_time", "tracking_time", "recognition_time", "faces_detected"]:
            if key in self.stats and len(self.stats[key]) > 100:
                self.stats[key] = self.stats[key][-100:]
        
        return self.tracked_faces
    
    def _update_tracked_faces(self, detected_faces: List[FaceBox], frame: np.ndarray) -> None:
        """
        Update tracked faces with new detections.
        
        Args:
            detected_faces: List of detected face boxes in current frame
            frame: Current video frame
        """
        # If we have detections, use them to update existing tracks
        if detected_faces:
            matched_faces = self._match_faces_to_tracks(detected_faces)
            
            # Update existing tracks with new detections
            for face_id, face_box in matched_faces.items():
                # Apply Kalman filter
                filtered_box = self.tracking_filter.update(face_id, face_box, self.frame_id)
                
                if face_id in self.tracked_faces:
                    # Update existing tracked face
                    self.tracked_faces[face_id].update(filtered_box, self.frame_id)
                else:
                    # Create new tracked face
                    self.tracked_faces[face_id] = TrackedFace(
                        face_id=face_id,
                        box=filtered_box,
                        first_seen=self.frame_id,
                        last_seen=self.frame_id
                    )
        
        # For faces not detected in this frame, use Kalman filter prediction
        else:
            for face_id in list(self.tracked_faces.keys()):
                # Get prediction for this face
                predicted_box = self.tracking_filter.predict(face_id)
                
                if predicted_box is not None:
                    # Update tracked face with prediction
                    self.tracked_faces[face_id].update(predicted_box, self.frame_id)
        
        # Clean up old tracks
        self._cleanup_tracks()
    
    def _match_faces_to_tracks(self, detected_faces: List[FaceBox]) -> Dict[int, FaceBox]:
        """
        Match detected faces to existing tracks using IoU.
        
        Args:
            detected_faces: List of detected face boxes
            
        Returns:
            Dictionary mapping face IDs to their matched face boxes
        """
        matched_faces = {}
        
        # If no tracked faces yet, assign new IDs to all detections
        if not self.tracked_faces:
            for face_box in detected_faces:
                face_id = self.next_face_id
                self.next_face_id += 1
                matched_faces[face_id] = face_box
            return matched_faces
        
        # Get predictions for all existing tracks
        predictions = self.tracking_filter.get_all_predictions()
        
        # Calculate IoU between all detections and all predictions
        iou_matrix = np.zeros((len(detected_faces), len(predictions)))
        
        for i, face_box in enumerate(detected_faces):
            for j, (face_id, pred_box) in enumerate(predictions.items()):
                iou = self._calculate_iou(face_box, pred_box)
                iou_matrix[i, j] = iou
        
        # Match detections to predictions
        # Simple greedy matching for now
        while np.max(iou_matrix) > 0.3:  # IoU threshold for matching
            # Find best match
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            # Get face ID for this prediction
            face_id = list(predictions.keys())[j]
            
            # Assign detection to track
            matched_faces[face_id] = detected_faces[i]
            
            # Remove this pair from consideration
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        # Handle unmatched detections
        for i, face_box in enumerate(detected_faces):
            if np.max(iou_matrix[i, :]) == 0:  # Unmatched detection
                # Create new track
                face_id = self.next_face_id
                self.next_face_id += 1
                matched_faces[face_id] = face_box
        
        return matched_faces
    
    def _calculate_iou(self, box1: FaceBox, box2: FaceBox) -> float:
        """
        Calculate Intersection over Union between two face boxes.
        
        Args:
            box1: First face box
            box2: Second face box
            
        Returns:
            IoU value between 0 and 1
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
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _update_face_identities(self, frame: np.ndarray) -> None:
        """
        Update face identities using face recognition.
        
        Args:
            frame: Current video frame
        """
        for face_id, tracked_face in self.tracked_faces.items():
            # Skip faces that already have an identity
            if tracked_face.identity is not None:
                continue
            
            # Only recognize stable tracks
            if not tracked_face.is_stable():
                continue
            
            # Extract face region from frame
            face_box = tracked_face.box
            face_image = frame[int(face_box.y1):int(face_box.y2), int(face_box.x1):int(face_box.x2)]
            
            if face_image.size == 0:
                continue
            
            # Extract face embedding
            embedding = self.face_recognizer.extract_embedding(face_image)
            
            if embedding is None:
                continue
            
            # Store embedding
            tracked_face.embedding = embedding
            
            # Find matching identity
            identity = self.face_recognizer.find_matching_face(embedding)
            
            if identity is not None:
                tracked_face.identity = identity
                logger.info(f"Frame {self.frame_id}: Face {face_id} identified as {identity.name}")
    
    def _update_speaker_detection(self) -> None:
        """
        Update speaker detection based on face tracking.
        
        This implements a simple speaker detection algorithm that considers:
        1. Face size (larger faces are more likely to be speakers)
        2. Face position (centered faces are more likely to be speakers)
        3. Face stability (consistently tracked faces are more likely to be speakers)
        """
        if not self.tracked_faces:
            return
        
        speaker_scores = {}
        
        # Calculate speaker score for each tracked face
        for face_id, tracked_face in self.tracked_faces.items():
            # Skip unstable tracks
            if not tracked_face.is_stable():
                continue
            
            # Calculate face size score (larger = higher score)
            face_box = tracked_face.box
            size_score = face_box.width * face_box.height
            
            # Calculate position score (closer to center = higher score)
            center_x = (face_box.x1 + face_box.x2) / 2
            center_y = (face_box.y1 + face_box.y2) / 2
            
            # Assume center of frame is at (0.5, 0.5) in normalized coordinates
            position_score = 1.0 - (abs(center_x - 0.5) + abs(center_y - 0.5))
            
            # Calculate stability score (longer tracks = higher score)
            stability_score = min(1.0, tracked_face.track_length / 30.0)
            
            # Calculate identity score (identified faces get higher score)
            identity_score = 1.0 if tracked_face.identity is not None else 0.5
            
            # Calculate final score
            final_score = (
                0.4 * size_score +
                0.3 * position_score +
                0.2 * stability_score +
                0.1 * identity_score
            )
            
            speaker_scores[face_id] = final_score
        
        # Find face with highest speaker score
        if speaker_scores:
            new_speaker_id = max(speaker_scores.keys(), key=lambda k: speaker_scores[k])
            
            # If we have a new speaker, update speaker status
            if new_speaker_id != self.speaker_id:
                # Update previous speaker status
                if self.speaker_id is not None and self.speaker_id in self.tracked_faces:
                    self.tracked_faces[self.speaker_id].is_speaker = False
                
                # Update new speaker status
                self.speaker_id = new_speaker_id
                self.tracked_faces[new_speaker_id].is_speaker = True
                self.tracked_faces[new_speaker_id].frames_as_speaker += 1
                
                logger.info(f"Frame {self.frame_id}: Speaker changed to face {new_speaker_id}")
            
            # Update frames as speaker for current speaker
            elif self.speaker_id is not None and self.speaker_id in self.tracked_faces:
                self.tracked_faces[self.speaker_id].frames_as_speaker += 1
    
    def _cleanup_tracks(self) -> None:
        """Clean up old face tracks that haven't been updated recently."""
        # Remove faces from tracker
        removed_ids = self.tracking_filter.cleanup(self.frame_id)
        
        # Remove corresponding tracked faces
        for face_id in removed_ids:
            if face_id in self.tracked_faces:
                logger.debug(f"Frame {self.frame_id}: Removing track {face_id}")
                del self.tracked_faces[face_id]
        
        # Update speaker if needed
        if self.speaker_id not in self.tracked_faces:
            self.speaker_id = None
    
    def get_speaker_face(self) -> Optional[TrackedFace]:
        """Get the current speaker face."""
        if self.speaker_id is not None and self.speaker_id in self.tracked_faces:
            return self.tracked_faces[self.speaker_id]
        return None
    
    def register_identity(self, face_id: int, name: str) -> bool:
        """
        Register a face as a named identity.
        
        Args:
            face_id: ID of the face to register
            name: Name to associate with this face
            
        Returns:
            True if registration was successful, False otherwise
        """
        if face_id not in self.tracked_faces:
            logger.warning(f"Cannot register identity: Face ID {face_id} not found")
            return False
        
        tracked_face = self.tracked_faces[face_id]
        
        if tracked_face.embedding is None:
            logger.warning(f"Cannot register identity: Face ID {face_id} has no embedding")
            return False
        
        # Register identity
        self.face_recognizer.register_identity(tracked_face.embedding, name)
        
        # Update tracked face
        identity = self.face_recognizer.find_matching_face(tracked_face.embedding)
        if identity is not None:
            tracked_face.identity = identity
            logger.info(f"Registered face {face_id} as identity '{name}'")
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset the face tracking manager."""
        self.tracked_faces.clear()
        self.next_face_id = 0
        self.frame_id = 0
        self.speaker_id = None
        
        if self.tracking_filter:
            self.tracking_filter.reset()
        
        if self.optimizer:
            self.optimizer.stop_batch_processing()
        
        logger.info("Face tracking manager reset")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'optimizer') and self.optimizer:
            self.optimizer.stop_batch_processing() 