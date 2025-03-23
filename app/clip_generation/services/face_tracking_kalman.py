"""
Face Tracking Kalman Filter Module

This module provides Kalman filtering functionality for face tracking,
helping maintain temporal consistency in face tracking across frames.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any

from app.clip_generation.services.face_tracking import FaceBox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Kalman filter for tracking face position and velocity.
    
    This implementation focuses on tracking the position and velocity
    of a face bounding box across frames, providing smoother tracking
    and prediction capabilities.
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        error_cov_post: float = 1.0
    ):
        """
        Initialize Kalman filter for face tracking.
        
        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
            error_cov_post: A priori error estimate covariance
        """
        # State: [x, y, width, height, vx, vy, vw, vh]
        # Where (x,y) is the center, (w,h) is size, (vx,vy,vw,vh) are velocities
        self.state_size = 8
        self.measurement_size = 4  # We measure [x, y, width, height]
        
        # Initialize filter state
        self.kalman = None
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.error_cov_post = error_cov_post
        
        # Initialize Kalman filter matrices
        self._init_kalman()
        
        logger.debug("Initialized Kalman filter for face tracking")
    
    def _init_kalman(self) -> None:
        """Initialize the Kalman filter matrices."""
        # Create matrices
        self.kalman = {
            # State transition matrix (A)
            'A': np.eye(self.state_size, dtype=np.float32),
            
            # Measurement matrix (H)
            'H': np.zeros((self.measurement_size, self.state_size), dtype=np.float32),
            
            # Process noise covariance matrix (Q)
            'Q': np.eye(self.state_size, dtype=np.float32) * self.process_noise,
            
            # Measurement noise covariance matrix (R)
            'R': np.eye(self.measurement_size, dtype=np.float32) * self.measurement_noise,
            
            # Error covariance matrix (P)
            'P': np.eye(self.state_size, dtype=np.float32) * self.error_cov_post,
            
            # State vector (x)
            'x': np.zeros((self.state_size, 1), dtype=np.float32),
            
            # Identity matrix (I)
            'I': np.eye(self.state_size, dtype=np.float32)
        }
        
        # Set up state transition matrix (constant velocity model)
        # Add velocity influence on position (dt=1)
        for i in range(4):
            self.kalman['A'][i, i+4] = 1.0
        
        # Set up measurement matrix (we can only measure position and size)
        for i in range(self.measurement_size):
            self.kalman['H'][i, i] = 1.0
    
    def init(self, box: FaceBox) -> None:
        """
        Initialize the Kalman filter with a face box.
        
        Args:
            box: Initial face bounding box
        """
        # Extract center and size
        cx = (box.x1 + box.x2) / 2
        cy = (box.y1 + box.y2) / 2
        width = box.width
        height = box.height
        
        # Initialize state vector
        self.kalman['x'] = np.array([
            [cx], [cy], [width], [height],  # Position and size
            [0], [0], [0], [0]              # Velocities (initially zero)
        ], dtype=np.float32)
        
        # Reset error covariance
        self.kalman['P'] = np.eye(self.state_size, dtype=np.float32) * self.error_cov_post
        
        logger.debug(f"Initialized Kalman filter with box: {box.to_dict()}")
    
    def predict(self) -> FaceBox:
        """
        Predict the next state of the tracked face.
        
        Returns:
            Predicted face box
        """
        # Predict next state
        x_pred = self.kalman['A'] @ self.kalman['x']
        
        # Update error covariance
        P_pred = self.kalman['A'] @ self.kalman['P'] @ self.kalman['A'].T + self.kalman['Q']
        
        # Store predictions
        self.kalman['x_pred'] = x_pred
        self.kalman['P_pred'] = P_pred
        
        # Convert predicted state to bounding box
        cx, cy, width, height = x_pred[0, 0], x_pred[1, 0], x_pred[2, 0], x_pred[3, 0]
        
        # Calculate bounding box coordinates
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2
        
        # Create and return face box
        return FaceBox(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            confidence=0.5,  # Arbitrary confidence for prediction
            model="kalman"
        )
    
    def update(self, box: FaceBox) -> FaceBox:
        """
        Update the Kalman filter with a new measurement.
        
        Args:
            box: New face bounding box measurement
            
        Returns:
            Corrected face box
        """
        # Extract center and size for measurement
        cx = (box.x1 + box.x2) / 2
        cy = (box.y1 + box.y2) / 2
        width = box.width
        height = box.height
        
        # Create measurement vector
        z = np.array([
            [cx], [cy], [width], [height]
        ], dtype=np.float32)
        
        # If we haven't predicted yet, just use the measurement
        if not hasattr(self.kalman, 'x_pred'):
            self.kalman['x_pred'] = self.kalman['x']
            self.kalman['P_pred'] = self.kalman['P']
        
        # Calculate Kalman gain
        K = self.kalman['P_pred'] @ self.kalman['H'].T @ np.linalg.inv(
            self.kalman['H'] @ self.kalman['P_pred'] @ self.kalman['H'].T + self.kalman['R']
        )
        
        # Update state with measurement
        self.kalman['x'] = self.kalman['x_pred'] + K @ (z - self.kalman['H'] @ self.kalman['x_pred'])
        
        # Update error covariance
        self.kalman['P'] = (self.kalman['I'] - K @ self.kalman['H']) @ self.kalman['P_pred']
        
        # Convert corrected state to bounding box
        cx, cy, width, height = self.kalman['x'][0, 0], self.kalman['x'][1, 0], self.kalman['x'][2, 0], self.kalman['x'][3, 0]
        
        # Calculate bounding box coordinates
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2
        
        # Create and return face box (with original confidence)
        return FaceBox(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            confidence=box.confidence,
            landmarks=box.landmarks,
            face_id=box.face_id,
            embedding=box.embedding,
            model=box.model
        )
    
    def get_velocity(self) -> Tuple[float, float, float, float]:
        """
        Get the current velocity estimates.
        
        Returns:
            Tuple of (vx, vy, vw, vh) - velocities of position and size
        """
        vx = self.kalman['x'][4, 0]
        vy = self.kalman['x'][5, 0]
        vw = self.kalman['x'][6, 0]
        vh = self.kalman['x'][7, 0]
        
        return float(vx), float(vy), float(vw), float(vh)
    
    def reset(self) -> None:
        """Reset the Kalman filter."""
        self._init_kalman()


class FaceTrackingFilter:
    """
    Manages Kalman filters for multiple face tracks.
    
    This class handles creating, updating, and managing Kalman filters
    for multiple faces being tracked simultaneously.
    """
    
    def __init__(self):
        """Initialize the face tracking filter manager."""
        self.filters = {}  # Dict[face_id, KalmanFilter]
        self.last_seen = {}  # Dict[face_id, frame_idx]
        self.max_idle_frames = 30  # Maximum frames a face can be missing
        
        logger.info("Initialized face tracking filter manager")
    
    def update(self, face_id: int, box: FaceBox, frame_idx: int) -> FaceBox:
        """
        Update the filter for a specific face.
        
        Args:
            face_id: Face ID
            box: Face bounding box
            frame_idx: Current frame index
            
        Returns:
            Filtered face box
        """
        # Create filter if it doesn't exist
        if face_id not in self.filters:
            self.filters[face_id] = KalmanFilter()
            self.filters[face_id].init(box)
            filtered_box = box  # First frame, just use the detection
        else:
            # Get filter and update
            kalman = self.filters[face_id]
            kalman.predict()  # Predict first
            filtered_box = kalman.update(box)  # Then update with measurement
        
        # Update last seen frame
        self.last_seen[face_id] = frame_idx
        
        return filtered_box
    
    def predict(self, face_id: int) -> Optional[FaceBox]:
        """
        Predict the position of a face without measurement.
        
        Args:
            face_id: Face ID
            
        Returns:
            Predicted face box or None if filter doesn't exist
        """
        if face_id not in self.filters:
            return None
        
        return self.filters[face_id].predict()
    
    def cleanup(self, current_frame_idx: int) -> List[int]:
        """
        Remove filters for faces that haven't been seen for too long.
        
        Args:
            current_frame_idx: Current frame index
            
        Returns:
            List of removed face IDs
        """
        removed_ids = []
        
        for face_id, last_frame in list(self.last_seen.items()):
            if current_frame_idx - last_frame > self.max_idle_frames:
                # Remove this face
                self.filters.pop(face_id, None)
                self.last_seen.pop(face_id, None)
                removed_ids.append(face_id)
                
                logger.debug(f"Removed tracking filter for face {face_id} (not seen for {current_frame_idx - last_frame} frames)")
        
        return removed_ids
    
    def get_all_predictions(self) -> Dict[int, FaceBox]:
        """
        Get predictions for all currently tracked faces.
        
        Returns:
            Dictionary of face_id to predicted face box
        """
        predictions = {}
        
        for face_id, kalman in self.filters.items():
            predictions[face_id] = kalman.predict()
        
        return predictions
    
    def reset(self) -> None:
        """Reset all filters."""
        self.filters.clear()
        self.last_seen.clear()
        logger.info("Reset all face tracking filters") 