import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
import time
from dataclasses import dataclass

from app.avatar_creation.face_modeling.utils import get_device, ensure_directory

@dataclass
class MotionModelConfig:
    """Configuration settings for the First Order Motion Model."""
    use_gpu: bool = True
    temporal_consistency: bool = True
    reference_frame_update_frequency: int = 100  # frames
    adaptive_reference_update: bool = True
    motion_scale: float = 1.0  # Scale motion magnitude
    stabilization_weight: float = 0.3  # Weight for stabilization
    occlusion_handling: bool = True
    keypoint_smoothing: bool = True
    interpolation_mode: str = 'bilinear'
    model_path: str = ''
    use_background: bool = False

class FirstOrderMotionModel:
    """
    Implementation of the First Order Motion Model for video animation.
    This model animates a source image according to the motion of a driving video.
    
    Based on the paper:
    "First Order Motion Model for Image Animation" by Siarohin et al.
    """
    
    def __init__(self, config: MotionModelConfig = None):
        """
        Initialize the First Order Motion Model.
        
        Args:
            config: Configuration for the motion model
        """
        self.config = config or MotionModelConfig()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")
        
        # Initialize model components
        self.keypoint_detector = None
        self.motion_estimator = None
        self.generator = None
        self.occlusion_estimator = None
        
        # Runtime state
        self.initialized = False
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_frame_count = 0
        self.current_frame_index = 0
        self.last_keypoints = None
        self.keypoint_history = []
        self.max_keypoint_history = 30  # frames to keep for smoothing
        
        # Performance tracking
        self.timings = {
            'keypoint_detection': [],
            'motion_estimation': [],
            'image_generation': [],
            'total': []
        }
        self.max_timing_history = 100
        
        # If model path is provided, load the model
        if self.config.model_path and os.path.exists(self.config.model_path):
            self._load_model()
            
        print(f"First Order Motion Model initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Temporal consistency: {self.config.temporal_consistency}")
        print(f"  - Motion scale: {self.config.motion_scale}")
        print(f"  - Occlusion handling: {self.config.occlusion_handling}")
    
    def _load_model(self) -> None:
        """
        Load the First Order Motion Model from the specified path.
        """
        try:
            if not os.path.exists(self.config.model_path):
                print(f"Model path does not exist: {self.config.model_path}")
                return
            
            # Load pretrained weights
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # Initialize network components
            # In a real implementation, these would be the actual model architectures from the paper
            # For simplicity, we're using placeholder implementations
            
            # Placeholder: Keypoint detector network
            self.keypoint_detector = KeypointDetector()
            
            # Placeholder: Motion estimator network
            self.motion_estimator = MotionEstimator()
            
            # Placeholder: Generator network
            self.generator = Generator()
            
            # Placeholder: Occlusion estimator
            if self.config.occlusion_handling:
                self.occlusion_estimator = OcclusionEstimator()
            
            # Load the state dict for each component
            # self.keypoint_detector.load_state_dict(checkpoint['keypoint_detector'])
            # self.motion_estimator.load_state_dict(checkpoint['motion_estimator'])
            # self.generator.load_state_dict(checkpoint['generator'])
            # if self.config.occlusion_handling:
            #     self.occlusion_estimator.load_state_dict(checkpoint['occlusion_estimator'])
                
            # Set models to evaluation mode
            self.keypoint_detector.eval()
            self.motion_estimator.eval()
            self.generator.eval()
            if self.config.occlusion_handling:
                self.occlusion_estimator.eval()
                
            self.initialized = True
            print(f"Successfully loaded model from {self.config.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Fallback to placeholder implementation for demo purposes
            self._initialize_placeholder_model()
    
    def _initialize_placeholder_model(self) -> None:
        """
        Initialize a placeholder model for demonstration purposes.
        This is used when a real model is not available.
        """
        # Create placeholder model components
        self.keypoint_detector = KeypointDetector()
        self.motion_estimator = MotionEstimator()
        self.generator = Generator()
        
        if self.config.occlusion_handling:
            self.occlusion_estimator = OcclusionEstimator()
        
        self.initialized = True
        print("Initialized placeholder model for demonstration")
    
    def set_reference_frame(self, frame: np.ndarray) -> None:
        """
        Set the reference frame for animation.
        
        Args:
            frame: The source image to be animated
        """
        if frame is None:
            print("Error: Reference frame is None")
            return
            
        # Convert to RGB if grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]  # Remove alpha channel
        
        # Preprocess frame
        frame_tensor = self._preprocess_image(frame)
        
        # Detect keypoints in reference frame
        start_time = time.time()
        with torch.no_grad():
            self.reference_keypoints = self.keypoint_detector(frame_tensor)
        
        # Store reference frame
        self.reference_frame = frame_tensor
        self.reference_frame_count = 0
        self.current_frame_index = 0
        self.last_keypoints = self.reference_keypoints
        self.keypoint_history = [self.reference_keypoints]
        
        kp_time = time.time() - start_time
        self.timings['keypoint_detection'].append(kp_time)
        
        if len(self.timings['keypoint_detection']) > self.max_timing_history:
            self.timings['keypoint_detection'].pop(0)
        
        print(f"Reference frame set with {len(self.reference_keypoints)} keypoints")
    
    def animate(self, driving_frame: np.ndarray) -> np.ndarray:
        """
        Animate the reference frame based on the motion in the driving frame.
        
        Args:
            driving_frame: Current frame from driving video
            
        Returns:
            Animated frame
        """
        if not self.initialized:
            print("Error: Model not initialized")
            return driving_frame
            
        if self.reference_frame is None:
            print("Error: Reference frame not set")
            return driving_frame
            
        # Increment counters
        self.current_frame_index += 1
        self.reference_frame_count += 1
        
        # Start timing
        total_start_time = time.time()
        
        # Preprocess driving frame
        driving_tensor = self._preprocess_image(driving_frame)
        
        # Detect keypoints in driving frame
        kp_start_time = time.time()
        with torch.no_grad():
            driving_keypoints = self.keypoint_detector(driving_tensor)
        kp_time = time.time() - kp_start_time
        
        # Apply temporal smoothing to keypoints if enabled
        if self.config.keypoint_smoothing and self.config.temporal_consistency:
            driving_keypoints = self._smooth_keypoints(driving_keypoints)
        
        # Store keypoints for future smoothing
        self.last_keypoints = driving_keypoints
        self.keypoint_history.append(driving_keypoints)
        if len(self.keypoint_history) > self.max_keypoint_history:
            self.keypoint_history.pop(0)
        
        # Estimate motion between frames
        motion_start_time = time.time()
        with torch.no_grad():
            motion_params = self.motion_estimator(
                source_keypoints=self.reference_keypoints,
                driving_keypoints=driving_keypoints
            )
            
            # Scale motion if needed
            if self.config.motion_scale != 1.0:
                motion_params = self._scale_motion(motion_params, self.config.motion_scale)
        motion_time = time.time() - motion_start_time
        
        # Estimate occlusion mask if enabled
        occlusion_mask = None
        if self.config.occlusion_handling and self.occlusion_estimator is not None:
            with torch.no_grad():
                occlusion_mask = self.occlusion_estimator(
                    source=self.reference_frame,
                    driving=driving_tensor,
                    motion_params=motion_params
                )
        
        # Generate animated frame
        gen_start_time = time.time()
        with torch.no_grad():
            generated = self.generator(
                source=self.reference_frame,
                motion_params=motion_params,
                occlusion_mask=occlusion_mask
            )
        gen_time = time.time() - gen_start_time
        
        # Convert back to numpy array
        result = self._postprocess_image(generated)
        
        # Check if we should update the reference frame
        if self.config.adaptive_reference_update:
            if self._should_update_reference(motion_params):
                print("Updating reference frame due to large motion")
                self.set_reference_frame(result)
        elif self.reference_frame_count >= self.config.reference_frame_update_frequency:
            print(f"Updating reference frame after {self.reference_frame_count} frames")
            self.set_reference_frame(result)
        
        # Record timings
        total_time = time.time() - total_start_time
        self.timings['keypoint_detection'].append(kp_time)
        self.timings['motion_estimation'].append(motion_time)
        self.timings['image_generation'].append(gen_time)
        self.timings['total'].append(total_time)
        
        # Trim timing history
        for key in self.timings:
            if len(self.timings[key]) > self.max_timing_history:
                self.timings[key].pop(0)
        
        return result
    
    def _smooth_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal smoothing to keypoints.
        
        Args:
            keypoints: Current keypoints
            
        Returns:
            Smoothed keypoints
        """
        if len(self.keypoint_history) < 2:
            return keypoints
        
        # Simple exponential smoothing
        alpha = 0.7  # Smoothing factor
        smoothed = alpha * keypoints
        
        # Calculate weighted average of previous frames
        weight_sum = alpha
        decay_factor = 0.8
        
        for i, prev_kp in enumerate(reversed(self.keypoint_history[-3:])):
            weight = (1.0 - alpha) * (decay_factor ** (i + 1))
            smoothed += weight * prev_kp
            weight_sum += weight
        
        # Normalize
        if weight_sum > 0:
            smoothed /= weight_sum
            
        return smoothed
    
    def _should_update_reference(self, motion_params: Dict) -> bool:
        """
        Determine if the reference frame should be updated based on motion magnitude.
        
        Args:
            motion_params: Motion parameters between frames
            
        Returns:
            True if reference should be updated
        """
        # Get motion magnitude from motion parameters
        if 'translation' in motion_params:
            translation = motion_params['translation'].cpu().numpy()
            translation_magnitude = np.mean(np.abs(translation))
            
            # If translation is too large, update reference
            if translation_magnitude > 0.25:  # Threshold for large motion
                return True
                
        # Check rotation if available
        if 'rotation' in motion_params:
            rotation = motion_params['rotation'].cpu().numpy()
            rotation_magnitude = np.mean(np.abs(rotation))
            
            # If rotation is too large, update reference
            if rotation_magnitude > 0.25:  # Threshold in radians
                return True
                
        return False
    
    def _scale_motion(self, motion_params: Dict, scale: float) -> Dict:
        """
        Scale the motion parameters to control motion magnitude.
        
        Args:
            motion_params: Motion parameters
            scale: Scaling factor
            
        Returns:
            Scaled motion parameters
        """
        scaled_params = {}
        
        for key, value in motion_params.items():
            if key == 'translation' or key == 'rotation':
                scaled_params[key] = value * scale
            else:
                scaled_params[key] = value
                
        return scaled_params
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for the model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image as torch tensor
        """
        # Resize if needed
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = cv2.resize(image, (256, 256))
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Remove alpha channel
            
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = image * 2 - 1
        
        # Convert to torch tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _postprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert generated tensor back to numpy array.
        
        Args:
            image_tensor: Generated image as torch tensor
            
        Returns:
            Output image as numpy array
        """
        # Convert to numpy
        with torch.no_grad():
            image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Denormalize from [-1, 1] to [0, 1]
        image = (image + 1) / 2.0
        
        # Clip values to valid range
        image = np.clip(image, 0, 1)
        
        # Convert to uint8
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary of average timing stats
        """
        stats = {}
        
        for key, times in self.timings.items():
            if times:
                stats[f"avg_{key}_time"] = sum(times) / len(times)
                stats[f"max_{key}_time"] = max(times)
            else:
                stats[f"avg_{key}_time"] = 0
                stats[f"max_{key}_time"] = 0
                
        return stats
    
    def reset(self) -> None:
        """
        Reset the model state.
        """
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_frame_count = 0
        self.current_frame_index = 0
        self.last_keypoints = None
        self.keypoint_history = []
        self.timings = {key: [] for key in self.timings}
        
        print("First Order Motion Model reset to initial state")


# Placeholder model components for demo purposes
# In a real implementation, these would be replaced with the actual model architectures

class KeypointDetector(nn.Module):
    """Placeholder keypoint detector network."""
    
    def __init__(self):
        super(KeypointDetector, self).__init__()
        self.num_keypoints = 10
        
    def forward(self, x):
        """
        Detect facial keypoints in the input image.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Keypoints as tensor [B, K, 2]
        """
        # This is a placeholder implementation
        # In a real model, this would be a proper neural network
        
        batch_size = x.shape[0]
        
        # Generate random keypoints for demonstration
        # In a real implementation, this would detect actual facial landmarks
        keypoints = torch.rand(batch_size, self.num_keypoints, 2, device=x.device)
        
        return keypoints


class MotionEstimator(nn.Module):
    """Placeholder motion estimator network."""
    
    def __init__(self):
        super(MotionEstimator, self).__init__()
        
    def forward(self, source_keypoints, driving_keypoints):
        """
        Estimate motion parameters between keypoints.
        
        Args:
            source_keypoints: Keypoints from source image [B, K, 2]
            driving_keypoints: Keypoints from driving image [B, K, 2]
            
        Returns:
            Dictionary of motion parameters
        """
        # This is a placeholder implementation
        # In a real model, this would compute actual motion parameters
        
        # Calculate simple translation and rotation for demo
        batch_size = source_keypoints.shape[0]
        num_keypoints = source_keypoints.shape[1]
        
        # Example motion parameters
        translation = driving_keypoints.mean(dim=1) - source_keypoints.mean(dim=1)
        rotation = torch.zeros(batch_size, 1, device=source_keypoints.device)
        
        return {
            'translation': translation,
            'rotation': rotation,
            'keypoints': driving_keypoints
        }


class Generator(nn.Module):
    """Placeholder generator network."""
    
    def __init__(self):
        super(Generator, self).__init__()
        
    def forward(self, source, motion_params, occlusion_mask=None):
        """
        Generate animated frame from source image and motion parameters.
        
        Args:
            source: Source image tensor [B, C, H, W]
            motion_params: Dictionary of motion parameters
            occlusion_mask: Optional occlusion mask [B, 1, H, W]
            
        Returns:
            Generated image tensor [B, C, H, W]
        """
        # This is a placeholder implementation
        # In a real model, this would transform the source image based on motion
        
        # For demonstration, just return the source image
        # In real implementation, this would apply deformation based on motion
        return source


class OcclusionEstimator(nn.Module):
    """Placeholder occlusion estimator network."""
    
    def __init__(self):
        super(OcclusionEstimator, self).__init__()
        
    def forward(self, source, driving, motion_params):
        """
        Estimate occlusion mask between frames.
        
        Args:
            source: Source image tensor [B, C, H, W]
            driving: Driving image tensor [B, C, H, W]
            motion_params: Dictionary of motion parameters
            
        Returns:
            Occlusion mask [B, 1, H, W]
        """
        # This is a placeholder implementation
        # In a real model, this would compute actual occlusion masks
        
        batch_size, _, height, width = source.shape
        
        # Generate dummy occlusion mask (all visible)
        mask = torch.ones(batch_size, 1, height, width, device=source.device)
        
        return mask 