import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import time

from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    get_device,
    ensure_directory
)

class PoseEstimator:
    """
    Class for estimating body pose from images or videos.
    Supports both 2D and 3D pose estimation.
    """
    
    def __init__(self, 
                model_path: Optional[str] = None,
                use_3d: bool = True,
                use_temporal_smoothing: bool = True,
                confidence_threshold: float = 0.3):
        """
        Initialize the pose estimator.
        
        Args:
            model_path: Path to pose estimation model
            use_3d: Whether to estimate 3D pose
            use_temporal_smoothing: Whether to smooth pose sequences
            confidence_threshold: Minimum confidence for keypoints
        """
        self.device = get_device()
        self.model_path = model_path
        self.use_3d = use_3d
        self.use_temporal_smoothing = use_temporal_smoothing
        self.confidence_threshold = confidence_threshold
        
        # Initialize models
        self.pose_model, self.model_type = self._initialize_pose_model()
        
        # Initialize MediaPipe (if available) as a fallback
        self.mp_pose = self._initialize_mediapipe()
        
        # Define keypoint names for reference
        self.keypoint_names = [
            'nose',
            'neck',
            'right_shoulder',
            'right_elbow',
            'right_wrist',
            'left_shoulder',
            'left_elbow',
            'left_wrist',
            'right_hip',
            'right_knee',
            'right_ankle',
            'left_hip',
            'left_knee',
            'left_ankle',
            'right_eye',
            'left_eye',
            'right_ear',
            'left_ear'
        ]
        
        print(f"Initialized PoseEstimator (3D={use_3d}, model_type={self.model_type})")
    
    def _initialize_pose_model(self) -> Tuple[Any, str]:
        """
        Initialize the pose estimation model.
        
        Returns:
            Tuple of (model, model_type)
        """
        # Try to load a pre-trained model if available
        model = None
        model_type = "none"
        
        try:
            # First try to use pytorch models if available
            try:
                import torch
                import torchvision
                
                # Try to load detectron2 first
                try:
                    from detectron2.config import get_cfg
                    from detectron2.engine import DefaultPredictor
                    
                    # This is a placeholder for Detectron2 configuration
                    # In a real implementation, you would load proper config
                    cfg = get_cfg()
                    # Set config for keypoint detection
                    
                    if self.model_path and os.path.exists(self.model_path):
                        cfg.MODEL.WEIGHTS = self.model_path
                    
                    # Create predictor
                    model = DefaultPredictor(cfg)
                    model_type = "detectron2"
                    print("Using Detectron2 for pose estimation")
                    return model, model_type
                    
                except (ImportError, ModuleNotFoundError):
                    # If detectron2 fails, try HRNet
                    try:
                        # Placeholder for HRNet or similar model
                        print("Detectron2 not available, trying other models")
                        
                        # Try to use torchvision's keypoint detection
                        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
                        model.to(self.device)
                        model.eval()
                        model_type = "torchvision"
                        print("Using torchvision KeypointRCNN for pose estimation")
                        return model, model_type
                        
                    except Exception as e:
                        print(f"Error loading HRNet: {e}")
            
            except (ImportError, ModuleNotFoundError):
                print("PyTorch vision models not available")
            
            # If PyTorch models failed, try OpenCV models
            try:
                # Try OpenCV DNN
                from cv2.dnn import readNetFromTensorflow
                
                # Load model
                model_path = self.model_path
                if model_path is None or not os.path.exists(model_path):
                    # Use a default model if provided path doesn't exist
                    print("Model path not found, using OpenCV default")
                    
                    # In a real app, you would include these models or download them
                    model = cv2.dnn.readNetFromTensorflow("path/to/opencv/pose/model")
                else:
                    model = cv2.dnn.readNetFromTensorflow(model_path)
                
                model_type = "opencv"
                print("Using OpenCV DNN for pose estimation")
                return model, model_type
                
            except Exception as e:
                print(f"Error loading OpenCV model: {e}")
        
        except Exception as e:
            print(f"Error initializing pose models: {e}")
        
        # Return default
        return None, "none"
    
    def _initialize_mediapipe(self):
        """
        Initialize MediaPipe Pose as a fallback.
        
        Returns:
            MediaPipe Pose object or None if not available
        """
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            return mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2 if self.use_3d else 1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except ImportError:
            print("MediaPipe not available for fallback pose detection")
            return None
    
    def estimate_pose_from_image(self, image_path: str) -> Dict:
        """
        Estimate pose from a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing pose data
        """
        print(f"Estimating pose from image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        
        # Check if image is loaded
        if image is None:
            print(f"Failed to load image: {image_path}")
            return {'success': False, 'error': 'Failed to load image'}
        
        # Estimate pose using available models
        if self.model_type == "detectron2":
            pose_data = self._estimate_pose_detectron2(image)
        elif self.model_type == "torchvision":
            pose_data = self._estimate_pose_torchvision(image)
        elif self.model_type == "opencv":
            pose_data = self._estimate_pose_opencv(image)
        else:
            # Fallback to MediaPipe
            pose_data = self._estimate_pose_mediapipe(image)
        
        # If 3D pose estimation is enabled and we have 2D keypoints
        if self.use_3d and 'keypoints_2d' in pose_data and pose_data['success']:
            # Add 3D keypoints
            pose_data = self._estimate_3d_from_2d(pose_data)
        
        # Add image dimensions
        if pose_data['success']:
            pose_data['image_height'], pose_data['image_width'] = image.shape[:2]
        
        return pose_data
    
    def estimate_pose_from_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Estimate pose from a video.
        
        Args:
            video_path: Path to the input video
            output_dir: Optional directory to save visualization
            
        Returns:
            Dictionary containing pose data for each frame
        """
        print(f"Estimating pose from video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Check if video is opened
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return {'success': False, 'error': 'Failed to open video'}
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {fps} fps, {frame_count} frames")
        
        # Process frames
        all_poses = []
        frame_idx = 0
        
        # Setup video writer if output directory is provided
        video_writer = None
        if output_dir:
            ensure_directory(output_dir)
            output_video_path = os.path.join(output_dir, os.path.basename(video_path))
            # Ensure output path has .mp4 extension
            if not output_video_path.lower().endswith('.mp4'):
                output_video_path = os.path.splitext(output_video_path)[0] + '.mp4'
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Writing output video to: {output_video_path}")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                pose_data = self._estimate_pose_from_frame(frame)
                
                # Add frame index
                pose_data['frame'] = frame_idx
                
                # Add to list of poses
                all_poses.append(pose_data)
                
                # Write visualization if output directory is provided
                if video_writer is not None:
                    if pose_data['success']:
                        vis_frame = self.draw_pose_on_image(frame, pose_data)
                        video_writer.write(vis_frame)
                    else:
                        video_writer.write(frame)
                
                # Increment frame index
                frame_idx += 1
                
                # Print progress
                if frame_idx % 10 == 0:
                    print(f"Processed {frame_idx}/{frame_count} frames")
        
        except Exception as e:
            print(f"Error processing video: {e}")
        
        # Release resources
        cap.release()
        if video_writer is not None:
            video_writer.release()
        
        # Apply temporal smoothing if enabled
        if self.use_temporal_smoothing:
            all_poses = self._apply_temporal_smoothing(all_poses)
        
        # Create final result dictionary
        result = {
            'success': True,
            'video_path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'poses': all_poses
        }
        
        # If 3D poses are requested, convert 2D sequences to 3D
        if self.use_3d:
            result['poses_3d'] = self._estimate_3d_sequence(all_poses)
        
        return result
    
    def _estimate_pose_from_frame(self, frame: np.ndarray) -> Dict:
        """
        Estimate pose from a single video frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing pose data
        """
        # Estimate pose using available models
        if self.model_type == "detectron2":
            pose_data = self._estimate_pose_detectron2(frame)
        elif self.model_type == "torchvision":
            pose_data = self._estimate_pose_torchvision(frame)
        elif self.model_type == "opencv":
            pose_data = self._estimate_pose_opencv(frame)
        else:
            # Fallback to MediaPipe
            pose_data = self._estimate_pose_mediapipe(frame)
        
        # If 3D pose estimation is enabled and we have 2D keypoints
        if self.use_3d and 'keypoints_2d' in pose_data and pose_data['success']:
            # Add 3D keypoints
            pose_data = self._estimate_3d_from_2d(pose_data)
        
        # Add image dimensions
        if pose_data['success']:
            pose_data['image_height'], pose_data['image_width'] = frame.shape[:2]
        
        return pose_data
    
    def _estimate_pose_detectron2(self, image: np.ndarray) -> Dict:
        """
        Estimate pose using Detectron2.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing pose data
        """
        # This is a placeholder for Detectron2 implementation
        # In a real implementation, you would process the image
        # and extract keypoints from the results
        
        print("Detectron2 pose estimation not fully implemented")
        
        # Return empty result
        return {'success': False, 'error': 'Detectron2 implementation incomplete'}
    
    def _estimate_pose_torchvision(self, image: np.ndarray) -> Dict:
        """
        Estimate pose using torchvision's KeypointRCNN.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing pose data
        """
        try:
            # Convert image to RGB if it's BGR
            if image.shape[2] == 3 and image.dtype == np.uint8:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Convert image to tensor
            tensor_image = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.pose_model(tensor_image)
            
            # Check if any person was detected
            if len(outputs[0]['boxes']) == 0:
                return {'success': False, 'error': 'No person detected'}
            
            # Get the person with highest score
            scores = outputs[0]['scores'].cpu().numpy()
            if max(scores) < self.confidence_threshold:
                return {'success': False, 'error': 'Low confidence detection'}
            
            # Get best person index
            best_idx = np.argmax(scores)
            
            # Get keypoints
            keypoints = outputs[0]['keypoints'][best_idx].cpu().numpy()
            
            # Convert to desired format [x, y, confidence]
            keypoints_2d = keypoints[:, :2]  # xy coordinates
            confidences = keypoints[:, 2]    # confidence values
            
            # Filter low-confidence keypoints
            keypoints_2d[confidences < self.confidence_threshold] = np.nan
            
            return {
                'success': True,
                'keypoints_2d': keypoints_2d,
                'confidences': confidences,
                'bbox': outputs[0]['boxes'][best_idx].cpu().numpy()
            }
            
        except Exception as e:
            print(f"Error in torchvision pose estimation: {e}")
            return {'success': False, 'error': str(e)}
    
    def _estimate_pose_opencv(self, image: np.ndarray) -> Dict:
        """
        Estimate pose using OpenCV DNN.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing pose data
        """
        # This is a placeholder for OpenCV DNN implementation
        # In a real implementation, you would process the image
        # and extract keypoints from the results
        
        print("OpenCV pose estimation not fully implemented")
        
        # Return empty result
        return {'success': False, 'error': 'OpenCV implementation incomplete'}
    
    def _estimate_pose_mediapipe(self, image: np.ndarray) -> Dict:
        """
        Estimate pose using MediaPipe.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing pose data
        """
        # Check if MediaPipe is available
        if self.mp_pose is None:
            return {'success': False, 'error': 'MediaPipe not available'}
        
        try:
            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.mp_pose.process(image_rgb)
            
            # Check if pose was detected
            if not results.pose_landmarks:
                return {'success': False, 'error': 'No pose detected'}
            
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Convert to numpy array
            h, w, _ = image.shape
            keypoints_2d = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
            confidences = np.array([lm.visibility for lm in landmarks])
            
            # Map MediaPipe landmarks to our keypoint format
            # MediaPipe has 33 landmarks, we need to map to our format
            
            # Define indices mapping (MediaPipe indices -> our indices)
            # This is a simplified mapping, adjust as needed
            mapping = {
                0: 0,  # nose
                11: 1,  # neck (approx)
                12: 2,  # right shoulder
                14: 3,  # right elbow
                16: 4,  # right wrist
                11: 5,  # left shoulder
                13: 6,  # left elbow
                15: 7,  # left wrist
                24: 8,  # right hip
                26: 9,  # right knee
                28: 10,  # right ankle
                23: 11,  # left hip
                25: 12,  # left knee
                27: 13,  # left ankle
                5: 14,  # right eye
                2: 15,  # left eye
                8: 16,  # right ear
                7: 17   # left ear
            }
            
            # Map keypoints
            mapped_keypoints = np.zeros((len(self.keypoint_names), 2))
            mapped_confidences = np.zeros(len(self.keypoint_names))
            
            for mp_idx, our_idx in mapping.items():
                if mp_idx < len(keypoints_2d):
                    mapped_keypoints[our_idx] = keypoints_2d[mp_idx]
                    mapped_confidences[our_idx] = confidences[mp_idx]
            
            # Filter low-confidence keypoints
            mapped_keypoints[mapped_confidences < self.confidence_threshold] = np.nan
            
            return {
                'success': True,
                'keypoints_2d': mapped_keypoints,
                'confidences': mapped_confidences,
                'raw_landmarks': landmarks
            }
            
        except Exception as e:
            print(f"Error in MediaPipe pose estimation: {e}")
            return {'success': False, 'error': str(e)}
    
    def _estimate_3d_from_2d(self, pose_data: Dict) -> Dict:
        """
        Estimate 3D pose from 2D keypoints.
        
        Args:
            pose_data: Dictionary containing 2D pose data
            
        Returns:
            Dictionary with added 3D pose data
        """
        # This is a simplified implementation
        # In a real implementation, you would use a proper 3D lifting model
        
        # Check if we have 2D keypoints
        if 'keypoints_2d' not in pose_data or not pose_data['success']:
            return pose_data
        
        try:
            # Get 2D keypoints
            keypoints_2d = pose_data['keypoints_2d']
            
            # Create placeholder 3D keypoints
            # In a real implementation, this would be a proper 2D-to-3D lifting
            # For now, just set Z based on typical human proportions
            
            # Add artificial Z coordinates
            keypoints_3d = np.zeros((keypoints_2d.shape[0], 3))
            keypoints_3d[:, :2] = keypoints_2d  # Copy X, Y
            
            # Simplified depth approximation
            # In a real implementation, this would use a trained model
            
            # Torso (shoulders, hips)
            if not np.isnan(keypoints_2d[2, 0]) and not np.isnan(keypoints_2d[5, 0]):
                # Shoulders
                shoulder_width = np.linalg.norm(keypoints_2d[2] - keypoints_2d[5])
                # Use shoulder width to scale Z values
                scale = shoulder_width / 100.0
                
                # Set torso depths
                for idx in [1, 2, 5, 8, 11]:  # neck, shoulders, hips
                    if idx < len(keypoints_3d) and not np.isnan(keypoints_2d[idx, 0]):
                        keypoints_3d[idx, 2] = 0  # reference plane
                
                # Arms and legs (approximate)
                # Right arm
                for idx in [3, 4]:  # right elbow, wrist
                    if idx < len(keypoints_3d) and not np.isnan(keypoints_2d[idx, 0]):
                        keypoints_3d[idx, 2] = scale * 20  # slightly forward
                
                # Left arm
                for idx in [6, 7]:  # left elbow, wrist
                    if idx < len(keypoints_3d) and not np.isnan(keypoints_2d[idx, 0]):
                        keypoints_3d[idx, 2] = scale * 20  # slightly forward
                
                # Face
                for idx in [0, 14, 15, 16, 17]:  # nose, eyes, ears
                    if idx < len(keypoints_3d) and not np.isnan(keypoints_2d[idx, 0]):
                        keypoints_3d[idx, 2] = scale * -30  # forward from body
                
                # Legs
                for idx in [9, 10, 12, 13]:  # knees, ankles
                    if idx < len(keypoints_3d) and not np.isnan(keypoints_2d[idx, 0]):
                        keypoints_3d[idx, 2] = scale * 10  # slightly forward
            
            # Add 3D keypoints to pose data
            pose_data['keypoints_3d'] = keypoints_3d
            
        except Exception as e:
            print(f"Error estimating 3D pose: {e}")
        
        return pose_data
    
    def _estimate_3d_sequence(self, poses: List[Dict]) -> List:
        """
        Estimate 3D pose sequence from a list of 2D poses.
        
        Args:
            poses: List of pose data dictionaries
            
        Returns:
            List of 3D pose data
        """
        # Process each pose in the sequence
        poses_3d = []
        
        for pose in poses:
            # Skip frames with no detection
            if not pose.get('success', False):
                poses_3d.append(None)
                continue
            
            # Estimate 3D from 2D
            pose_3d = self._estimate_3d_from_2d(pose)
            poses_3d.append(pose_3d.get('keypoints_3d', None))
        
        return poses_3d
    
    def _apply_temporal_smoothing(self, poses: List[Dict], window_size: int = 5) -> List[Dict]:
        """
        Apply temporal smoothing to a sequence of poses.
        
        Args:
            poses: List of pose data dictionaries
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed poses
        """
        # Check if we have enough poses to smooth
        if len(poses) < 3:
            return poses
        
        # Create smoothed poses
        smoothed_poses = poses.copy()
        
        # Get valid poses (with successful detection)
        valid_indices = [i for i, pose in enumerate(poses) if pose.get('success', False)]
        
        if len(valid_indices) < 3:
            return poses
        
        try:
            # For each valid pose
            for key in ['keypoints_2d', 'keypoints_3d']:
                # Skip if key doesn't exist in poses
                if key not in poses[valid_indices[0]]:
                    continue
                
                # Get keypoints array from valid poses
                keypoints_list = [poses[i][key] for i in valid_indices if key in poses[i]]
                
                if not keypoints_list:
                    continue
                
                # Stack into a 3D array (frame, keypoint, dimension)
                keypoints_array = np.stack(keypoints_list)
                
                # Apply smoothing
                smoothed_array = self._smooth_array(keypoints_array, window_size)
                
                # Update poses with smoothed keypoints
                for idx, orig_idx in enumerate(valid_indices):
                    if idx < len(smoothed_array):
                        smoothed_poses[orig_idx][key] = smoothed_array[idx]
        
        except Exception as e:
            print(f"Error in temporal smoothing: {e}")
        
        return smoothed_poses
    
    def _smooth_array(self, array: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Smooth a 3D array using a moving average.
        
        Args:
            array: Input array with shape (frames, keypoints, dimensions)
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed array
        """
        # Get array shape
        n_frames, n_keypoints, n_dims = array.shape
        
        # Create output array
        smoothed = np.copy(array)
        
        # Handle NaN values (missing keypoints)
        mask = np.isnan(array)
        
        # For each keypoint and dimension
        for k in range(n_keypoints):
            for d in range(n_dims):
                # Get values for this keypoint and dimension
                values = array[:, k, d]
                
                # Skip if all values are NaN
                if np.all(np.isnan(values)):
                    continue
                
                # Create a copy with NaN values replaced by closest valid value
                filled = np.copy(values)
                
                # Forward fill
                for i in range(1, n_frames):
                    if np.isnan(filled[i]):
                        filled[i] = filled[i-1]
                
                # Backward fill (for any remaining NaNs at the beginning)
                for i in range(n_frames-2, -1, -1):
                    if np.isnan(filled[i]):
                        filled[i] = filled[i+1]
                
                # Apply moving average
                for i in range(n_frames):
                    # Get window bounds
                    start = max(0, i - window_size // 2)
                    end = min(n_frames, i + window_size // 2 + 1)
                    
                    # Get window values
                    window = filled[start:end]
                    
                    # Calculate weighted average (higher weight for center)
                    weights = np.ones(len(window))
                    mid = len(window) // 2
                    weights[mid] = 2.0  # Higher weight for the current frame
                    
                    # Set smoothed value
                    smoothed[i, k, d] = np.average(window, weights=weights)
                
                # Restore original NaN values if all window values were NaN
                smoothed[:, k, d] = np.where(np.all(mask[:, k, d:d+1], axis=1), np.nan, smoothed[:, k, d])
        
        return smoothed
    
    def convert_pose_to_smpl(self, pose_data: Dict) -> Dict:
        """
        Convert pose data to SMPL parameters.
        
        Args:
            pose_data: Dictionary containing pose data
            
        Returns:
            Dictionary with SMPL parameters
        """
        # This is a placeholder for SMPL parameter estimation
        # In a real implementation, you would use a proper regressor
        
        # Check if we have 3D keypoints
        if 'keypoints_3d' not in pose_data or not pose_data['success']:
            return {'success': False, 'error': 'No 3D keypoints available'}
        
        # Create placeholder SMPL parameters
        smpl_params = {
            'success': True,
            'global_orient': np.zeros(3),  # Global rotation
            'body_pose': np.zeros(69),     # Joint rotations (23 joints x 3)
            'betas': np.zeros(10)          # Shape parameters
        }
        
        # In a real implementation, you would estimate these parameters
        # from the 3D keypoints using a trained regressor
        
        return smpl_params
    
    def draw_pose_on_image(self, image: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw pose keypoints and skeleton on an image.
        
        Args:
            image: Input image
            pose_data: Dictionary containing pose data
            
        Returns:
            Image with pose visualization
        """
        # Check if pose was detected
        if not pose_data.get('success', False) or 'keypoints_2d' not in pose_data:
            return image
        
        # Make a copy of the image
        vis_image = image.copy()
        
        # Get keypoints
        keypoints = pose_data['keypoints_2d']
        confidences = pose_data.get('confidences', np.ones(len(keypoints)))
        
        # Define skeleton connections
        skeleton = [
            (0, 1),    # nose to neck
            (1, 2),    # neck to right shoulder
            (1, 5),    # neck to left shoulder
            (2, 3),    # right shoulder to right elbow
            (3, 4),    # right elbow to right wrist
            (5, 6),    # left shoulder to left elbow
            (6, 7),    # left elbow to left wrist
            (1, 8),    # neck to right hip
            (1, 11),   # neck to left hip
            (8, 9),    # right hip to right knee
            (9, 10),   # right knee to right ankle
            (11, 12),  # left hip to left knee
            (12, 13),  # left knee to left ankle
            (0, 14),   # nose to right eye
            (0, 15),   # nose to left eye
            (14, 16),  # right eye to right ear
            (15, 17)   # left eye to left ear
        ]
        
        # Draw skeleton
        for start_idx, end_idx in skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                not np.isnan(keypoints[start_idx, 0]) and not np.isnan(keypoints[end_idx, 0])):
                # Get confidence
                conf = min(confidences[start_idx], confidences[end_idx])
                
                # Skip low confidence connections
                if conf < self.confidence_threshold:
                    continue
                
                # Get points
                start_point = tuple(map(int, keypoints[start_idx]))
                end_point = tuple(map(int, keypoints[end_idx]))
                
                # Draw line
                cv2.line(vis_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if not np.isnan(x) and not np.isnan(y) and confidences[i] >= self.confidence_threshold:
                # Set color based on confidence
                color = (0, int(255 * confidences[i]), 255)
                
                # Draw circle
                cv2.circle(vis_image, (int(x), int(y)), 5, color, -1)
                
                # Add keypoint index
                cv2.putText(vis_image, str(i), (int(x) + 10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return vis_image
    
    def save_pose_visualization(self, image_path: str, pose_data: Dict, output_path: str) -> str:
        """
        Save pose visualization to a file.
        
        Args:
            image_path: Path to the input image
            pose_data: Dictionary containing pose data
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        # Load image
        image = load_image(image_path)
        
        # Check if image is loaded
        if image is None:
            print(f"Failed to load image: {image_path}")
            return ""
        
        # Draw pose on image
        vis_image = self.draw_pose_on_image(image, pose_data)
        
        # Create output directory if needed
        ensure_directory(os.path.dirname(output_path))
        
        # Save visualization
        save_image(output_path, vis_image)
        
        print(f"Saved pose visualization to: {output_path}")
        return output_path
    
    def save_pose_to_json(self, pose_data: Dict, output_path: str) -> str:
        """
        Save pose data to a JSON file.
        
        Args:
            pose_data: Dictionary containing pose data
            output_path: Path to save the JSON file
            
        Returns:
            Path to the saved file
        """
        # Create a copy of the pose data
        json_data = pose_data.copy()
        
        # Convert numpy arrays to lists
        for key, value in json_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
        
        # Handle nested dictionaries
        for key, value in list(json_data.items()):
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_data[key][k] = v.tolist()
        
        # Handle lists of dictionaries (e.g., for video poses)
        if 'poses' in json_data:
            for i, pose in enumerate(json_data['poses']):
                for key, value in pose.items():
                    if isinstance(value, np.ndarray):
                        json_data['poses'][i][key] = value.tolist()
        
        # Create output directory if needed
        ensure_directory(os.path.dirname(output_path))
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved pose data to: {output_path}")
        return output_path 