import os
import numpy as np
import torch
import cv2
import trimesh
from typing import Dict, List, Tuple, Union, Optional
import json

from app.avatar_creation.face_modeling.utils import (
    load_image,
    get_device,
    ensure_directory
)

class BodyMeasurement:
    """
    Class for extracting anthropometric measurements from images or 3D body scans.
    Supports measurement of height, weight, limb lengths, circumferences, etc.
    """
    
    def __init__(self, 
                model_path: Optional[str] = None,
                use_3d: bool = True,
                high_precision: bool = False,
                reference_height_cm: Optional[float] = None):
        """
        Initialize the body measurement module.
        
        Args:
            model_path: Path to pre-trained measurement model
            use_3d: Whether to use 3D measurement techniques
            high_precision: Whether to use high-precision algorithms
            reference_height_cm: Optional reference height in cm for scale calibration
        """
        self.device = get_device()
        self.model_path = model_path
        self.use_3d = use_3d
        self.high_precision = high_precision
        self.reference_height_cm = reference_height_cm
        
        # Initialize models
        self.measurement_model = self._initialize_measurement_model()
        
        # Define standard measurement names
        self.measurement_names = [
            'height',
            'shoulder_width',
            'chest_circumference',
            'waist_circumference',
            'hip_circumference',
            'inseam',
            'arm_length',
            'neck_circumference',
            'thigh_circumference',
            'calf_circumference',
            'bicep_circumference',
            'forearm_circumference',
            'wrist_circumference'
        ]
        
        print(f"Initialized BodyMeasurement (3D={use_3d}, high_precision={high_precision})")
    
    def _initialize_measurement_model(self):
        """
        Initialize the measurement model.
        
        Returns:
            Initialized model or None if not available
        """
        # This is a simplified implementation
        # In a real implementation, you would load a proper ML model
        
        try:
            # Load a pre-trained model if available
            if self.model_path and os.path.exists(self.model_path):
                # Load model (placeholder)
                print(f"Loading measurement model from {self.model_path}")
                return "placeholder_model"
        except Exception as e:
            print(f"Error loading measurement model: {e}")
        
        print("Using simplified measurement model")
        return None
    
    def measure_from_image(self, image_path: str, pose_data: Optional[Dict] = None) -> Dict:
        """
        Extract measurements from a single image.
        
        Args:
            image_path: Path to the input image
            pose_data: Optional pose estimation data for better accuracy
            
        Returns:
            Dictionary of measurements
        """
        print(f"Measuring body from image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        
        # Detect pose if not provided
        if pose_data is None:
            try:
                from app.avatar_creation.body_modeling.pose_estimation import PoseEstimator
                pose_estimator = PoseEstimator(use_3d=self.use_3d)
                pose_data = pose_estimator.estimate_pose_from_image(image_path)
            except Exception as e:
                print(f"Error estimating pose: {e}")
                pose_data = None
        
        # Extract measurements
        measurements = self._extract_measurements_from_image(image, pose_data)
        
        # Apply scale calibration if reference height is provided
        if self.reference_height_cm is not None and 'height' in measurements:
            scale_factor = self.reference_height_cm / measurements['height']
            for key in measurements:
                if key != 'height' and isinstance(measurements[key], (int, float)):
                    # Scale non-ratio measurements (circumferences, lengths, etc.)
                    measurements[key] *= scale_factor
        
        return measurements
    
    def measure_from_3d_mesh(self, mesh: trimesh.Trimesh, joints: Optional[np.ndarray] = None) -> Dict:
        """
        Extract measurements from a 3D mesh.
        
        Args:
            mesh: Input body mesh
            joints: Optional joint positions for better accuracy
            
        Returns:
            Dictionary of measurements
        """
        print("Measuring body from 3D mesh")
        
        # If 3D measurement is not enabled, fall back to a simplified approach
        if not self.use_3d:
            return self._extract_measurements_from_mesh_simplified(mesh)
        
        # Extract measurements
        measurements = self._extract_measurements_from_mesh(mesh, joints)
        
        return measurements
    
    def _extract_measurements_from_image(self, image: np.ndarray, pose_data: Optional[Dict]) -> Dict:
        """
        Extract measurements from an image.
        
        Args:
            image: Input image
            pose_data: Pose estimation data
            
        Returns:
            Dictionary of measurements
        """
        # This is a simplified implementation
        # In a real implementation, you would use a proper ML model
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Initialize measurements with default values
        measurements = {name: 0.0 for name in self.measurement_names}
        
        if pose_data is None or not pose_data.get('success', False):
            print("No pose data available, using rough estimation")
            
            # Very rough estimation based on image
            # In a real implementation, you would use a proper algorithm
            
            # For demonstration, set some placeholder values
            measurements['height'] = 175.0  # cm
            measurements['shoulder_width'] = 40.0  # cm
            measurements['chest_circumference'] = 90.0  # cm
            measurements['waist_circumference'] = 80.0  # cm
            measurements['hip_circumference'] = 95.0  # cm
            
            return measurements
        
        # Use pose data for better estimation
        keypoints = pose_data.get('keypoints_2d')
        
        if keypoints is None or len(keypoints) < 15:
            print("Insufficient keypoints, using rough estimation")
            return measurements
        
        # Using pose keypoints for measurements
        # For a real implementation, these would be much more sophisticated
        
        # Calculate height based on keypoints
        # Assuming keypoints: 0=nose, 10=right_ankle, 13=left_ankle
        nose_y = keypoints[0, 1]
        right_ankle_y = keypoints[10, 1]
        left_ankle_y = keypoints[13, 1]
        
        # Use average of ankles for bottom
        ankle_y = (right_ankle_y + left_ankle_y) / 2
        
        # Height in pixels
        height_px = ankle_y - nose_y
        
        # Rough conversion to cm (assuming a typical person is ~170cm from nose to ankle)
        pixel_to_cm = 170.0 / height_px
        
        # Measurements from pose keypoints
        # Shoulder width (keypoints: 2=right_shoulder, 5=left_shoulder)
        right_shoulder = keypoints[2]
        left_shoulder = keypoints[5]
        shoulder_width_px = np.linalg.norm(right_shoulder - left_shoulder)
        shoulder_width_cm = shoulder_width_px * pixel_to_cm
        
        # Rough calculations for other measurements
        # These would be much more sophisticated in a real implementation
        
        # Save measurements
        measurements['height'] = (ankle_y - nose_y) * pixel_to_cm + 15  # Add ~15cm for top of head
        measurements['shoulder_width'] = shoulder_width_cm
        
        # Rough estimations for other measurements
        # In a real implementation, these would be more accurate
        measurements['chest_circumference'] = shoulder_width_cm * 2.4
        measurements['waist_circumference'] = shoulder_width_cm * 2.0
        measurements['hip_circumference'] = shoulder_width_cm * 2.6
        
        # Limb measurements
        # Arm length (shoulder to wrist)
        right_elbow = keypoints[3]
        right_wrist = keypoints[4]
        arm_length_px = np.linalg.norm(right_shoulder - right_elbow) + np.linalg.norm(right_elbow - right_wrist)
        measurements['arm_length'] = arm_length_px * pixel_to_cm
        
        # Inseam (hip to ankle)
        right_hip = keypoints[8]
        right_knee = keypoints[9]
        inseam_px = np.linalg.norm(right_hip - right_knee) + np.linalg.norm(right_knee - keypoints[10])
        measurements['inseam'] = inseam_px * pixel_to_cm
        
        return measurements
    
    def _extract_measurements_from_mesh(self, mesh: trimesh.Trimesh, joints: Optional[np.ndarray]) -> Dict:
        """
        Extract measurements from a 3D mesh.
        
        Args:
            mesh: Input body mesh
            joints: Joint positions
            
        Returns:
            Dictionary of measurements
        """
        # This is a simplified implementation
        # In a real implementation, you would use a proper algorithm
        
        # Initialize measurements
        measurements = {name: 0.0 for name in self.measurement_names}
        
        # Get mesh bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        
        # Height (Y-axis)
        height = bounds[1, 1] - bounds[0, 1]
        
        # Convert to cm (assuming the mesh is in meters)
        height_cm = height * 100.0
        
        # Scale factor to convert to cm
        scale_to_cm = 100.0
        
        # For more accurate measurements, we need to know the exact points
        # In a real implementation, this would be based on proper body segmentation
        
        # For now, use simplified calculations
        
        # Calculate measurements based on mesh geometry
        # Width at different heights
        
        # Check if we have joints
        if joints is not None and len(joints) >= 14:
            # Use joints for more accurate measurements
            # Assuming standard joint order: 
            # 0=nose, 1=neck, 2=right_shoulder, 3=right_elbow, 4=right_wrist, 
            # 5=left_shoulder, 6=left_elbow, 7=left_wrist, 8=right_hip, 9=right_knee,
            # 10=right_ankle, 11=left_hip, 12=left_knee, 13=left_ankle
            
            # Shoulder width (distance between shoulder joints)
            right_shoulder = joints[2]
            left_shoulder = joints[5]
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder) * scale_to_cm
            
            # Height
            head_top_y = bounds[1, 1]  # Highest point
            ankle_y = (joints[10, 1] + joints[13, 1]) / 2  # Average of ankles
            height_cm = (head_top_y - ankle_y) * scale_to_cm
            
            # Arm length
            right_arm_length = (np.linalg.norm(joints[2] - joints[3]) + 
                               np.linalg.norm(joints[3] - joints[4])) * scale_to_cm
            
            # Inseam
            right_inseam = (np.linalg.norm(joints[8] - joints[9]) + 
                           np.linalg.norm(joints[9] - joints[10])) * scale_to_cm
            
            # Save measurements
            measurements['height'] = height_cm
            measurements['shoulder_width'] = shoulder_width
            measurements['arm_length'] = right_arm_length
            measurements['inseam'] = right_inseam
            
        else:
            # No joints available, use simplified approach
            
            # Get the Y-values at different levels for slicing the mesh
            y_values = {
                'chest': bounds[0, 1] + 0.7 * height,
                'waist': bounds[0, 1] + 0.55 * height,
                'hips': bounds[0, 1] + 0.45 * height,
                'thigh': bounds[0, 1] + 0.3 * height,
                'calf': bounds[0, 1] + 0.15 * height
            }
            
            # Calculate circumferences at different heights
            for part, y in y_values.items():
                circumference = self._calculate_circumference_at_height(mesh, y) * scale_to_cm
                
                if part == 'chest':
                    measurements['chest_circumference'] = circumference
                elif part == 'waist':
                    measurements['waist_circumference'] = circumference
                elif part == 'hips':
                    measurements['hip_circumference'] = circumference
                elif part == 'thigh':
                    measurements['thigh_circumference'] = circumference
                elif part == 'calf':
                    measurements['calf_circumference'] = circumference
            
            # Shoulder width (X-axis at shoulder height)
            y_shoulder = bounds[0, 1] + 0.75 * height
            shoulder_points = mesh.vertices[np.abs(mesh.vertices[:, 1] - y_shoulder) < 0.02]
            if len(shoulder_points) > 0:
                shoulder_width = np.max(shoulder_points[:, 0]) - np.min(shoulder_points[:, 0])
                measurements['shoulder_width'] = shoulder_width * scale_to_cm
            
            # Height
            measurements['height'] = height_cm
            
            # Rough estimates for other measurements
            measurements['arm_length'] = 0.33 * height_cm
            measurements['inseam'] = 0.45 * height_cm
        
        return measurements
    
    def _extract_measurements_from_mesh_simplified(self, mesh: trimesh.Trimesh) -> Dict:
        """
        Extract measurements from a 3D mesh using a simplified approach.
        
        Args:
            mesh: Input body mesh
            
        Returns:
            Dictionary of measurements
        """
        # Get mesh bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        
        # Height (Y-axis)
        height = bounds[1, 1] - bounds[0, 1]
        
        # Width (X-axis)
        width = bounds[1, 0] - bounds[0, 0]
        
        # Depth (Z-axis)
        depth = bounds[1, 2] - bounds[0, 2]
        
        # Convert to cm (assuming the mesh is in meters)
        scale_to_cm = 100.0
        height_cm = height * scale_to_cm
        width_cm = width * scale_to_cm
        depth_cm = depth * scale_to_cm
        
        # Simplified measurements based on standard proportions
        measurements = {
            'height': height_cm,
            'shoulder_width': width_cm * 0.8,
            'chest_circumference': (width_cm + depth_cm) * 1.25,
            'waist_circumference': (width_cm + depth_cm) * 1.15,
            'hip_circumference': (width_cm + depth_cm) * 1.3,
            'inseam': height_cm * 0.45,
            'arm_length': height_cm * 0.33,
            'neck_circumference': (width_cm + depth_cm) * 0.4,
            'thigh_circumference': width_cm * 0.6,
            'calf_circumference': width_cm * 0.4,
            'bicep_circumference': width_cm * 0.25,
            'forearm_circumference': width_cm * 0.2,
            'wrist_circumference': width_cm * 0.15
        }
        
        return measurements
    
    def _calculate_circumference_at_height(self, mesh: trimesh.Trimesh, height: float) -> float:
        """
        Calculate the circumference of the mesh at a specific height.
        
        Args:
            mesh: Input body mesh
            height: Y coordinate at which to calculate circumference
            
        Returns:
            Circumference in the same units as the mesh
        """
        # Find vertices close to the given height
        epsilon = 0.02  # Tolerance
        section_points = mesh.vertices[np.abs(mesh.vertices[:, 1] - height) < epsilon]
        
        if len(section_points) < 3:
            return 0.0
        
        try:
            # Project points to XZ plane
            points_2d = section_points[:, [0, 2]]
            
            # Compute convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_2d)
            
            # Calculate perimeter
            perimeter = 0.0
            hull_points = points_2d[hull.vertices]
            
            # Add distances between consecutive hull points
            for i in range(len(hull.vertices)):
                j = (i + 1) % len(hull.vertices)
                perimeter += np.linalg.norm(hull_points[i] - hull_points[j])
            
            return perimeter
            
        except Exception as e:
            print(f"Error calculating circumference: {e}")
            
            # Fallback: estimate using bounding box
            max_x = np.max(section_points[:, 0])
            min_x = np.min(section_points[:, 0])
            max_z = np.max(section_points[:, 2])
            min_z = np.min(section_points[:, 2])
            
            # Approximate as an ellipse
            a = (max_x - min_x) / 2  # Semi-major axis
            b = (max_z - min_z) / 2  # Semi-minor axis
            
            # Approximate ellipse perimeter
            h = ((a - b) / (a + b)) ** 2
            circumference = np.pi * (a + b) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))
            
            return circumference
    
    def convert_measurements_to_size(self, measurements: Dict, size_standard: str = 'us') -> Dict:
        """
        Convert measurements to standard sizes.
        
        Args:
            measurements: Dictionary of measurements
            size_standard: Size standard to use ('us', 'eu', 'uk', 'international')
            
        Returns:
            Dictionary of sizes
        """
        print(f"Converting measurements to {size_standard} sizes")
        
        # This is a simplified implementation
        # In a real implementation, you would use a proper sizing database
        
        # Initialize sizes
        sizes = {}
        
        # Check if we have the necessary measurements
        required_measurements = ['chest_circumference', 'waist_circumference', 'hip_circumference']
        missing_measurements = [m for m in required_measurements if m not in measurements]
        
        if missing_measurements:
            print(f"Missing measurements: {missing_measurements}")
            return {'error': 'Missing required measurements'}
        
        # Simplified size calculation for demonstration
        # In a real implementation, this would be more sophisticated
        
        # Extract measurements
        chest = measurements['chest_circumference']
        waist = measurements['waist_circumference']
        hip = measurements['hip_circumference']
        
        # US sizing for shirts (simplified)
        if chest < 85:
            shirt_size = 'XS'
        elif chest < 95:
            shirt_size = 'S'
        elif chest < 105:
            shirt_size = 'M'
        elif chest < 115:
            shirt_size = 'L'
        elif chest < 125:
            shirt_size = 'XL'
        else:
            shirt_size = 'XXL'
        
        # US sizing for pants (simplified)
        if waist < 75:
            pant_size = '28'
        elif waist < 80:
            pant_size = '30'
        elif waist < 85:
            pant_size = '32'
        elif waist < 90:
            pant_size = '34'
        elif waist < 95:
            pant_size = '36'
        elif waist < 100:
            pant_size = '38'
        else:
            pant_size = '40+'
        
        # Create size dictionary
        sizes = {
            'shirt': shirt_size,
            'pants': pant_size
        }
        
        # Add additional size information if available
        if 'height' in measurements:
            height = measurements['height']
            
            # Simplified height-based sizing
            if height < 165:
                sizes['height_category'] = 'Short'
            elif height < 180:
                sizes['height_category'] = 'Regular'
            else:
                sizes['height_category'] = 'Tall'
        
        if 'inseam' in measurements:
            inseam = measurements['inseam']
            sizes['inseam'] = f"{int(inseam)}"
        
        return sizes
    
    def save_measurements_to_json(self, measurements: Dict, output_path: str) -> None:
        """
        Save measurements to a JSON file.
        
        Args:
            measurements: Dictionary of measurements
            output_path: Path to save the JSON file
        """
        # Create output directory if needed
        ensure_directory(os.path.dirname(output_path))
        
        # Save measurements
        with open(output_path, 'w') as f:
            json.dump(measurements, f, indent=2)
        
        print(f"Measurements saved to: {output_path}")
    
    def generate_measurement_visualization(self, image: np.ndarray, measurements: Dict) -> np.ndarray:
        """
        Generate a visualization of measurements on an image.
        
        Args:
            image: Input image
            measurements: Dictionary of measurements
            
        Returns:
            Image with visualized measurements
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        # Add measurements as text
        y_offset = 30
        for i, (name, value) in enumerate(measurements.items()):
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue
                
            # Format the measurement
            if name in ['height', 'arm_length', 'inseam', 'shoulder_width']:
                text = f"{name}: {value:.1f} cm"
            else:
                text = f"{name}: {value:.1f}"
                
            # Add text to image
            cv2.putText(vis_image, text, (10, y_offset + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return vis_image 