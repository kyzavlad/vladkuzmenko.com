import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
import json
from PIL import Image
import trimesh

try:
    # Try to import SMPL dependencies
    # SMPL is a statistical body model commonly used for body reconstruction
    import smplx
    SMPL_AVAILABLE = True
except ImportError:
    print("Warning: SMPL model not available. Using simplified body model instead.")
    SMPL_AVAILABLE = False

from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    get_device,
    ensure_directory
)

class BodyReconstructor:
    """
    Class for reconstructing 3D body models from images or measurements.
    Supports both SMPL-based reconstruction (if available) and simplified parametric models.
    """
    
    def __init__(self, 
                model_path: Optional[str] = None,
                gender: str = "neutral",
                num_betas: int = 10,
                use_pose_blendshapes: bool = True,
                use_face_contour: bool = True,
                high_quality: bool = False):
        """
        Initialize the body reconstruction module.
        
        Args:
            model_path: Path to pre-trained model or SMPL parameters
            gender: Gender for the body model ('male', 'female', or 'neutral')
            num_betas: Number of shape parameters to use
            use_pose_blendshapes: Whether to use pose-dependent blendshapes
            use_face_contour: Whether to include face contour landmarks
            high_quality: Whether to use high-quality settings
        """
        self.device = get_device()
        self.model_path = model_path
        self.gender = gender
        self.num_betas = num_betas
        self.use_pose_blendshapes = use_pose_blendshapes
        self.use_face_contour = use_face_contour
        self.high_quality = high_quality
        
        # Body model
        self.body_model = None
        
        # Initialize SMPL model if available
        if SMPL_AVAILABLE and self.model_path:
            try:
                self._initialize_smpl_model()
            except Exception as e:
                print(f"Error initializing SMPL model: {e}")
                print("Falling back to simplified body model")
        
        # Use simplified model as fallback
        if self.body_model is None:
            self._initialize_simplified_model()
        
        # Pose estimator for initial pose
        self.pose_estimator = self._initialize_pose_estimator()
    
    def _initialize_smpl_model(self):
        """
        Initialize the SMPL body model.
        """
        if not SMPL_AVAILABLE:
            return
        
        try:
            self.body_model = smplx.create(
                self.model_path,
                model_type='smpl',
                gender=self.gender,
                num_betas=self.num_betas,
                use_face_contour=self.use_face_contour,
                batch_size=1
            ).to(self.device)
            print(f"SMPL model initialized with gender: {self.gender}")
        except Exception as e:
            print(f"Failed to initialize SMPL model: {e}")
            self.body_model = None
    
    def _initialize_simplified_model(self):
        """
        Initialize a simplified parametric body model as fallback.
        """
        print("Initializing simplified parametric body model")
        # This is a placeholder for a simplified body model implementation
        # In practice, you would implement a proper parametric model here
        
        class SimplifiedBodyModel:
            def __init__(self, gender='neutral', device='cpu', high_quality=False):
                self.gender = gender
                self.device = device
                self.high_quality = high_quality
                self.template_path = os.path.join(
                    os.path.dirname(__file__), 
                    'data', 
                    f'template_{gender}.obj'
                )
                
                # Create default template if not exists
                if not os.path.exists(self.template_path):
                    ensure_directory(os.path.dirname(self.template_path))
                    self._create_default_template()
                
                # Load template mesh
                self.template_mesh = self._load_template()
                
                # Shape parameters (height, weight, proportions)
                self.shape_params = {
                    'height': 175.0,  # cm
                    'weight': 70.0,   # kg
                    'chest': 90.0,    # cm
                    'waist': 80.0,    # cm
                    'hips': 90.0      # cm
                }
                
                # Joint positions (simplified)
                self.joints = self._create_default_joints()
            
            def _create_default_template(self):
                """Create a very simplified human mesh template."""
                try:
                    # Try to create a basic human mesh
                    # In a real implementation, you would use a proper template
                    import trimesh
                    from trimesh.creation import capsule
                    
                    # Create torso
                    torso = capsule(height=0.6, radius=0.2)
                    
                    # Create head
                    head = trimesh.primitives.Sphere(radius=0.15)
                    head.apply_translation([0, 0, 0.5])
                    
                    # Combine parts
                    mesh = trimesh.util.concatenate([torso, head])
                    
                    # Save template
                    mesh.export(self.template_path)
                    print(f"Created default template at {self.template_path}")
                    
                except Exception as e:
                    print(f"Error creating default template: {e}")
                    # Create a minimal OBJ file with a cube
                    with open(self.template_path, 'w') as f:
                        f.write("# Minimal human body template\n")
                        f.write("v -0.5 -0.5 -0.5\n")
                        f.write("v -0.5 -0.5 0.5\n")
                        f.write("v -0.5 0.5 -0.5\n")
                        f.write("v -0.5 0.5 0.5\n")
                        f.write("v 0.5 -0.5 -0.5\n")
                        f.write("v 0.5 -0.5 0.5\n")
                        f.write("v 0.5 0.5 -0.5\n")
                        f.write("v 0.5 0.5 0.5\n")
                        f.write("f 1 3 4 2\n")
                        f.write("f 5 7 8 6\n")
                        f.write("f 1 5 6 2\n")
                        f.write("f 3 7 8 4\n")
                        f.write("f 1 3 7 5\n")
                        f.write("f 2 4 8 6\n")
            
            def _load_template(self):
                """Load the template mesh."""
                try:
                    import trimesh
                    return trimesh.load(self.template_path)
                except Exception as e:
                    print(f"Error loading template mesh: {e}")
                    # Return a simple cube as fallback
                    return trimesh.primitives.Box()
            
            def _create_default_joints(self):
                """Create default skeleton joints."""
                # Simplified skeletal structure with basic joints
                joints = {
                    'root': np.array([0.0, 0.0, 0.0]),
                    'spine': np.array([0.0, 0.0, 0.2]),
                    'neck': np.array([0.0, 0.0, 0.4]),
                    'head': np.array([0.0, 0.0, 0.5]),
                    'left_shoulder': np.array([-0.2, 0.0, 0.4]),
                    'left_elbow': np.array([-0.4, 0.0, 0.3]),
                    'left_wrist': np.array([-0.5, 0.0, 0.2]),
                    'right_shoulder': np.array([0.2, 0.0, 0.4]),
                    'right_elbow': np.array([0.4, 0.0, 0.3]),
                    'right_wrist': np.array([0.5, 0.0, 0.2]),
                    'left_hip': np.array([-0.1, 0.0, -0.1]),
                    'left_knee': np.array([-0.15, 0.0, -0.3]),
                    'left_ankle': np.array([-0.15, 0.0, -0.5]),
                    'right_hip': np.array([0.1, 0.0, -0.1]),
                    'right_knee': np.array([0.15, 0.0, -0.3]),
                    'right_ankle': np.array([0.15, 0.0, -0.5])
                }
                return joints
            
            def forward(self, betas=None, global_orient=None, body_pose=None, transl=None):
                """Forward pass to generate mesh based on parameters."""
                # In a real implementation, this would deform the template
                # based on the provided parameters
                
                # Create a simple result structure similar to SMPL
                result = {
                    'vertices': np.asarray(self.template_mesh.vertices),
                    'joints': np.array(list(self.joints.values())),
                    'faces': np.asarray(self.template_mesh.faces)
                }
                
                # Apply any transformations if provided
                if transl is not None:
                    result['vertices'] = result['vertices'] + transl.cpu().numpy()
                    for joint_name in self.joints:
                        self.joints[joint_name] = self.joints[joint_name] + transl.cpu().numpy()
                
                # Modify vertices based on shape parameters if provided
                if betas is not None:
                    # Simplified shape deformation
                    # Just scale the mesh based on first beta parameter
                    scale_factor = 1.0 + 0.1 * betas[0, 0].item()
                    result['vertices'] = result['vertices'] * scale_factor
                
                return result
        
        # Initialize the simplified model
        self.body_model = SimplifiedBodyModel(
            gender=self.gender, 
            device=self.device, 
            high_quality=self.high_quality
        )
        print("Simplified body model initialized")
    
    def _initialize_pose_estimator(self):
        """
        Initialize a simple pose estimator.
        """
        # In a real implementation, you would use a proper pose estimator like MediaPipe
        # This is a simplified placeholder
        
        class SimplePoseEstimator:
            def __init__(self, device='cpu'):
                self.device = device
                print("Initializing simple pose estimator")
            
            def estimate_pose(self, image):
                # Placeholder for pose estimation
                # Returns default pose parameters
                return {
                    'global_orient': torch.zeros((1, 3), device=self.device),
                    'body_pose': torch.zeros((1, 69), device=self.device),
                    'betas': torch.zeros((1, 10), device=self.device),
                    'transl': torch.zeros((1, 3), device=self.device)
                }
        
        return SimplePoseEstimator(device=self.device)
    
    def reconstruct_from_image(self, image_path: str) -> Dict:
        """
        Reconstruct a 3D body model from a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing the reconstruction results
        """
        print(f"Reconstructing body from image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        
        # Estimate pose from image
        pose_params = self.pose_estimator.estimate_pose(image)
        
        # Forward pass through body model
        body_model_output = self.body_model.forward(**pose_params)
        
        # Extract results
        vertices = body_model_output['vertices']
        faces = body_model_output['faces']
        joints = body_model_output['joints']
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return {
            'mesh': mesh,
            'joints': joints,
            'pose_params': pose_params,
            'image': image
        }
    
    def reconstruct_from_measurements(self, measurements: Dict[str, float]) -> Dict:
        """
        Reconstruct a 3D body model from body measurements.
        
        Args:
            measurements: Dictionary of body measurements
                (height, weight, chest, waist, hips, etc.)
            
        Returns:
            Dictionary containing the reconstruction results
        """
        print("Reconstructing body from measurements")
        
        # Convert measurements to shape parameters
        # This is a simplified mapping and would be much more sophisticated in a real implementation
        betas = torch.zeros((1, self.num_betas), device=self.device)
        
        # Map height to first beta parameter (simplified)
        if 'height' in measurements:
            # Normalize height (assuming average height is 170cm with std of 10cm)
            height_normalized = (measurements['height'] - 170.0) / 10.0
            betas[0, 0] = torch.tensor(height_normalized, device=self.device)
        
        # Map weight to second beta parameter (simplified)
        if 'weight' in measurements:
            # Normalize weight (assuming average weight is 70kg with std of 15kg)
            weight_normalized = (measurements['weight'] - 70.0) / 15.0
            betas[0, 1] = torch.tensor(weight_normalized, device=self.device)
        
        # Default pose parameters
        pose_params = {
            'global_orient': torch.zeros((1, 3), device=self.device),
            'body_pose': torch.zeros((1, 69), device=self.device),
            'betas': betas,
            'transl': torch.zeros((1, 3), device=self.device)
        }
        
        # Forward pass through body model
        body_model_output = self.body_model.forward(**pose_params)
        
        # Extract results
        vertices = body_model_output['vertices']
        faces = body_model_output['faces']
        joints = body_model_output['joints']
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return {
            'mesh': mesh,
            'joints': joints,
            'pose_params': pose_params,
            'measurements': measurements
        }
    
    def reconstruct_from_multiple_images(self, image_paths: List[str]) -> Dict:
        """
        Reconstruct a 3D body model from multiple images showing different views.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            Dictionary containing the reconstruction results
        """
        print(f"Reconstructing body from {len(image_paths)} images")
        
        # For simplicity, use the first image for initial reconstruction
        # In a real implementation, you would integrate information from all views
        initial_result = self.reconstruct_from_image(image_paths[0])
        
        # In a real implementation, you would refine the initial result
        # using information from all views
        # This is a placeholder for that process
        
        return {
            'mesh': initial_result['mesh'],
            'joints': initial_result['joints'],
            'pose_params': initial_result['pose_params'],
            'images': [load_image(path) for path in image_paths]
        }
    
    def reconstruct_from_video(self, video_path: str, frame_rate: int = 5) -> Dict:
        """
        Reconstruct a 3D body model from a video.
        
        Args:
            video_path: Path to input video
            frame_rate: Number of frames to extract per second
            
        Returns:
            Dictionary containing the reconstruction results
        """
        print(f"Reconstructing body from video: {video_path}")
        
        # Extract frames from video
        frames = self._extract_frames_from_video(video_path, frame_rate)
        
        if not frames:
            raise ValueError(f"Failed to extract frames from video: {video_path}")
        
        print(f"Extracted {len(frames)} frames from video")
        
        # Use the extracted frames for reconstruction
        # For simplicity, use the middle frame for initial reconstruction
        middle_frame = frames[len(frames) // 2]
        
        # Save the middle frame to a temporary file
        temp_frame_path = "_temp_frame.jpg"
        cv2.imwrite(temp_frame_path, cv2.cvtColor(middle_frame, cv2.COLOR_RGB2BGR))
        
        # Reconstruct from the middle frame
        result = self.reconstruct_from_image(temp_frame_path)
        
        # Clean up temporary file
        try:
            os.remove(temp_frame_path)
        except:
            pass
        
        # Add frames to the result
        result['frames'] = frames
        
        return result
    
    def _extract_frames_from_video(self, video_path: str, frame_rate: int) -> List[np.ndarray]:
        """
        Extract frames from a video at the specified frame rate.
        
        Args:
            video_path: Path to the video
            frame_rate: Number of frames to extract per second
            
        Returns:
            List of extracted frames as NumPy arrays (RGB format)
        """
        frames = []
        
        try:
            # Open the video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video FPS: {fps}, Total frames: {frame_count}")
            
            # Calculate frame interval
            frame_interval = int(fps / frame_rate)
            
            # Extract frames
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_idx += 1
            
            # Release the video
            cap.release()
            
        except Exception as e:
            print(f"Error extracting frames from video: {e}")
        
        return frames
    
    def save_mesh(self, mesh, output_path: str) -> str:
        """
        Save a mesh to a file.
        
        Args:
            mesh: Mesh to save
            output_path: Path to save the mesh
            
        Returns:
            Path to the saved mesh
        """
        # Create output directory if needed
        ensure_directory(os.path.dirname(output_path))
        
        # Save mesh
        mesh.export(output_path)
        
        return output_path
    
    def export_for_texturing(self, mesh, output_path: str) -> str:
        """
        Export a mesh for texture mapping.
        
        Args:
            mesh: Mesh to export
            output_path: Path to save the mesh
            
        Returns:
            Path to the exported mesh
        """
        # This is a simplified export function
        # In a real implementation, you might prepare the mesh for UV mapping
        return self.save_mesh(mesh, output_path)
    
    def apply_pose(self, mesh, pose_params: Dict) -> trimesh.Trimesh:
        """
        Apply a pose to a mesh.
        
        Args:
            mesh: Input mesh
            pose_params: Pose parameters
            
        Returns:
            Posed mesh
        """
        # This is a simplified implementation
        # In a real implementation, this would use skinning to deform the mesh
        
        # For now, just return the original mesh
        return mesh 