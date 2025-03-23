import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import cv2
import face_alignment
import open3d as o3d
import trimesh
from PIL import Image
import scipy.io as sio

from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    preprocess_image,
    tensor_to_image,
    image_to_tensor,
    get_device,
    ensure_directory
)

class BFMModel:
    """
    Basel Face Model (BFM) implementation for 3D face reconstruction.
    Provides access to the parametric face model components.
    """
    
    def __init__(self, bfm_path: str):
        """
        Initialize the BFM model.
        
        Args:
            bfm_path: Path to the BFM model file (.mat format)
        """
        self.bfm_path = bfm_path
        self.model = self._load_bfm()
        
    def _load_bfm(self) -> Dict:
        """
        Load the Basel Face Model from the provided path.
        
        Returns:
            Dictionary containing the model components
        """
        if not os.path.exists(self.bfm_path):
            raise FileNotFoundError(f"BFM model file not found: {self.bfm_path}")
        
        # Load the BFM model (typically in .mat format)
        try:
            model = sio.loadmat(self.bfm_path)
        except Exception as e:
            # Fallback to a simplified model for demonstration
            print(f"Error loading BFM model: {e}")
            print("Creating a simplified model for demonstration purposes")
            model = self._create_dummy_model()
            
        return model
    
    def _create_dummy_model(self) -> Dict:
        """
        Create a simplified dummy BFM model for demonstration.
        
        Returns:
            Dictionary containing the simplified model components
        """
        # Create a simplified face model with basic components
        # In a real implementation, you would load the actual BFM model components
        
        # Create some basic shape components
        vertex_count = 5000  # Simplified vertex count
        dim = 3  # 3D vertices (x, y, z)
        id_component_count = 80
        expr_component_count = 64
        
        dummy_model = {
            # Mean shape
            'shapeMU': np.zeros((vertex_count * dim, 1)),
            
            # Shape basis (identity)
            'shapePC': np.random.normal(0, 1, (vertex_count * dim, id_component_count)),
            
            # Shape standard deviations
            'shapeEV': np.abs(np.random.normal(0, 0.1, (id_component_count, 1))),
            
            # Expression basis
            'expPC': np.random.normal(0, 1, (vertex_count * dim, expr_component_count)),
            
            # Expression standard deviations
            'expEV': np.abs(np.random.normal(0, 0.1, (expr_component_count, 1))),
            
            # Face topology (triangles)
            'tri': np.array([[0, 1, 2], [1, 3, 2]]),  # Just a placeholder
            
            # Texture model components (simplified)
            'texMU': np.ones((vertex_count * 3, 1)) * 127.5,  # Mean texture (neutral gray)
            'texPC': np.random.normal(0, 1, (vertex_count * 3, 80)),  # Texture basis
            'texEV': np.abs(np.random.normal(0, 0.1, (80, 1)))  # Texture standard deviations
        }
        
        return dummy_model
    
    def get_shape_basis(self) -> np.ndarray:
        """
        Get the shape basis (identity components).
        
        Returns:
            Shape basis matrix
        """
        return self.model['shapePC']
    
    def get_expression_basis(self) -> np.ndarray:
        """
        Get the expression basis.
        
        Returns:
            Expression basis matrix
        """
        return self.model['expPC']
    
    def get_mean_shape(self) -> np.ndarray:
        """
        Get the mean face shape.
        
        Returns:
            Mean shape vector
        """
        return self.model['shapeMU']
    
    def get_triangles(self) -> np.ndarray:
        """
        Get the face topology (triangles).
        
        Returns:
            Triangle indices
        """
        return self.model['tri']
    
    def generate_shape(self, identity_params: np.ndarray, expression_params: np.ndarray) -> np.ndarray:
        """
        Generate a 3D face shape from identity and expression parameters.
        
        Args:
            identity_params: Identity parameters
            expression_params: Expression parameters
            
        Returns:
            3D face vertices
        """
        # Get model components
        shape_mu = self.get_mean_shape()
        shape_basis = self.get_shape_basis()
        exp_basis = self.get_expression_basis()
        
        # Generate shape
        shape = shape_mu.copy()
        
        # Add identity components
        if identity_params is not None:
            shape += shape_basis @ identity_params
        
        # Add expression components
        if expression_params is not None:
            shape += exp_basis @ expression_params
        
        # Reshape to 3D vertices
        vertex_count = shape.shape[0] // 3
        vertices = shape.reshape(vertex_count, 3)
        
        return vertices

class AdvancedFaceReconstructor:
    """
    Advanced implementation of 3D face reconstruction.
    Uses the Basel Face Model (BFM) and neural networks for accurate reconstruction.
    """
    
    def __init__(self, 
                bfm_path: Optional[str] = None,
                model_path: Optional[str] = None,
                use_deep_learning: bool = True):
        """
        Initialize the advanced face reconstructor.
        
        Args:
            bfm_path: Path to Basel Face Model file
            model_path: Path to pre-trained model weights
            use_deep_learning: Whether to use deep learning-based approach
        """
        self.device = get_device()
        self.use_deep_learning = use_deep_learning
        
        # Initialize BFM model
        self.bfm = None
        if bfm_path and os.path.exists(bfm_path):
            try:
                self.bfm = BFMModel(bfm_path)
            except Exception as e:
                print(f"Error initializing BFM model: {e}")
                print("Using simplified reconstruction methods instead")
        
        # Initialize face alignment model for landmark detection
        self.face_alignment_model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize deep learning model if requested
        self.dl_model = None
        if use_deep_learning and model_path and os.path.exists(model_path):
            self.dl_model = self._load_deep_learning_model(model_path)
    
    def _load_deep_learning_model(self, model_path: str) -> Optional[nn.Module]:
        """
        Load a pre-trained deep learning model for face reconstruction.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded model or None if loading fails
        """
        # This is a placeholder for loading a deep learning model
        # In a real implementation, load the specific model architecture and weights
        
        try:
            # Example model architecture (simplified ResNet)
            class FaceReconstructionNet(nn.Module):
                def __init__(self, identity_dim=80, expression_dim=64):
                    super().__init__()
                    self.identity_dim = identity_dim
                    self.expression_dim = expression_dim
                    
                    # Feature extraction layers
                    self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
                    self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
                    self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
                    
                    # Fully connected layers
                    self.fc1 = nn.Linear(256 * 8 * 8, 1024)
                    self.fc2 = nn.Linear(1024, 512)
                    
                    # Output layers
                    self.fc_id = nn.Linear(512, identity_dim)
                    self.fc_exp = nn.Linear(512, expression_dim)
                    self.fc_tex = nn.Linear(512, identity_dim)  # Texture parameters
                    self.fc_pose = nn.Linear(512, 6)  # Rotation (3) + translation (3)
                
                def forward(self, x):
                    # Input is expected to be 3x128x128
                    x = F.relu(self.conv1(x))  # 32x64x64
                    x = F.relu(self.conv2(x))  # 64x32x32
                    x = F.relu(self.conv3(x))  # 128x16x16
                    x = F.relu(self.conv4(x))  # 256x8x8
                    
                    x = x.view(-1, 256 * 8 * 8)
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    
                    # Output parameters
                    id_params = self.fc_id(x)
                    exp_params = self.fc_exp(x)
                    tex_params = self.fc_tex(x)
                    pose_params = self.fc_pose(x)
                    
                    return {
                        'id_params': id_params,
                        'exp_params': exp_params,
                        'tex_params': tex_params,
                        'pose_params': pose_params
                    }
            
            # Create model
            model = FaceReconstructionNet().to(self.device)
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading deep learning model: {e}")
            return None
    
    def preprocess_for_reconstruction(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for reconstruction.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to expected input size
        resized = cv2.resize(image, (224, 224))
        
        # Convert to tensor and normalize
        tensor = image_to_tensor(resized).to(self.device)
        
        # Normalize to [-1, 1]
        tensor = tensor * 2.0 - 1.0
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def detect_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Detect facial landmarks using the face alignment model.
        
        Args:
            image: Input face image
            
        Returns:
            Detected landmarks (68 3D points)
        """
        try:
            # Detect landmarks
            landmarks = self.face_alignment_model.get_landmarks_from_image(image)[0]
            return landmarks
        except Exception as e:
            print(f"Error detecting landmarks: {e}")
            # Return dummy landmarks
            return np.zeros((68, 3))
    
    def estimate_parameters_from_landmarks(self, landmarks: np.ndarray) -> Dict:
        """
        Estimate model parameters from detected landmarks.
        
        Args:
            landmarks: Detected facial landmarks
            
        Returns:
            Dictionary of estimated parameters
        """
        # This is a simplified implementation
        # In a real system, use a more sophisticated fitting algorithm
        
        # Simplified parameter estimation
        # Generate random parameters for demonstration
        identity_params = np.random.normal(0, 0.1, (80, 1))
        expression_params = np.random.normal(0, 0.1, (64, 1))
        
        return {
            'identity_params': identity_params,
            'expression_params': expression_params
        }
    
    def reconstruct_3d_face(self, image: np.ndarray) -> Dict:
        """
        Reconstruct a 3D face from a single image.
        
        Args:
            image: Input face image
            
        Returns:
            Dictionary containing the reconstruction results
        """
        # Make sure image is RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]  # Remove alpha channel
        
        # Detect landmarks for all methods
        landmarks = self.detect_landmarks(image)
        
        # Use deep learning approach if available
        if self.use_deep_learning and self.dl_model is not None:
            # Preprocess image
            input_tensor = self.preprocess_for_reconstruction(image)
            
            # Run inference
            with torch.no_grad():
                prediction = self.dl_model(input_tensor)
            
            # Extract parameters
            id_params = prediction['id_params'].cpu().numpy()[0]
            exp_params = prediction['exp_params'].cpu().numpy()[0]
            
            # Reshape for BFM model
            id_params = id_params.reshape(-1, 1)
            exp_params = exp_params.reshape(-1, 1)
            
        else:
            # Fallback to landmark-based fitting
            params = self.estimate_parameters_from_landmarks(landmarks)
            id_params = params['identity_params']
            exp_params = params['expression_params']
        
        # Generate 3D mesh using BFM model
        if self.bfm is not None:
            vertices = self.bfm.generate_shape(id_params, exp_params)
            triangles = self.bfm.get_triangles()
            
            # Create mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            # Compute normals
            mesh.compute_vertex_normals()
        else:
            # Fallback to landmark-based simple mesh
            mesh = self._create_simple_mesh_from_landmarks(landmarks)
        
        return {
            "mesh": mesh,
            "landmarks": landmarks,
            "landmarks_3d": landmarks,  # Already 3D from face_alignment
            "identity_params": id_params,
            "expression_params": exp_params
        }
    
    def _create_simple_mesh_from_landmarks(self, landmarks: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Create a simple mesh from landmarks when BFM is not available.
        
        Args:
            landmarks: 3D landmarks
            
        Returns:
            Simple face mesh
        """
        # Create a simple triangulation of the landmarks
        from scipy.spatial import Delaunay
        
        # Use only 2D coordinates for triangulation
        landmarks_2d = landmarks[:, :2]
        triangulation = Delaunay(landmarks_2d)
        
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(landmarks)
        mesh.triangles = o3d.utility.Vector3iVector(triangulation.simplices)
        
        # Compute normals
        mesh.compute_vertex_normals()
        
        return mesh
    
    def reconstruct_from_video(self, video_path: str, sampling_rate: int = 5) -> Dict:
        """
        Reconstruct a 3D face from video by aggregating multiple frames.
        
        Args:
            video_path: Path to input video
            sampling_rate: Number of frames to skip between samples
            
        Returns:
            Dictionary containing the aggregated reconstruction
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Containers for aggregation
        all_identity_params = []
        all_expression_params = []
        all_landmarks = []
        frame_count = 0
        
        # Process video frames
        while True:
            # Read frame
            ret, frame = cap.read()
            
            # Break if no more frames
            if not ret:
                break
            
            # Process only every sampling_rate frames
            if frame_count % sampling_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Reconstruct face
                    result = self.reconstruct_3d_face(frame_rgb)
                    
                    # Store parameters
                    all_identity_params.append(result['identity_params'])
                    all_expression_params.append(result['expression_params'])
                    all_landmarks.append(result['landmarks'])
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        # Release video
        cap.release()
        
        # Check if we have any valid reconstructions
        if not all_identity_params:
            raise ValueError("No valid reconstructions from video")
        
        # Aggregate parameters (average identity, representative expression)
        avg_identity_params = np.mean(all_identity_params, axis=0)
        
        # Use the expression from the middle frame as representative
        mid_idx = len(all_expression_params) // 2
        representative_expression_params = all_expression_params[mid_idx]
        
        # Generate final mesh using BFM model
        if self.bfm is not None:
            vertices = self.bfm.generate_shape(avg_identity_params, representative_expression_params)
            triangles = self.bfm.get_triangles()
            
            # Create mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            # Compute normals
            mesh.compute_vertex_normals()
        else:
            # Fallback to landmark-based simple mesh
            avg_landmarks = np.mean(all_landmarks, axis=0)
            mesh = self._create_simple_mesh_from_landmarks(avg_landmarks)
        
        return {
            "mesh": mesh,
            "aggregated_landmarks": np.mean(all_landmarks, axis=0),
            "identity_params": avg_identity_params,
            "expression_params": representative_expression_params,
            "frame_count": frame_count,
            "processed_frames": len(all_identity_params)
        }
    
    def save_mesh(self, mesh: o3d.geometry.TriangleMesh, output_path: str) -> None:
        """
        Save a mesh to file.
        
        Args:
            mesh: Mesh to save
            output_path: Output file path
        """
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_path))
        
        # Save mesh
        o3d.io.write_triangle_mesh(output_path, mesh)
    
    def export_for_texture_mapping(self, mesh: o3d.geometry.TriangleMesh, output_path: str) -> str:
        """
        Export mesh in a format suitable for texture mapping.
        
        Args:
            mesh: Mesh to export
            output_path: Output file path
            
        Returns:
            Path to exported mesh file
        """
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_path))
        
        # Save mesh
        o3d.io.write_triangle_mesh(output_path, mesh)
        
        return output_path 