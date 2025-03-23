import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Dict, List, Tuple, Union, Optional
import open3d as o3d
import trimesh
from scipy.spatial import Delaunay

from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    preprocess_image,
    image_to_tensor,
    tensor_to_image,
    get_device,
    ensure_directory
)
from app.avatar_creation.face_modeling.advanced_face_reconstruction import (
    AdvancedFaceReconstructor,
    BFMModel
)

class MultiViewReconstructor:
    """
    Class for 3D face reconstruction from multiple images (multi-view).
    Combines information from multiple viewpoints for more accurate reconstruction.
    """
    
    def __init__(self, 
                 bfm_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 use_deep_learning: bool = True):
        """
        Initialize the multi-view reconstructor.
        
        Args:
            bfm_path: Path to Basel Face Model file
            model_path: Path to pre-trained model weights
            use_deep_learning: Whether to use deep learning-based approach
        """
        self.device = get_device()
        
        # Initialize the base reconstructor for single images
        self.base_reconstructor = AdvancedFaceReconstructor(
            bfm_path=bfm_path,
            model_path=model_path,
            use_deep_learning=use_deep_learning
        )
        
        # BFM model reference from base reconstructor
        self.bfm = self.base_reconstructor.bfm
    
    def estimate_view_parameters(self, image: np.ndarray) -> Dict:
        """
        Estimate camera/view parameters for an image.
        
        Args:
            image: Input face image
            
        Returns:
            Dictionary containing estimated view parameters
        """
        # Detect landmarks to estimate rough head pose
        landmarks = self.base_reconstructor.detect_landmarks(image)
        
        # Simple head pose estimation based on landmarks
        # A real implementation would use a more sophisticated algorithm
        
        # Use specific landmark points for pose estimation
        # Assuming landmarks are in the format of 68-point model:
        # - Point 33 (nose tip)
        # - Point 8 (chin)
        # - Point 36 (left eye corner)
        # - Point 45 (right eye corner)
        
        # Calculate some basic metrics for head pose
        if landmarks.shape[0] >= 68:
            nose_tip = landmarks[33]
            chin = landmarks[8]
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            
            # Calculate eye line
            eye_line = right_eye - left_eye
            
            # Calculate face normal (simplified)
            face_normal = np.cross(eye_line, chin - nose_tip)
            face_normal = face_normal / np.linalg.norm(face_normal)
            
            # Extract approximate Euler angles (yaw, pitch, roll)
            # This is a simplified approximation
            yaw = np.arctan2(face_normal[0], face_normal[2])
            pitch = np.arcsin(np.clip(face_normal[1], -1, 1))
            roll = np.arctan2(eye_line[1], eye_line[0])
            
            rotation = np.array([yaw, pitch, roll])
        else:
            # Fallback if not enough landmarks
            rotation = np.zeros(3)
        
        # Estimate rough translation (simplified)
        if landmarks.shape[0] > 0:
            # Use centroid of landmarks as rough translation
            centroid = np.mean(landmarks, axis=0)
            translation = centroid
        else:
            # Fallback
            translation = np.zeros(3)
        
        return {
            "rotation": rotation,
            "translation": translation,
            "landmarks": landmarks
        }
    
    def reconstruct_from_multiple_images(self, 
                                        image_paths: List[str], 
                                        weights: Optional[List[float]] = None) -> Dict:
        """
        Reconstruct a 3D face from multiple images.
        
        Args:
            image_paths: List of paths to input images
            weights: Optional weights for each image (if None, equal weights are used)
            
        Returns:
            Dictionary containing the reconstruction results
        """
        # Load images
        images = [load_image(path) for path in image_paths]
        
        if not images:
            raise ValueError("No valid images provided")
        
        # Set equal weights if not provided
        if weights is None:
            weights = [1.0 / len(images)] * len(images)
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Containers for parameter aggregation
        all_identity_params = []
        all_expression_params = []
        all_landmarks = []
        all_view_params = []
        
        # Process each image
        for i, image in enumerate(images):
            try:
                # Reconstruct face from this view
                result = self.base_reconstructor.reconstruct_3d_face(image)
                
                # Estimate view parameters
                view_params = self.estimate_view_parameters(image)
                
                # Store parameters with weight
                all_identity_params.append((result["identity_params"], weights[i]))
                all_expression_params.append((result["expression_params"], weights[i]))
                all_landmarks.append((result["landmarks"], weights[i]))
                all_view_params.append((view_params, weights[i]))
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
        
        if not all_identity_params:
            raise ValueError("No valid reconstructions from images")
        
        # Aggregate parameters (weighted average)
        # For identity, compute weighted average
        avg_identity_params = np.zeros_like(all_identity_params[0][0])
        total_weight = 0
        
        for params, weight in all_identity_params:
            avg_identity_params += params * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_identity_params /= total_weight
        
        # For expression, use the parameters from the frontal view or the highest-weighted view
        # Determine which view is most frontal based on yaw angle
        frontal_idx = 0
        min_yaw = float('inf')
        
        for i, (view_params, weight) in enumerate(all_view_params):
            yaw = abs(view_params["rotation"][0])  # Absolute yaw angle
            if yaw < min_yaw:
                min_yaw = yaw
                frontal_idx = i
        
        # Use expression from most frontal view
        representative_expression_params = all_expression_params[frontal_idx][0]
        
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
            # Combine landmarks from all views
            combined_landmarks = np.zeros_like(all_landmarks[0][0])
            total_weight = 0
            
            for landmarks, weight in all_landmarks:
                combined_landmarks += landmarks * weight
                total_weight += weight
            
            if total_weight > 0:
                combined_landmarks /= total_weight
            
            mesh = self._create_simple_mesh_from_landmarks(combined_landmarks)
        
        return {
            "mesh": mesh,
            "identity_params": avg_identity_params,
            "expression_params": representative_expression_params,
            "view_params": all_view_params
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
    
    def reconstruct_with_shape_from_shading(self, 
                                           mesh: o3d.geometry.TriangleMesh,
                                           images: List[np.ndarray],
                                           view_params: List[Dict]) -> o3d.geometry.TriangleMesh:
        """
        Enhance reconstruction with shape-from-shading technique.
        
        Args:
            mesh: Base mesh to enhance
            images: Input images
            view_params: View parameters for each image
            
        Returns:
            Enhanced mesh with finer details
        """
        # This is a placeholder for a shape-from-shading implementation
        # Shape-from-shading is a technique to recover fine details from images
        # based on shading variations
        
        # In a real implementation, this would:
        # 1. Render the mesh from each camera viewpoint
        # 2. Compare the rendered image with the actual image
        # 3. Estimate the surface normal corrections
        # 4. Update the mesh vertices to match the estimated normals
        
        # For this demo, we'll just return the original mesh
        return mesh
    
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