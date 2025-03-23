import os
import numpy as np
import torch
import mediapipe as mp
from typing import Dict, List, Tuple, Union, Optional
import face_alignment
import open3d as o3d
import trimesh

from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    preprocess_image,
    tensor_to_image,
    image_to_tensor,
    get_device
)

class FaceReconstructor:
    """
    Class for 3D face reconstruction from 2D images.
    Uses a combination of methods including MediaPipe Face Mesh and 
    face alignment models for accurate reconstruction.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the face reconstructor.
        
        Args:
            model_path: Path to pre-trained model weights (if needed)
        """
        self.device = get_device()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Initialize face alignment
        self.face_alignment_model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D, 
            device=str(self.device)
        )
        
        # Load custom model if provided
        self.custom_model = None
        if model_path and os.path.exists(model_path):
            self.custom_model = torch.load(model_path, map_location=self.device)
    
    def detect_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Detect facial landmarks in a 2D image.
        
        Args:
            image: Input image as a numpy array
            
        Returns:
            Array of facial landmarks
        """
        # Convert the image to RGB if needed
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Convert to RGB
            
        # Process the image with MediaPipe
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image")
        
        # Extract landmarks
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def reconstruct_3d_face(self, image: np.ndarray) -> Dict:
        """
        Reconstruct a 3D face model from a 2D image.
        
        Args:
            image: Input image as a numpy array
            
        Returns:
            Dictionary containing 3D face mesh and parameters
        """
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Detect facial landmarks
        landmarks = self.detect_landmarks(image)
        
        # Get 3D landmarks from face alignment model
        tensor_img = image_to_tensor(preprocessed_image)
        landmarks_3d = self.face_alignment_model.get_landmarks_from_image(tensor_img.cpu().numpy()[0].transpose(1, 2, 0) * 255)[0]
        
        # Create a 3D mesh from landmarks
        mesh = self._create_mesh_from_landmarks(landmarks_3d)
        
        # Apply custom model for refinement if available
        if self.custom_model:
            # Implementation depends on the specific model architecture
            # This is a placeholder for custom model inference
            pass
        
        return {
            "mesh": mesh,
            "landmarks": landmarks,
            "landmarks_3d": landmarks_3d
        }
    
    def _create_mesh_from_landmarks(self, landmarks_3d: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Create a 3D mesh from facial landmarks.
        
        Args:
            landmarks_3d: 3D facial landmarks
            
        Returns:
            3D triangle mesh
        """
        # Create point cloud from landmarks
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(landmarks_3d)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Create a mesh using Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        
        return mesh
    
    def reconstruct_from_video(self, video_path: str, sampling_rate: int = 5) -> Dict:
        """
        Reconstruct a 3D face model from a video by aggregating multiple frames.
        
        Args:
            video_path: Path to the input video
            sampling_rate: Number of frames to sample per second
            
        Returns:
            Dictionary containing 3D face mesh and parameters
        """
        import cv2
        
        # Open the video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sampling_rate)
        
        all_landmarks = []
        frame_count = 0
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            # Process only selected frames
            if frame_count % frame_interval == 0:
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect landmarks
                    landmarks = self.detect_landmarks(frame_rgb)
                    all_landmarks.append(landmarks)
                except Exception as e:
                    print(f"Failed to process frame {frame_count}: {e}")
            
            frame_count += 1
        
        video.release()
        
        # Aggregate landmarks from multiple frames for a more accurate reconstruction
        if not all_landmarks:
            raise ValueError("No valid face landmarks detected in the video")
            
        aggregated_landmarks = np.mean(all_landmarks, axis=0)
        
        # Create a mesh from the aggregated landmarks
        mesh = self._create_mesh_from_landmarks(aggregated_landmarks)
        
        return {
            "mesh": mesh,
            "aggregated_landmarks": aggregated_landmarks,
            "frame_count": frame_count,
            "processed_frames": len(all_landmarks)
        }
    
    def save_mesh(self, mesh: o3d.geometry.TriangleMesh, save_path: str) -> None:
        """
        Save a 3D mesh to file.
        
        Args:
            mesh: 3D triangle mesh
            save_path: Path where to save the mesh
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the mesh
        o3d.io.write_triangle_mesh(save_path, mesh)
    
    def export_for_texture_mapping(self, mesh: o3d.geometry.TriangleMesh, save_path: str) -> str:
        """
        Export a 3D mesh in a format suitable for texture mapping.
        
        Args:
            mesh: 3D triangle mesh
            save_path: Path where to save the exported mesh
            
        Returns:
            Path to the exported mesh file
        """
        # Convert to trimesh for additional export options
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        # Export as OBJ for texture mapping compatibility
        obj_path = save_path.replace('.ply', '.obj')
        trimesh_mesh.export(obj_path)
        
        return obj_path
