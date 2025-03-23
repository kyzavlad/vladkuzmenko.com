import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
import kornia
from sklearn.cluster import KMeans

from app.avatar_creation.face_modeling.utils import (
    tensor_to_image,
    image_to_tensor,
    get_device
)

class FeaturePreservation:
    """
    Class for preserving detailed facial features during 3D reconstruction.
    Implements algorithms to maintain identity and distinctive facial features.
    """
    
    def __init__(self, feature_threshold: float = 0.75):
        """
        Initialize the feature preservation system.
        
        Args:
            feature_threshold: Threshold for feature detection sensitivity (0-1)
        """
        self.device = get_device()
        self.feature_threshold = feature_threshold
        
        # Initialize feature detection parameters
        self.corner_detection_params = {
            'block_size': 5,
            'ksize': 3,
            'k': 0.04
        }
    
    def detect_important_features(self, image: np.ndarray) -> Dict:
        """
        Detect important facial features that need preservation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing feature points and importance maps
        """
        # Convert image to grayscale for feature detection
        if image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Harris corner detection for distinct features
        corner_response = cv2.cornerHarris(
            gray_image.astype(np.float32),
            self.corner_detection_params['block_size'],
            self.corner_detection_params['ksize'],
            self.corner_detection_params['k']
        )
        
        # Normalize corner response
        corner_response = cv2.normalize(corner_response, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Create a binary mask of important corner features
        corner_mask = corner_response > self.feature_threshold
        
        # Use edge detection for structural features
        edges = cv2.Canny(gray_image, 100, 200)
        
        # Generate a feature importance map
        importance_map = np.zeros_like(gray_image, dtype=np.float32)
        importance_map[corner_mask] = 1.0
        importance_map = cv2.add(importance_map, edges.astype(np.float32) / 255.0)
        importance_map = np.clip(importance_map, 0, 1)
        
        # Find feature points using Good Features to Track
        feature_points = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        if feature_points is None:
            feature_points = np.array([])
        
        return {
            "importance_map": importance_map,
            "feature_points": feature_points,
            "corner_response": corner_response,
            "edge_map": edges
        }
    
    def compute_feature_descriptors(self, image: np.ndarray, feature_points: np.ndarray) -> np.ndarray:
        """
        Compute descriptors for detected feature points.
        
        Args:
            image: Input image
            feature_points: Array of feature points
            
        Returns:
            Array of feature descriptors
        """
        # Convert to grayscale if needed
        if image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Initialize ORB descriptor
        orb = cv2.ORB_create()
        
        # Convert feature points to KeyPoint objects
        keypoints = [cv2.KeyPoint(x=float(point[0][0]), y=float(point[0][1]), size=20) for point in feature_points]
        
        # Compute descriptors
        _, descriptors = orb.compute(gray_image, keypoints)
        
        # Return empty array if no descriptors found
        if descriptors is None:
            return np.array([])
            
        return descriptors
    
    def cluster_features(self, feature_points: np.ndarray, descriptors: np.ndarray, n_clusters: int = 5) -> Dict:
        """
        Cluster feature points based on their descriptors.
        
        Args:
            feature_points: Array of feature points
            descriptors: Array of feature descriptors
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with feature clusters
        """
        if len(feature_points) == 0 or len(descriptors) == 0:
            return {
                "cluster_centers": np.array([]),
                "labels": np.array([]),
                "clustered_points": {}
            }
        
        # Adjust number of clusters based on available points
        actual_clusters = min(n_clusters, len(feature_points))
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
        labels = kmeans.fit_predict(descriptors)
        
        # Group points by cluster
        clustered_points = {}
        for i, label in enumerate(labels):
            if label not in clustered_points:
                clustered_points[label] = []
            clustered_points[label].append(feature_points[i])
        
        return {
            "cluster_centers": kmeans.cluster_centers_,
            "labels": labels,
            "clustered_points": clustered_points
        }
    
    def generate_feature_constraints(self, mesh_vertices: np.ndarray, feature_points: np.ndarray) -> Dict:
        """
        Generate constraints for mesh vertices based on feature points.
        
        Args:
            mesh_vertices: 3D mesh vertices
            feature_points: 2D feature points
            
        Returns:
            Dictionary with vertex constraints
        """
        # This is a simplified approach - proper implementation would use 
        # perspective projection and closest point mapping
        
        # For each feature point, find the closest mesh vertex in 2D projection
        vertex_constraints = {}
        for i, point in enumerate(feature_points):
            # Project 3D vertices to 2D for comparison (simplified approach)
            # In a real implementation, use proper camera parameters
            vertices_2d = mesh_vertices[:, :2]
            
            # Find the closest vertex to the feature point
            point_coords = point[0]
            distances = np.sqrt(np.sum((vertices_2d - point_coords) ** 2, axis=1))
            closest_vertex_idx = np.argmin(distances)
            
            # Store as a constraint
            vertex_constraints[closest_vertex_idx] = {
                "feature_idx": i,
                "weight": 1.0,  # Weight for importance during mesh deformation
                "position": mesh_vertices[closest_vertex_idx]
            }
        
        return {
            "vertex_constraints": vertex_constraints,
            "constrained_indices": list(vertex_constraints.keys())
        }
    
    def apply_feature_preserving_deformation(self, 
                                           vertices: np.ndarray, 
                                           constraints: Dict) -> np.ndarray:
        """
        Apply feature-preserving deformation to the mesh.
        
        Args:
            vertices: Original mesh vertices
            constraints: Vertex constraints
            
        Returns:
            Deformed vertices with preserved features
        """
        # Create a copy of vertices for deformation
        deformed_vertices = vertices.copy()
        
        # Apply constraints to maintain important features
        for vertex_idx, constraint in constraints["vertex_constraints"].items():
            weight = constraint["weight"]
            position = constraint["position"]
            
            # Force the vertex to maintain its position (weighted by importance)
            deformed_vertices[vertex_idx] = deformed_vertices[vertex_idx] * (1 - weight) + position * weight
        
        # This is a simplified approach - a full implementation would:
        # 1. Create a deformation graph
        # 2. Solve for optimal vertex positions with constraints
        # 3. Use Laplacian mesh deformation or similar techniques
        
        return deformed_vertices
    
    def enhance_feature_regions(self, 
                              texture_image: np.ndarray, 
                              importance_map: np.ndarray,
                              enhancement_factor: float = 1.2) -> np.ndarray:
        """
        Enhance texture in important feature regions.
        
        Args:
            texture_image: Texture image
            importance_map: Feature importance map
            enhancement_factor: Factor for enhancing features
            
        Returns:
            Enhanced texture image
        """
        # Convert to tensors for Kornia operations
        texture_tensor = image_to_tensor(texture_image)
        importance_tensor = torch.from_numpy(importance_map).to(self.device).unsqueeze(0).unsqueeze(0)
        
        # Apply sharpening to the texture
        sharpened = kornia.filters.unsharp_mask(
            texture_tensor.to(self.device),
            kernel_size=(5, 5),
            sigma=(1.0, 1.0)
        )
        
        # Blend original and sharpened based on importance map
        blended = texture_tensor.to(self.device) * (1 - importance_tensor) + \
                 sharpened * importance_tensor * enhancement_factor
        
        # Convert back to numpy
        enhanced_texture = tensor_to_image(blended)
        
        return enhanced_texture
    
    def analyze_feature_fidelity(self, 
                               original_image: np.ndarray, 
                               rendered_image: np.ndarray) -> float:
        """
        Analyze feature fidelity between original and rendered images.
        
        Args:
            original_image: Original input image
            rendered_image: Rendered image of the 3D model
            
        Returns:
            Feature fidelity score (0-1)
        """
        # Extract SIFT features from both images
        sift = cv2.SIFT_create()
        
        # Convert to grayscale
        if original_image.shape[2] == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original_image
            
        if rendered_image.shape[2] == 3:
            rendered_gray = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2GRAY)
        else:
            rendered_gray = rendered_image
        
        # Detect and compute descriptors
        kp1, des1 = sift.detectAndCompute(original_gray, None)
        kp2, des2 = sift.detectAndCompute(rendered_gray, None)
        
        # If no features found, return 0
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return 0.0
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Calculate fidelity score
        fidelity_score = len(good_matches) / max(len(kp1), 1)
        
        # Normalize to 0-1
        return min(fidelity_score, 1.0)
