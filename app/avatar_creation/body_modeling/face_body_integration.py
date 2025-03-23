import os
import numpy as np
import torch
import trimesh
import cv2
from typing import Dict, List, Tuple, Union, Optional
import json

from app.avatar_creation.face_modeling.utils import (
    get_device,
    ensure_directory
)

class FaceBodyIntegrator:
    """
    Class for integrating face and body meshes to create a complete avatar.
    Handles alignment, scaling, and blending of face and body models.
    """
    
    def __init__(self, 
                blend_region_size: float = 0.1,
                preserve_face_details: bool = True,
                smooth_transition: bool = True,
                high_quality: bool = False):
        """
        Initialize the face-body integrator.
        
        Args:
            blend_region_size: Size of the region for blending face and body
            preserve_face_details: Whether to preserve facial details during integration
            smooth_transition: Whether to smooth the transition between face and body
            high_quality: Whether to use high-quality integration
        """
        self.device = get_device()
        self.blend_region_size = blend_region_size
        self.preserve_face_details = preserve_face_details
        self.smooth_transition = smooth_transition
        self.high_quality = high_quality
        
        print(f"Initialized FaceBodyIntegrator (preserve_face_details={preserve_face_details})")
    
    def integrate_face_and_body(self, 
                             face_mesh: trimesh.Trimesh, 
                             body_mesh: trimesh.Trimesh) -> Dict:
        """
        Integrate face and body meshes.
        
        Args:
            face_mesh: Face mesh
            body_mesh: Body mesh
            
        Returns:
            Dictionary containing integrated mesh and associated data
        """
        print("Integrating face and body meshes")
        
        # Calculate face and body bounds
        face_bounds = self._calculate_mesh_bounds(face_mesh)
        body_bounds = self._calculate_mesh_bounds(body_mesh)
        
        # Estimate head position and dimensions in body mesh
        head_height, head_position = self._estimate_head_position_in_body(body_mesh)
        print(f"Estimated head height: {head_height:.2f}, position: {head_position}")
        
        # Calculate scale and translation for face mesh
        face_scale, face_translation = self._calculate_face_transform(
            face_bounds, body_bounds, head_height, head_position)
        
        # Apply transformation to face mesh
        transformed_face = face_mesh.copy()
        transformed_face.apply_scale(face_scale)
        transformed_face.apply_translation(face_translation)
        
        # Check for intersection
        intersection = self._check_mesh_intersection(transformed_face, body_mesh)
        if intersection:
            print(f"Meshes intersect by approximately {intersection:.2f} units")
            
            # Adjust face position to avoid excessive intersection
            if intersection > 0.05 * head_height:
                adjustment = np.array([0, 0, (intersection - 0.05 * head_height)])
                transformed_face.apply_translation(adjustment)
                print(f"Adjusted face position to reduce intersection")
        
        # Create a blend mask for smooth integration
        face_mask = self._create_blend_mask(transformed_face, body_mesh, head_position, head_height)
        
        # Define the target head region in body mesh for replacement
        body_head_region = self._define_head_region(body_mesh, head_position, head_height)
        
        # Check if face and body overlap sufficiently
        overlap = self._check_face_body_overlap(transformed_face, body_mesh, face_mask)
        if not overlap:
            print("Warning: Face and body do not overlap sufficiently")
        
        # Create integrated mesh
        if self.high_quality:
            integrated_mesh = self._high_quality_integration(
                transformed_face, body_mesh, face_mask, head_position, head_height)
        else:
            integrated_mesh = self._basic_integration(
                transformed_face, body_mesh, body_head_region, face_mask)
        
        # Smooth transition region if requested
        if self.smooth_transition:
            integrated_mesh = self._smooth_transition_region(
                integrated_mesh, face_mask, head_position, head_height)
        
        # Create result
        result = {
            'mesh': integrated_mesh,
            'face_mask': face_mask,
            'face_scale': face_scale,
            'face_translation': face_translation,
            'head_position': head_position,
            'head_height': head_height
        }
        
        return result
    
    def integrate_with_textures(self,
                             face_mesh: trimesh.Trimesh,
                             body_mesh: trimesh.Trimesh,
                             face_texture: Optional[np.ndarray] = None,
                             body_texture: Optional[np.ndarray] = None) -> Dict:
        """
        Integrate face and body meshes with textures.
        
        Args:
            face_mesh: Face mesh
            body_mesh: Body mesh
            face_texture: Optional face texture
            body_texture: Optional body texture
            
        Returns:
            Dictionary containing integrated mesh and texture
        """
        # Check for UV coordinates
        face_has_uv = (hasattr(face_mesh.visual, 'uv') and face_mesh.visual.uv is not None)
        body_has_uv = (hasattr(body_mesh.visual, 'uv') and body_mesh.visual.uv is not None)
        
        if not (face_has_uv and body_has_uv):
            print("UV coordinates required for texture integration")
            
            # Generate UV coordinates if needed
            if not face_has_uv:
                print("Generating UV coordinates for face mesh")
                face_mesh = self._generate_uv_coordinates(face_mesh)
            
            if not body_has_uv:
                print("Generating UV coordinates for body mesh")
                body_mesh = self._generate_uv_coordinates(body_mesh)
        
        # Perform basic mesh integration
        integration_result = self.integrate_face_and_body(face_mesh, body_mesh)
        integrated_mesh = integration_result['mesh']
        face_mask = integration_result['face_mask']
        
        # Handle textures
        if face_texture is not None and body_texture is not None:
            # Integrate textures
            integrated_texture = self._integrate_textures(
                integrated_mesh, face_texture, body_texture, face_mask)
            
            # Apply texture to mesh
            integrated_mesh.visual.material.image = integrated_texture
            
            # Add texture to result
            integration_result['texture'] = integrated_texture
        
        return integration_result
    
    def _calculate_mesh_bounds(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Calculate mesh bounds.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Array of min and max bounds
        """
        # Extract bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        return bounds
    
    def _estimate_head_position_in_body(self, body_mesh: trimesh.Trimesh) -> Tuple[float, np.ndarray]:
        """
        Estimate head position and dimensions in the body mesh.
        
        Args:
            body_mesh: Body mesh
            
        Returns:
            Tuple of (head_height, head_position)
        """
        # Get mesh bounds
        bounds = self._calculate_mesh_bounds(body_mesh)
        
        # Extract dimensions
        height = bounds[1, 1] - bounds[0, 1]
        width = bounds[1, 0] - bounds[0, 0]
        
        # Estimate head height as a proportion of body height
        head_height = height * 0.15  # Typical head height
        
        # Estimate head position (top of the mesh)
        head_position = np.array([
            (bounds[0, 0] + bounds[1, 0]) / 2,  # Center X
            bounds[1, 1] - head_height / 2,     # Top Y (minus half head height)
            (bounds[0, 2] + bounds[1, 2]) / 2   # Center Z
        ])
        
        return head_height, head_position
    
    def _calculate_face_transform(self, face_bounds: np.ndarray, 
                               body_bounds: np.ndarray,
                               head_height: float,
                               head_position: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate scale and translation for face mesh.
        
        Args:
            face_bounds: Face mesh bounds
            body_bounds: Body mesh bounds
            head_height: Estimated head height
            head_position: Estimated head position
            
        Returns:
            Tuple of (scale_factor, translation)
        """
        # Calculate face dimensions
        face_height = face_bounds[1, 1] - face_bounds[0, 1]
        face_center = (face_bounds[0] + face_bounds[1]) / 2
        
        # Calculate scale to match estimated head height
        scale_factor = head_height / face_height
        
        # Calculate translation to position face at head position
        translation = head_position - face_center * scale_factor
        
        return scale_factor, translation
    
    def _check_mesh_intersection(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> float:
        """
        Check if two meshes intersect and estimate the intersection amount.
        
        Args:
            mesh1: First mesh
            mesh2: Second mesh
            
        Returns:
            Intersection amount (0 if no intersection)
        """
        # This is a simplified approach
        # In a real implementation, you would use a proper mesh intersection algorithm
        
        # Check for intersecting bounding boxes
        bounds1 = self._calculate_mesh_bounds(mesh1)
        bounds2 = self._calculate_mesh_bounds(mesh2)
        
        # Check for intersection in each dimension
        intersection = np.zeros(3)
        for i in range(3):
            min_upper = min(bounds1[1, i], bounds2[1, i])
            max_lower = max(bounds1[0, i], bounds2[0, i])
            intersection[i] = max(0, min_upper - max_lower)
        
        # Return the depth of intersection in the forward direction (Z)
        return intersection[2]
    
    def _create_blend_mask(self, face_mesh: trimesh.Trimesh, 
                        body_mesh: trimesh.Trimesh,
                        head_position: np.ndarray,
                        head_height: float) -> np.ndarray:
        """
        Create a blend mask for face-body integration.
        
        Args:
            face_mesh: Transformed face mesh
            body_mesh: Body mesh
            head_position: Estimated head position
            head_height: Estimated head height
            
        Returns:
            Blend mask for vertices
        """
        # Calculate blend region dimensions
        blend_radius = head_height * self.blend_region_size
        
        # Create mask for face mesh vertices
        face_mask = np.zeros(len(face_mesh.vertices))
        
        # Define neck position (approximate)
        neck_position = head_position.copy()
        neck_position[1] -= head_height * 0.8  # Move down from head center
        
        # Calculate distance from each vertex to neck position
        for i, vertex in enumerate(face_mesh.vertices):
            # Calculate distance from vertex to neck plane
            dist_to_neck = vertex[1] - neck_position[1]
            
            if dist_to_neck > blend_radius:
                # Above blend region - fully face
                face_mask[i] = 1.0
            elif dist_to_neck > 0:
                # In blend region - gradual transition
                face_mask[i] = dist_to_neck / blend_radius
            else:
                # Below blend region - fully body
                face_mask[i] = 0.0
        
        return face_mask
    
    def _define_head_region(self, body_mesh: trimesh.Trimesh, 
                        head_position: np.ndarray,
                        head_height: float) -> np.ndarray:
        """
        Define the head region in the body mesh to be replaced.
        
        Args:
            body_mesh: Body mesh
            head_position: Estimated head position
            head_height: Estimated head height
            
        Returns:
            Mask for head region vertices
        """
        # Calculate head region bounds
        # Horizontal radius based on head height
        horizontal_radius = head_height * 0.7  # Slightly wider than height
        
        # Vertical bound for head region
        upper_bound = head_position[1] + head_height * 0.6  # Above head center
        lower_bound = head_position[1] - head_height * 0.8  # Below head (neck)
        
        # Create mask for body mesh vertices
        head_region_mask = np.zeros(len(body_mesh.vertices), dtype=bool)
        
        # Identify vertices in head region
        for i, vertex in enumerate(body_mesh.vertices):
            # Check if vertex is within vertical bounds
            if vertex[1] <= upper_bound and vertex[1] >= lower_bound:
                # Calculate horizontal distance from head center
                dx = vertex[0] - head_position[0]
                dz = vertex[2] - head_position[2]
                horizontal_dist = np.sqrt(dx**2 + dz**2)
                
                # Check if within horizontal bounds
                if horizontal_dist <= horizontal_radius:
                    head_region_mask[i] = True
        
        return head_region_mask
    
    def _check_face_body_overlap(self, face_mesh: trimesh.Trimesh, 
                               body_mesh: trimesh.Trimesh,
                               face_mask: np.ndarray) -> bool:
        """
        Check if face and body mesh overlap sufficiently for integration.
        
        Args:
            face_mesh: Transformed face mesh
            body_mesh: Body mesh
            face_mask: Blend mask for face vertices
            
        Returns:
            True if overlap is sufficient, False otherwise
        """
        # Get blend region vertices (where mask is between 0 and 1)
        blend_vertices = face_mesh.vertices[(face_mask > 0) & (face_mask < 1)]
        
        if len(blend_vertices) == 0:
            return False
        
        # Create a bounding box for blend region
        blend_min = blend_vertices.min(axis=0)
        blend_max = blend_vertices.max(axis=0)
        
        # Check if any body vertices are within blend region
        body_in_blend = False
        for vertex in body_mesh.vertices:
            if (vertex >= blend_min).all() and (vertex <= blend_max).all():
                body_in_blend = True
                break
        
        return body_in_blend
    
    def _basic_integration(self, face_mesh: trimesh.Trimesh, 
                        body_mesh: trimesh.Trimesh,
                        body_head_region: np.ndarray,
                        face_mask: np.ndarray) -> trimesh.Trimesh:
        """
        Perform basic integration of face and body meshes.
        
        Args:
            face_mesh: Transformed face mesh
            body_mesh: Body mesh
            body_head_region: Mask for head region in body mesh
            face_mask: Blend mask for face vertices
            
        Returns:
            Integrated mesh
        """
        # Create copies of the meshes
        face_mesh = face_mesh.copy()
        body_mesh = body_mesh.copy()
        
        # Remove head region from body mesh
        body_vertices = body_mesh.vertices[~body_head_region]
        body_faces = []
        
        # Keep only faces that don't contain any head region vertices
        for face in body_mesh.faces:
            if not any(body_head_region[face]):
                # Adjust face indices to account for removed vertices
                new_face = [np.sum(~body_head_region[:idx]) for idx in face]
                body_faces.append(new_face)
        
        body_faces = np.array(body_faces)
        
        # Keep face mesh vertices with non-zero mask
        face_vertices = face_mesh.vertices[face_mask > 0]
        face_mask_filtered = face_mask[face_mask > 0]
        
        # Keep only faces with all vertices in the kept region
        face_vertex_mask = face_mask > 0
        face_faces = []
        
        for face in face_mesh.faces:
            if all(face_vertex_mask[face]):
                # Adjust face indices to account for removed vertices
                new_face = [np.sum(face_vertex_mask[:idx]) for idx in face]
                face_faces.append(new_face)
        
        face_faces = np.array(face_faces)
        
        # Combine vertices and faces
        combined_vertices = np.vstack((body_vertices, face_vertices))
        
        # Adjust face indices for face mesh
        face_faces += len(body_vertices)
        
        combined_faces = np.vstack((body_faces, face_faces))
        
        # Create combined mesh
        combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
        
        return combined_mesh
    
    def _high_quality_integration(self, face_mesh: trimesh.Trimesh, 
                               body_mesh: trimesh.Trimesh,
                               face_mask: np.ndarray,
                               head_position: np.ndarray,
                               head_height: float) -> trimesh.Trimesh:
        """
        Perform high-quality integration of face and body meshes.
        
        Args:
            face_mesh: Transformed face mesh
            body_mesh: Body mesh
            face_mask: Blend mask for face vertices
            head_position: Estimated head position
            head_height: Estimated head height
            
        Returns:
            Integrated mesh
        """
        # This is a more sophisticated version of integration
        # that could use mesh editing tools like Laplacian deformation
        
        print("Using high-quality integration")
        
        # For now, we'll implement a slightly improved version of basic integration
        
        # Define blend region bounds
        upper_blend = head_position[1] - head_height * 0.3  # Upper part of neck
        lower_blend = head_position[1] - head_height * 0.8  # Lower part of neck
        
        # Get vertices and faces for both meshes
        face_vertices = face_mesh.vertices.copy()
        face_faces = face_mesh.faces.copy()
        
        body_vertices = body_mesh.vertices.copy()
        body_faces = body_mesh.faces.copy()
        
        # Create body mask (opposite of face mask)
        body_mask = np.ones(len(body_vertices))
        
        # Calculate distance from each body vertex to neck region
        for i, vertex in enumerate(body_vertices):
            if vertex[1] >= upper_blend:
                # Above blend region - fully face
                body_mask[i] = 0.0
            elif vertex[1] <= lower_blend:
                # Below blend region - fully body
                body_mask[i] = 1.0
            else:
                # In blend region - gradual transition
                t = (vertex[1] - lower_blend) / (upper_blend - lower_blend)
                body_mask[i] = 1.0 - t
        
        # Blend vertices in transition region
        blend_indices_face = (face_mask < 1.0) & (face_mask > 0.0)
        blend_indices_body = (body_mask < 1.0) & (body_mask > 0.0)
        
        # Create point clouds for closest point queries
        face_blend_points = face_vertices[blend_indices_face]
        body_blend_points = body_vertices[blend_indices_body]
        
        if len(face_blend_points) > 0 and len(body_blend_points) > 0:
            # Create KD-trees for both point sets
            try:
                from scipy.spatial import cKDTree
                
                face_tree = cKDTree(face_blend_points)
                body_tree = cKDTree(body_blend_points)
                
                # For each blend point in face, find closest in body
                for i, idx in enumerate(np.where(blend_indices_face)[0]):
                    dist, closest = body_tree.query(face_vertices[idx])
                    if dist < head_height * 0.2:  # Only blend if close enough
                        closest_body_point = body_blend_points[closest]
                        # Blend based on mask value
                        blend_factor = face_mask[idx]
                        face_vertices[idx] = (blend_factor * face_vertices[idx] + 
                                              (1 - blend_factor) * closest_body_point)
                
                # For each blend point in body, find closest in face
                for i, idx in enumerate(np.where(blend_indices_body)[0]):
                    dist, closest = face_tree.query(body_vertices[idx])
                    if dist < head_height * 0.2:  # Only blend if close enough
                        closest_face_point = face_blend_points[closest]
                        # Blend based on mask value
                        blend_factor = body_mask[idx]
                        body_vertices[idx] = (blend_factor * body_vertices[idx] + 
                                              (1 - blend_factor) * closest_face_point)
            
            except ImportError:
                print("SciPy not available, falling back to basic blending")
        
        # Combine vertices and faces
        # Remove body vertices that are fully replaced by face (mask = 0)
        keep_body_vertices = body_mask > 0.01
        kept_body_vertices = body_vertices[keep_body_vertices]
        
        # Create mapping for body vertex indices
        body_vertex_map = np.cumsum(keep_body_vertices) - 1
        
        # Filter body faces
        kept_body_faces = []
        for face in body_faces:
            # Keep face if all vertices are kept
            if all(keep_body_vertices[face]):
                # Remap vertex indices
                new_face = [body_vertex_map[idx] for idx in face]
                kept_body_faces.append(new_face)
        
        kept_body_faces = np.array(kept_body_faces)
        
        # Remove face vertices that are fully replaced by body (mask = 0)
        keep_face_vertices = face_mask > 0.01
        kept_face_vertices = face_vertices[keep_face_vertices]
        
        # Create mapping for face vertex indices
        face_vertex_map = np.cumsum(keep_face_vertices) - 1
        
        # Filter face faces
        kept_face_faces = []
        for face in face_faces:
            # Keep face if all vertices are kept
            if all(keep_face_vertices[face]):
                # Remap vertex indices
                new_face = [face_vertex_map[idx] for idx in face]
                kept_face_faces.append(new_face)
        
        kept_face_faces = np.array(kept_face_faces)
        
        # Adjust face indices for combining
        kept_face_faces += len(kept_body_vertices)
        
        # Combine vertices and faces
        combined_vertices = np.vstack((kept_body_vertices, kept_face_vertices))
        combined_faces = np.vstack((kept_body_faces, kept_face_faces)) if len(kept_body_faces) > 0 and len(kept_face_faces) > 0 else np.array([])
        
        if len(combined_faces) == 0:
            print("Warning: No faces in combined mesh - using fallback")
            return self._basic_integration(face_mesh, body_mesh, 
                                      np.zeros(len(body_mesh.vertices), dtype=bool), 
                                      face_mask)
        
        # Create combined mesh
        combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
        
        return combined_mesh
    
    def _smooth_transition_region(self, mesh: trimesh.Trimesh, 
                               face_mask: np.ndarray,
                               head_position: np.ndarray,
                               head_height: float) -> trimesh.Trimesh:
        """
        Smooth the transition region between face and body.
        
        Args:
            mesh: Integrated mesh
            face_mask: Blend mask for face vertices
            head_position: Estimated head position
            head_height: Estimated head height
            
        Returns:
            Smoothed mesh
        """
        # This is a placeholder for a proper mesh smoothing algorithm
        # In a real implementation, you would use Laplacian smoothing
        # or other mesh processing techniques
        
        # For demonstration, we'll implement a simple smoothing approach
        try:
            # Define smoothing region bounds
            upper_bound = head_position[1] - head_height * 0.3  # Upper part of neck
            lower_bound = head_position[1] - head_height * 0.8  # Lower part of neck
            
            # Identify vertices in smoothing region
            smoothing_region = np.zeros(len(mesh.vertices), dtype=bool)
            for i, vertex in enumerate(mesh.vertices):
                if vertex[1] <= upper_bound and vertex[1] >= lower_bound:
                    # In neck region
                    horizontal_dist = np.sqrt((vertex[0] - head_position[0])**2 + 
                                            (vertex[2] - head_position[2])**2)
                    if horizontal_dist < head_height * 0.5:
                        smoothing_region[i] = True
            
            # Get adjacency information
            adjacency = mesh.vertex_adjacency
            
            # Apply smoothing iterations
            smoothing_steps = 5
            smoothed_vertices = mesh.vertices.copy()
            
            for _ in range(smoothing_steps):
                new_positions = smoothed_vertices.copy()
                
                # For each vertex in smoothing region
                for i in np.where(smoothing_region)[0]:
                    # Get adjacent vertices
                    adj_vertices = adjacency[i]
                    
                    if len(adj_vertices) > 0:
                        # Calculate average position of neighbors
                        avg_position = np.mean(smoothed_vertices[list(adj_vertices)], axis=0)
                        
                        # Blend current position with average
                        blend_factor = 0.5  # 50% blend
                        new_positions[i] = (1 - blend_factor) * smoothed_vertices[i] + blend_factor * avg_position
                
                smoothed_vertices = new_positions
            
            # Create smoothed mesh
            smoothed_mesh = mesh.copy()
            smoothed_mesh.vertices = smoothed_vertices
            
            return smoothed_mesh
            
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return mesh
    
    def _generate_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Generate UV coordinates for a mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Mesh with UV coordinates
        """
        # Try to use xatlas if available
        try:
            import xatlas
            
            # Unwrap mesh using xatlas
            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
            
            # Create new mesh with UV coordinates
            new_mesh = trimesh.Trimesh(
                vertices=mesh.vertices[vmapping],
                faces=indices,
                process=False
            )
            
            # Apply UV coordinates
            new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
            
            return new_mesh
            
        except ImportError:
            print("xatlas not available, using fallback UV mapping")
        
        # Fallback: Generate cylindrical UV mapping
        # This is a simplified approach
        
        # Get mesh bounds
        bounds = self._calculate_mesh_bounds(mesh)
        height = bounds[1, 1] - bounds[0, 1]
        
        # Create UV coordinates
        uv = np.zeros((len(mesh.vertices), 2))
        
        for i, vertex in enumerate(mesh.vertices):
            # Normalize height
            v = (vertex[1] - bounds[0, 1]) / height
            
            # Calculate angle around vertical axis
            angle = np.arctan2(vertex[2] - (bounds[0, 2] + bounds[1, 2]) / 2,
                             vertex[0] - (bounds[0, 0] + bounds[1, 0]) / 2)
            
            # Convert angle to U coordinate
            u = (angle + np.pi) / (2 * np.pi)
            
            # Set UV coordinates
            uv[i] = [u, 1 - v]  # Invert V to match typical texture coordinates
        
        # Create new mesh with UV coordinates
        new_mesh = mesh.copy()
        new_mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
        
        return new_mesh
    
    def _integrate_textures(self, mesh: trimesh.Trimesh,
                        face_texture: np.ndarray,
                        body_texture: np.ndarray,
                        face_mask: np.ndarray) -> np.ndarray:
        """
        Integrate face and body textures.
        
        Args:
            mesh: Integrated mesh
            face_texture: Face texture
            body_texture: Body texture
            face_mask: Blend mask for face vertices
            
        Returns:
            Integrated texture
        """
        # Check if mesh has UV coordinates
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            print("Mesh needs UV coordinates for texture integration")
            return body_texture
        
        # Resize textures to the same dimensions if needed
        target_size = max(face_texture.shape[:2] + body_texture.shape[:2])
        
        # Resize face texture if needed
        if face_texture.shape[0] != target_size or face_texture.shape[1] != target_size:
            face_texture = cv2.resize(face_texture, (target_size, target_size))
        
        # Resize body texture if needed
        if body_texture.shape[0] != target_size or body_texture.shape[1] != target_size:
            body_texture = cv2.resize(body_texture, (target_size, target_size))
        
        # Create output texture
        integrated_texture = body_texture.copy()
        
        # Get UV coordinates
        uv = mesh.visual.uv
        
        # Create UV face mask
        uv_face_mask = np.zeros((target_size, target_size))
        
        # Map face mask to UV space
        for i, mask_value in enumerate(face_mask):
            if i < len(uv):
                u, v = uv[i]
                x = int(u * (target_size - 1))
                y = int(v * (target_size - 1))
                
                if x >= 0 and x < target_size and y >= 0 and y < target_size:
                    uv_face_mask[y, x] = mask_value
        
        # Smooth the mask
        uv_face_mask = cv2.GaussianBlur(uv_face_mask, (15, 15), 0)
        
        # Blend textures
        for y in range(target_size):
            for x in range(target_size):
                blend_factor = uv_face_mask[y, x]
                if blend_factor > 0:
                    integrated_texture[y, x] = ((1 - blend_factor) * body_texture[y, x] + 
                                               blend_factor * face_texture[y, x])
        
        return integrated_texture
    
    def save_integrated_avatar(self, avatar_data: Dict, output_dir: str) -> Dict:
        """
        Save integrated avatar data to files.
        
        Args:
            avatar_data: Dictionary containing integrated avatar data
            output_dir: Directory to save files
            
        Returns:
            Dictionary with paths to saved files
        """
        # Create output directory
        ensure_directory(output_dir)
        
        # Get mesh
        mesh = avatar_data['mesh']
        
        # Save mesh
        obj_path = os.path.join(output_dir, 'integrated_avatar.obj')
        mesh.export(obj_path)
        
        # Save texture if available
        texture_path = None
        if 'texture' in avatar_data:
            texture_path = os.path.join(output_dir, 'integrated_texture.png')
            cv2.imwrite(texture_path, avatar_data['texture'])
        
        # Save metadata
        metadata = {
            'head_position': avatar_data['head_position'].tolist(),
            'head_height': float(avatar_data['head_height']),
            'face_scale': float(avatar_data['face_scale']),
            'face_translation': avatar_data['face_translation'].tolist()
        }
        
        metadata_path = os.path.join(output_dir, 'avatar_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Return paths
        result = {
            'mesh_path': obj_path,
            'texture_path': texture_path,
            'metadata_path': metadata_path
        }
        
        print(f"Saved integrated avatar to: {output_dir}")
        
        return result
    
    def create_avatar_preview(self, avatar_data: Dict, output_path: str, 
                           size: Tuple[int, int] = (800, 800)) -> str:
        """
        Create a preview image of the integrated avatar.
        
        Args:
            avatar_data: Dictionary containing integrated avatar data
            output_path: Path to save the preview image
            size: Size of the preview image
            
        Returns:
            Path to the saved preview
        """
        # Get mesh
        mesh = avatar_data['mesh']
        
        try:
            # Create scene for rendering
            scene = trimesh.Scene(mesh)
            
            # Set camera parameters
            angle = np.pi / 6  # 30 degrees
            camera_position = mesh.centroid + np.array([0, 0, 2]) * mesh.scale
            
            # Render scene
            try:
                # Try using pyrender for better quality
                import pyrender
                
                # Create pyrender scene
                pyrender_scene = pyrender.Scene()
                
                # Add mesh to scene
                pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
                pyrender_scene.add(pyrender_mesh)
                
                # Add lighting
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
                pyrender_scene.add(light, pose=np.eye(4))
                
                # Create camera
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=1.0)
                
                # Place camera
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = camera_position
                pyrender_scene.add(camera, pose=camera_pose)
                
                # Render
                renderer = pyrender.OffscreenRenderer(size[0], size[1])
                color, _ = renderer.render(pyrender_scene)
                
                # Save image
                preview = color
                
            except ImportError:
                # Fall back to trimesh rendering
                preview = scene.save_image(resolution=size, visible=True)
            
            # Save preview
            ensure_directory(os.path.dirname(output_path))
            if isinstance(preview, np.ndarray):
                cv2.imwrite(output_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            else:
                with open(output_path, 'wb') as f:
                    f.write(preview)
            
            print(f"Saved avatar preview to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating avatar preview: {e}")
            return "" 