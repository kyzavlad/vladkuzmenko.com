import os
import numpy as np
import cv2
import torch
import trimesh
from typing import Dict, List, Tuple, Union, Optional
import json

from app.avatar_creation.face_modeling.utils import (
    get_device,
    ensure_directory,
    load_image,
    save_image
)

class BodyTextureMapper:
    """
    Class for generating and mapping textures to body meshes.
    Supports both image-based texture projection and procedural texture generation.
    """
    
    def __init__(self, 
                resolution: Tuple[int, int] = (2048, 2048),
                use_high_quality: bool = False,
                use_pbr: bool = False,
                default_texture_dir: Optional[str] = None):
        """
        Initialize the body texture mapper.
        
        Args:
            resolution: Resolution of the generated textures
            use_high_quality: Whether to use high-quality settings
            use_pbr: Whether to generate PBR textures (albedo, normal, roughness, etc.)
            default_texture_dir: Directory with default textures
        """
        self.device = get_device()
        self.resolution = resolution
        self.use_high_quality = use_high_quality
        self.use_pbr = use_pbr
        self.default_texture_dir = default_texture_dir
        
        print(f"Initialized BodyTextureMapper (resolution={resolution}, pbr={use_pbr})")
    
    def generate_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Generate UV coordinates for a body mesh.
        
        Args:
            mesh: Input body mesh
            
        Returns:
            Body mesh with UV coordinates
        """
        # Check if mesh already has UV coordinates
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            print("Mesh already has UV coordinates")
            return mesh
        
        print("Generating UV coordinates for mesh")
        
        # Try to use xatlas if available for high-quality UV unwrapping
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
            
            print("Generated UV coordinates using xatlas")
            return new_mesh
            
        except ImportError:
            print("xatlas not available, using fallback UV mapping")
        
        # Fallback: generate simplified cylindrical UV mapping
        vertices = mesh.vertices.copy()
        
        # Get mesh bounds
        bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
        height = bounds[1, 1] - bounds[0, 1]
        
        # Center of the mesh
        center_x = (bounds[0, 0] + bounds[1, 0]) / 2
        center_z = (bounds[0, 2] + bounds[1, 2]) / 2
        
        # Normalize height
        y_normalized = (vertices[:, 1] - bounds[0, 1]) / height
        
        # Calculate angles for cylindrical mapping
        rel_x = vertices[:, 0] - center_x
        rel_z = vertices[:, 2] - center_z
        
        # Calculate angle from center
        angles = np.arctan2(rel_z, rel_x)
        
        # Normalize angles to [0, 1]
        u = (angles + np.pi) / (2 * np.pi)
        
        # Y coordinate is directly mapped to v
        v = 1.0 - y_normalized  # Invert to match typical texture coordinates
        
        # Create UV coordinates array
        uv = np.column_stack((u, v))
        
        # Create a copy of the mesh with UVs
        mesh_with_uv = mesh.copy()
        mesh_with_uv.visual = trimesh.visual.TextureVisuals(uv=uv)
        
        return mesh_with_uv
    
    def project_image_to_texture(self, 
                              mesh: trimesh.Trimesh, 
                              image_paths: List[str],
                              camera_matrices: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Project multiple images onto a mesh to create a texture.
        
        Args:
            mesh: Body mesh with UV coordinates
            image_paths: List of paths to input images
            camera_matrices: List of camera matrices for each image
            
        Returns:
            Texture image
        """
        print(f"Projecting {len(image_paths)} images to texture")
        
        # Check if mesh has UV coordinates
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            print("Mesh needs UV coordinates for texture projection")
            mesh = self.generate_uv_coordinates(mesh)
        
        # Load images
        images = []
        for path in image_paths:
            img = load_image(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image: {path}")
        
        if len(images) == 0:
            print("No valid images provided, generating default texture")
            return self.generate_default_texture(mesh)
        
        # Create an empty texture
        texture = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Create a weight map to blend multiple projections
        weight_map = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        
        # Get UV coordinates
        uvs = mesh.visual.uv
        
        # If camera matrices are not provided, estimate them
        if camera_matrices is None:
            camera_matrices = self._estimate_camera_matrices(mesh, len(images))
        
        # Project each image onto the texture
        for i, (image, camera_matrix) in enumerate(zip(images, camera_matrices)):
            # Project vertices to image space
            projected_texture, proj_weights = self._project_vertices_to_image(
                mesh.vertices, mesh.faces, uvs, image, camera_matrix
            )
            
            # Accumulate texture and weights
            valid_mask = proj_weights > 0
            texture[valid_mask] += projected_texture[valid_mask] * proj_weights[valid_mask, np.newaxis]
            weight_map[valid_mask] += proj_weights[valid_mask]
        
        # Normalize by weights
        valid_weights = weight_map > 0
        if np.any(valid_weights):
            texture[valid_weights] = texture[valid_weights] / weight_map[valid_weights, np.newaxis]
        
        # Fill in missing parts with default texture
        if np.any(weight_map == 0):
            default_texture = self.generate_default_texture(mesh)
            texture[weight_map == 0] = default_texture[weight_map == 0]
        
        return texture
    
    def _project_vertices_to_image(self, 
                                vertices: np.ndarray,
                                faces: np.ndarray,
                                uvs: np.ndarray,
                                image: np.ndarray,
                                camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project vertices to image space and sample texture.
        
        Args:
            vertices: Mesh vertices
            faces: Mesh faces
            uvs: UV coordinates
            image: Input image
            camera_matrix: Camera matrix
            
        Returns:
            Tuple of (projected_texture, weights)
        """
        # This is a simplified implementation
        # In a real implementation, you would use more accurate visibility testing
        
        # Initialize projected texture and weights
        projected_texture = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.float32)
        weights = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Project vertices to image space
        homogeneous_vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
        projected_vertices = np.dot(homogeneous_vertices, camera_matrix.T)
        
        # Normalize by dividing by w
        projected_vertices[:, :2] /= projected_vertices[:, 2:3]
        
        # Check which vertices are visible (in front of camera and within image bounds)
        visible = (projected_vertices[:, 2] > 0) & \
                 (projected_vertices[:, 0] >= 0) & (projected_vertices[:, 0] < img_width) & \
                 (projected_vertices[:, 1] >= 0) & (projected_vertices[:, 1] < img_height)
        
        # Convert UVs to texture coordinates
        tex_coords = np.zeros((len(uvs), 2), dtype=int)
        tex_coords[:, 0] = np.clip(uvs[:, 0] * (self.resolution[0] - 1), 0, self.resolution[0] - 1)
        tex_coords[:, 1] = np.clip(uvs[:, 1] * (self.resolution[1] - 1), 0, self.resolution[1] - 1)
        
        # For each face
        for face in faces:
            # Check if all vertices are visible
            if not all(visible[face]):
                continue
            
            # Get image coordinates for face vertices
            img_coords = projected_vertices[face, :2].astype(int)
            
            # Get texture coordinates for face vertices
            face_tex_coords = tex_coords[face]
            
            # Simple face visibility test based on depth
            face_depth = np.mean(projected_vertices[face, 2])
            
            # Calculate face normal for back-face culling
            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            
            # Get camera direction (simplified)
            camera_pos = np.zeros(3)  # Assume camera at origin in camera space
            face_center = np.mean(vertices[face], axis=0)
            view_dir = camera_pos - face_center
            view_dir = view_dir / np.linalg.norm(view_dir)
            
            # Check if face is facing towards camera
            if np.dot(normal, view_dir) <= 0:
                continue  # Back-face, skip
            
            # Sample colors from image
            for i in range(3):  # For each vertex in the face
                img_x, img_y = img_coords[i]
                tex_x, tex_y = face_tex_coords[i]
                
                # Check bounds
                if 0 <= img_x < img_width and 0 <= img_y < img_height and \
                   0 <= tex_x < self.resolution[0] and 0 <= tex_y < self.resolution[1]:
                    # Sample color
                    color = image[img_y, img_x]
                    
                    # Calculate weight based on depth and angle
                    weight = 1.0 / (1.0 + face_depth * 0.1)
                    
                    # Update projected texture and weight
                    if weight > weights[tex_y, tex_x]:
                        projected_texture[tex_y, tex_x] = color
                        weights[tex_y, tex_x] = weight
        
        return projected_texture, weights
    
    def _estimate_camera_matrices(self, mesh: trimesh.Trimesh, num_cameras: int) -> List[np.ndarray]:
        """
        Estimate camera matrices for texture projection.
        
        Args:
            mesh: Body mesh
            num_cameras: Number of cameras to generate
            
        Returns:
            List of camera matrices
        """
        # This is a simplified implementation
        # In a real implementation, you would use camera calibration or SfM
        
        # Get mesh center and dimensions
        center = mesh.centroid
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        dimensions = bounds[1] - bounds[0]
        
        # Calculate camera distance based on mesh size
        camera_distance = np.max(dimensions) * 2.0
        
        # Generate camera positions around the mesh
        camera_matrices = []
        
        for i in range(num_cameras):
            # Calculate angle
            angle = i * (2 * np.pi / num_cameras)
            
            # Calculate camera position
            cam_x = center[0] + np.cos(angle) * camera_distance
            cam_y = center[1]  # Same height as center
            cam_z = center[2] + np.sin(angle) * camera_distance
            camera_pos = np.array([cam_x, cam_y, cam_z])
            
            # Look-at matrix
            forward = center - camera_pos
            forward = forward / np.linalg.norm(forward)
            
            # Up vector (assuming Y is up)
            up = np.array([0, 1, 0])
            
            # Calculate right vector
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            # Recalculate up
            up = np.cross(right, forward)
            
            # Create view matrix
            view_matrix = np.zeros((4, 4))
            view_matrix[:3, 0] = right
            view_matrix[:3, 1] = up
            view_matrix[:3, 2] = -forward
            view_matrix[:3, 3] = camera_pos
            view_matrix[3, 3] = 1.0
            
            # Invert view matrix
            view_matrix_inv = np.linalg.inv(view_matrix)
            
            # Create projection matrix (perspective)
            fov = 60 * np.pi / 180  # 60 degrees
            aspect = 1.0  # Assuming square images
            znear = 0.1
            zfar = camera_distance * 4
            
            proj_matrix = np.zeros((4, 4))
            proj_matrix[0, 0] = 1.0 / np.tan(fov / 2) / aspect
            proj_matrix[1, 1] = 1.0 / np.tan(fov / 2)
            proj_matrix[2, 2] = -(zfar + znear) / (zfar - znear)
            proj_matrix[2, 3] = -2.0 * zfar * znear / (zfar - znear)
            proj_matrix[3, 2] = -1.0
            
            # Combined MVP matrix
            mvp_matrix = np.dot(proj_matrix, view_matrix_inv)
            
            camera_matrices.append(mvp_matrix)
        
        return camera_matrices
    
    def generate_default_texture(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Generate a default texture for a body mesh.
        
        Args:
            mesh: Body mesh with UV coordinates
            
        Returns:
            Default texture image
        """
        print("Generating default texture")
        
        # Check if default texture directory exists
        if self.default_texture_dir and os.path.exists(self.default_texture_dir):
            # Try to load a pre-made texture
            default_texture_path = os.path.join(self.default_texture_dir, "default_body_texture.png")
            if os.path.exists(default_texture_path):
                default_texture = load_image(default_texture_path)
                
                if default_texture is not None:
                    # Resize to target resolution
                    if default_texture.shape[:2] != self.resolution:
                        default_texture = cv2.resize(default_texture, self.resolution)
                    
                    return default_texture
        
        # Create a procedural texture
        return self.generate_procedural_texture(mesh)
    
    def generate_procedural_texture(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Generate a procedural texture for a body mesh.
        
        Args:
            mesh: Body mesh with UV coordinates
            
        Returns:
            Procedural texture image
        """
        print("Generating procedural texture")
        
        # Create a base color
        base_color = np.array([233, 214, 197])  # Light skin tone
        
        # Create texture with base color
        texture = np.ones((self.resolution[1], self.resolution[0], 3), dtype=np.uint8) * base_color
        
        # Add some variation
        noise = np.random.normal(0, 5, texture.shape).astype(np.int8)
        texture = np.clip(texture + noise, 0, 255).astype(np.uint8)
        
        # Create a gradient from top to bottom (slightly darker at the bottom)
        gradient = np.linspace(1.0, 0.95, self.resolution[1])
        gradient = np.tile(gradient[:, np.newaxis, np.newaxis], (1, self.resolution[0], 3))
        texture = (texture * gradient).astype(np.uint8)
        
        # Add some basic anatomical details
        if self.use_high_quality:
            texture = self._add_anatomical_details(texture, mesh)
        
        return texture
    
    def _add_anatomical_details(self, texture: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Add basic anatomical details to a procedural texture.
        
        Args:
            texture: Base texture
            mesh: Body mesh with UV coordinates
            
        Returns:
            Texture with anatomical details
        """
        # This is a simplified implementation
        # In a real implementation, you would use more sophisticated techniques
        
        # Get texture dimensions
        height, width = texture.shape[:2]
        
        # Create a copy of the texture
        detailed_texture = texture.copy()
        
        # Define regions for different details (simplified)
        regions = {
            'chest': (int(width * 0.3), int(width * 0.7), int(height * 0.2), int(height * 0.3)),
            'stomach': (int(width * 0.3), int(width * 0.7), int(height * 0.3), int(height * 0.5)),
            'face': (int(width * 0.4), int(width * 0.6), int(height * 0.05), int(height * 0.15))
        }
        
        # Add some subtle shading to chest
        chest_x1, chest_x2, chest_y1, chest_y2 = regions['chest']
        chest_color = np.array([225, 206, 187])  # Slightly darker
        
        chest_mask = np.zeros((height, width), dtype=np.float32)
        for y in range(chest_y1, chest_y2):
            for x in range(chest_x1, chest_x2):
                # Create a circular mask for chest
                dx = (x - (chest_x1 + chest_x2) / 2) / (chest_x2 - chest_x1) * 2
                dy = (y - chest_y1) / (chest_y2 - chest_y1) * 2
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < 1.0:
                    chest_mask[y, x] = 1.0 - dist
        
        # Apply chest shading
        for c in range(3):
            detailed_texture[:, :, c] = (
                detailed_texture[:, :, c] * (1 - chest_mask) + 
                chest_color[c] * chest_mask
            ).astype(np.uint8)
        
        # Add some subtle shading to stomach
        stomach_x1, stomach_x2, stomach_y1, stomach_y2 = regions['stomach']
        
        # Create a gradient for stomach muscles (simplified)
        stomach_mask = np.zeros((height, width), dtype=np.float32)
        for y in range(stomach_y1, stomach_y2):
            for x in range(stomach_x1, stomach_x2):
                # Create a grid pattern
                dx = (x - (stomach_x1 + stomach_x2) / 2) / (stomach_x2 - stomach_x1) * 6
                dy = (y - stomach_y1) / (stomach_y2 - stomach_y1) * 6
                
                # Create a subtle grid pattern
                grid = np.sin(dx * np.pi) * np.sin(dy * np.pi) * 0.05
                stomach_mask[y, x] = max(0, grid)
        
        # Apply stomach details
        detailed_texture = detailed_texture * (1 - stomach_mask[:, :, np.newaxis])
        
        # Add face details
        face_x1, face_x2, face_y1, face_y2 = regions['face']
        
        # Basic eyes and mouth (very simplified)
        eye_radius = max(1, int((face_x2 - face_x1) * 0.1))
        eye_y = int((face_y1 + face_y2) * 0.4)
        left_eye_x = int((face_x1 + face_x2) * 0.4)
        right_eye_x = int((face_x1 + face_x2) * 0.6)
        
        # Add eyes
        cv2.circle(detailed_texture, (left_eye_x, eye_y), eye_radius, (100, 100, 100), -1)
        cv2.circle(detailed_texture, (right_eye_x, eye_y), eye_radius, (100, 100, 100), -1)
        
        # Add mouth
        mouth_y = int((face_y1 + face_y2) * 0.7)
        cv2.line(detailed_texture, 
                (int((face_x1 + face_x2) * 0.4), mouth_y), 
                (int((face_x1 + face_x2) * 0.6), mouth_y), 
                (150, 100, 100), max(1, int(eye_radius * 0.8)))
        
        return detailed_texture
    
    def generate_pbr_textures(self, mesh: trimesh.Trimesh, base_texture: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate PBR textures for a body mesh.
        
        Args:
            mesh: Body mesh with UV coordinates
            base_texture: Base color texture
            
        Returns:
            Dictionary of PBR textures (albedo, normal, roughness, etc.)
        """
        print("Generating PBR textures")
        
        # Check if mesh has UV coordinates
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            print("Mesh needs UV coordinates for PBR textures")
            mesh = self.generate_uv_coordinates(mesh)
        
        # Initialize PBR textures
        pbr_textures = {
            'albedo': base_texture,  # Base color
            'normal': np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8),
            'roughness': np.zeros((self.resolution[1], self.resolution[0]), dtype=np.uint8),
            'metallic': np.zeros((self.resolution[1], self.resolution[0]), dtype=np.uint8),
            'ao': np.ones((self.resolution[1], self.resolution[0]), dtype=np.uint8) * 255
        }
        
        # Generate normal map
        pbr_textures['normal'] = self._generate_normal_map(mesh)
        
        # Generate roughness map (skin is generally not very rough)
        roughness = np.ones((self.resolution[1], self.resolution[0]), dtype=np.uint8) * 180
        
        # Add some variation to roughness
        noise = np.random.normal(0, 10, roughness.shape).astype(np.int8)
        roughness = np.clip(roughness + noise, 120, 220).astype(np.uint8)
        
        # Skin is rougher at joints and less rough on smooth areas
        # This is a simplified approximation
        
        # Get UV coordinates and normals
        uvs = mesh.visual.uv
        normals = mesh.vertex_normals
        
        # For each vertex in the mesh
        for i, uv in enumerate(uvs):
            # Get texture coordinates
            u, v = uv
            tx = int(u * (self.resolution[0] - 1))
            ty = int(v * (self.resolution[1] - 1))
            
            # Check bounds
            if 0 <= tx < self.resolution[0] and 0 <= ty < self.resolution[1]:
                # Get vertex normal
                normal = normals[i]
                
                # Roughness is higher where normal changes rapidly
                # This is a simplified heuristic
                if i > 0:
                    normal_diff = np.linalg.norm(normal - normals[i-1])
                    roughness_mod = int(normal_diff * 100)
                    
                    # Apply modification to roughness
                    roughness[ty, tx] = np.clip(roughness[ty, tx] + roughness_mod, 120, 220)
        
        pbr_textures['roughness'] = roughness
        
        # Metallic map (skin is not metallic)
        metallic = np.ones((self.resolution[1], self.resolution[0]), dtype=np.uint8) * 0
        pbr_textures['metallic'] = metallic
        
        # Ambient occlusion map
        ao = self._generate_ambient_occlusion(mesh)
        pbr_textures['ao'] = ao
        
        return pbr_textures
    
    def _generate_normal_map(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Generate a normal map for a mesh.
        
        Args:
            mesh: Body mesh with UV coordinates
            
        Returns:
            Normal map image
        """
        # Initialize normal map
        normal_map = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        normal_map[:, :] = [128, 128, 255]  # Default normal (pointing up)
        
        # Get UV coordinates and normals
        uvs = mesh.visual.uv
        normals = mesh.vertex_normals
        
        # For each face in the mesh
        for face in mesh.faces:
            # Get UV coordinates for this face
            face_uvs = uvs[face]
            
            # Get normals for this face
            face_normals = normals[face]
            
            # Get texture coordinates
            tex_coords = np.zeros((3, 2), dtype=int)
            tex_coords[:, 0] = np.clip(face_uvs[:, 0] * (self.resolution[0] - 1), 0, self.resolution[0] - 1)
            tex_coords[:, 1] = np.clip(face_uvs[:, 1] * (self.resolution[1] - 1), 0, self.resolution[1] - 1)
            
            # Rasterize triangle and write normals
            self._rasterize_triangle_with_normals(normal_map, tex_coords, face_normals)
        
        return normal_map
    
    def _rasterize_triangle_with_normals(self, 
                                      normal_map: np.ndarray, 
                                      tex_coords: np.ndarray,
                                      normals: np.ndarray) -> None:
        """
        Rasterize a triangle and write normals to the normal map.
        
        Args:
            normal_map: Normal map to write to
            tex_coords: Texture coordinates for triangle vertices
            normals: Normals for triangle vertices
        """
        # Sort vertices by y-coordinate
        sorted_indices = np.argsort(tex_coords[:, 1])
        v0 = tex_coords[sorted_indices[0]]
        v1 = tex_coords[sorted_indices[1]]
        v2 = tex_coords[sorted_indices[2]]
        
        n0 = normals[sorted_indices[0]]
        n1 = normals[sorted_indices[1]]
        n2 = normals[sorted_indices[2]]
        
        # Convert normals to texture space ([-1,1] to [0,255])
        n0_color = ((n0 + 1) * 127.5).astype(np.uint8)
        n1_color = ((n1 + 1) * 127.5).astype(np.uint8)
        n2_color = ((n2 + 1) * 127.5).astype(np.uint8)
        
        # Compute bounding box
        min_x = max(0, min(v0[0], v1[0], v2[0]))
        max_x = min(normal_map.shape[1] - 1, max(v0[0], v1[0], v2[0]))
        min_y = max(0, min(v0[1], v1[1], v2[1]))
        max_y = min(normal_map.shape[0] - 1, max(v0[1], v1[1], v2[1]))
        
        # Check if triangle is outside the texture
        if min_x >= max_x or min_y >= max_y:
            return
        
        # Compute edge functions
        def edge_function(a, b, p):
            return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
        
        # Compute total triangle area
        area = edge_function(v0, v1, v2)
        
        if abs(area) < 1e-6:
            return  # Degenerate triangle
        
        # Rasterize triangle
        for y in range(int(min_y), int(max_y) + 1):
            for x in range(int(min_x), int(max_x) + 1):
                p = np.array([x, y])
                
                # Compute barycentric coordinates
                w0 = edge_function(v1, v2, p) / area
                w1 = edge_function(v2, v0, p) / area
                w2 = edge_function(v0, v1, p) / area
                
                # Check if point is inside triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Interpolate normal
                    normal = w0 * n0_color + w1 * n1_color + w2 * n2_color
                    normal_map[y, x] = normal.astype(np.uint8)
    
    def _generate_ambient_occlusion(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Generate an ambient occlusion map for a mesh.
        
        Args:
            mesh: Body mesh with UV coordinates
            
        Returns:
            Ambient occlusion map
        """
        # This is a simplified implementation
        # In a real implementation, you would use ray tracing or screen space methods
        
        # Initialize AO map
        ao_map = np.ones((self.resolution[1], self.resolution[0]), dtype=np.uint8) * 255
        
        # Try to compute ambient occlusion using trimesh
        try:
            # Compute ambient occlusion
            ao_vertices = trimesh.proximity.ambient_occlusion(mesh, mesh.vertices)
            
            # Get UV coordinates
            uvs = mesh.visual.uv
            
            # For each vertex in the mesh
            for i, uv in enumerate(uvs):
                # Get texture coordinates
                u, v = uv
                tx = int(u * (self.resolution[0] - 1))
                ty = int(v * (self.resolution[1] - 1))
                
                # Check bounds
                if 0 <= tx < self.resolution[0] and 0 <= ty < self.resolution[1]:
                    # Set AO value
                    ao_value = int((1.0 - ao_vertices[i]) * 255)
                    ao_map[ty, tx] = ao_value
            
            # Blur to smooth out the map
            ao_map = cv2.GaussianBlur(ao_map, (5, 5), 1.0)
            
        except Exception as e:
            print(f"Error computing ambient occlusion: {e}")
            
            # Fallback: generate a simple AO map based on vertex normals
            normals = mesh.vertex_normals
            
            # For each face
            for face in mesh.faces:
                # Get UV coordinates for this face
                face_uvs = mesh.visual.uv[face]
                
                # Get normals for this face
                face_normals = normals[face]
                
                # Get texture coordinates
                tex_coords = np.zeros((3, 2), dtype=int)
                tex_coords[:, 0] = np.clip(face_uvs[:, 0] * (self.resolution[0] - 1), 0, self.resolution[0] - 1)
                tex_coords[:, 1] = np.clip(face_uvs[:, 1] * (self.resolution[1] - 1), 0, self.resolution[1] - 1)
                
                # Compute average normal direction (up is [0,1,0])
                up = np.array([0, 1, 0])
                
                # Compute dot products with up vector
                dots = [np.dot(normal, up) for normal in face_normals]
                avg_dot = np.mean(dots)
                
                # Apply some occlusion where normals point sideways or down
                ao_value = int((avg_dot * 0.5 + 0.5) * 255)
                
                # Write to AO map
                for i in range(3):
                    tx, ty = tex_coords[i]
                    if 0 <= tx < self.resolution[0] and 0 <= ty < self.resolution[1]:
                        ao_map[ty, tx] = min(ao_map[ty, tx], ao_value)
        
        return ao_map
    
    def save_texture(self, texture: np.ndarray, output_path: str) -> str:
        """
        Save a texture to a file.
        
        Args:
            texture: Texture image
            output_path: Path to save the texture
            
        Returns:
            Path to the saved texture
        """
        # Create output directory
        ensure_directory(os.path.dirname(output_path))
        
        # Save texture
        save_image(output_path, texture)
        
        print(f"Saved texture to: {output_path}")
        return output_path
    
    def save_pbr_textures(self, pbr_textures: Dict[str, np.ndarray], output_dir: str) -> Dict[str, str]:
        """
        Save PBR textures to files.
        
        Args:
            pbr_textures: Dictionary of PBR textures
            output_dir: Directory to save textures
            
        Returns:
            Dictionary with paths to saved textures
        """
        # Create output directory
        ensure_directory(output_dir)
        
        # Save textures
        texture_paths = {}
        
        for name, texture in pbr_textures.items():
            output_path = os.path.join(output_dir, f"{name}.png")
            
            # Save texture
            self.save_texture(texture, output_path)
            
            # Add to dictionary
            texture_paths[name] = output_path
        
        return texture_paths
    
    def apply_texture_to_mesh(self, mesh: trimesh.Trimesh, texture_path: str) -> trimesh.Trimesh:
        """
        Apply a texture to a mesh.
        
        Args:
            mesh: Body mesh with UV coordinates
            texture_path: Path to texture image
            
        Returns:
            Textured mesh
        """
        print(f"Applying texture to mesh: {texture_path}")
        
        # Check if mesh has UV coordinates
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            print("Mesh needs UV coordinates for texturing")
            mesh = self.generate_uv_coordinates(mesh)
        
        # Load texture
        texture = load_image(texture_path)
        
        if texture is None:
            print(f"Failed to load texture: {texture_path}")
            return mesh
        
        # Create a copy of the mesh
        textured_mesh = mesh.copy()
        
        # Create material
        material = trimesh.visual.material.SimpleMaterial(image=texture)
        
        # Apply material
        textured_mesh.visual.material = material
        
        return textured_mesh 