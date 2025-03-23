import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
import trimesh
import json
from PIL import Image

from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    get_device,
    ensure_directory
)

class BodyTextureMapper:
    """
    Class for generating and mapping textures to 3D body models.
    Supports texture projection from images, texture synthesis, and seam correction.
    """
    
    def __init__(self, 
                resolution: Tuple[int, int] = (2048, 2048),
                use_normal_maps: bool = True,
                use_advanced_synthesis: bool = False,
                optimize_uv: bool = True):
        """
        Initialize the body texture mapper.
        
        Args:
            resolution: Resolution of the texture maps
            use_normal_maps: Whether to generate normal maps
            use_advanced_synthesis: Whether to use advanced texture synthesis
            optimize_uv: Whether to optimize UV maps for better texture quality
        """
        self.device = get_device()
        self.resolution = resolution
        self.use_normal_maps = use_normal_maps
        self.use_advanced_synthesis = use_advanced_synthesis
        self.optimize_uv = optimize_uv
        
        # Image processing tools
        self.face_detector = None  # Placeholder for face detector
        
        print(f"Initialized BodyTextureMapper with resolution {resolution}")
    
    def generate_uv_mapping(self, mesh: trimesh.Trimesh) -> Dict:
        """
        Generate UV coordinates for a mesh.
        
        Args:
            mesh: Input mesh without UV coordinates
            
        Returns:
            Dictionary containing the mesh with UV coordinates
        """
        print("Generating UV mapping")
        
        # Check if mesh already has UV coordinates
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            print("Mesh already has UV coordinates")
            return {'mesh': mesh}
        
        # This is a simplified implementation
        # In a real implementation, you would use a proper UV unwrapping algorithm
        # like Xatlas or a custom unwrapping approach
        
        # Generate simple spherical UV mapping as a placeholder
        vertices = mesh.vertices
        
        # Normalize to range [-1, 1]
        min_vals = np.min(vertices, axis=0)
        max_vals = np.max(vertices, axis=0)
        vertices_normalized = (vertices - min_vals) / (max_vals - min_vals) * 2 - 1
        
        # Generate spherical UV coordinates
        u = np.arctan2(vertices_normalized[:, 0], vertices_normalized[:, 2]) / (2 * np.pi) + 0.5
        v = np.arcsin(np.clip(vertices_normalized[:, 1], -1, 1)) / np.pi + 0.5
        
        # Create UV array
        uv = np.column_stack((u, v))
        
        # Create a new mesh with UV coordinates
        uv_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            vertex_normals=mesh.vertex_normals
        )
        
        # Apply UV coordinates
        if hasattr(uv_mesh, 'visual') and hasattr(uv_mesh.visual, 'uv'):
            uv_mesh.visual.uv = uv
        else:
            # Create a material and assign UVs
            material = trimesh.visual.material.SimpleMaterial(name='default')
            uv_visual = trimesh.visual.texture.TextureVisuals(
                uv=uv, 
                material=material
            )
            uv_mesh.visual = uv_visual
        
        return {
            'mesh': uv_mesh,
            'uv': uv
        }
    
    def optimize_uv_mapping(self, mesh: trimesh.Trimesh) -> Dict:
        """
        Optimize the UV mapping of a mesh to minimize distortion.
        
        Args:
            mesh: Input mesh with UV coordinates
            
        Returns:
            Dictionary containing the mesh with optimized UV coordinates
        """
        if not self.optimize_uv:
            return {'mesh': mesh}
        
        print("Optimizing UV mapping")
        
        # For a real implementation, you would use a proper UV optimization algorithm
        # This is a placeholder for that process
        
        try:
            import xatlas
            
            # Extract the mesh data
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
            
            # Run xatlas to generate optimized UVs
            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices, faces)
            atlas.generate()
            vmapping, indices, uvs = atlas[0]  # Get the first chart
            
            # Create a new mesh with optimized UV coordinates
            optimized_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                face_normals=mesh.face_normals,
                vertex_normals=mesh.vertex_normals
            )
            
            # Apply optimized UV coordinates
            if hasattr(optimized_mesh, 'visual') and hasattr(optimized_mesh.visual, 'uv'):
                optimized_mesh.visual.uv = uvs
            else:
                # Create a material and assign UVs
                material = trimesh.visual.material.SimpleMaterial(name='default')
                uv_visual = trimesh.visual.texture.TextureVisuals(
                    uv=uvs, 
                    material=material
                )
                optimized_mesh.visual = uv_visual
            
            return {
                'mesh': optimized_mesh,
                'uv': uvs
            }
            
        except ImportError:
            print("Xatlas not available. Using original UV mapping.")
            return {'mesh': mesh}
        except Exception as e:
            print(f"Error optimizing UV mapping: {e}")
            return {'mesh': mesh}
    
    def project_texture_from_image(self, mesh: trimesh.Trimesh, image_path: str, 
                                 body_landmarks: Optional[np.ndarray] = None) -> Dict:
        """
        Project texture from an image onto a mesh.
        
        Args:
            mesh: Input mesh with UV coordinates
            image_path: Path to the image to project
            body_landmarks: Body landmarks for better alignment (optional)
            
        Returns:
            Dictionary containing the textured mesh and texture image
        """
        print(f"Projecting texture from image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        
        # Ensure mesh has UV coordinates
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            uv_result = self.generate_uv_mapping(mesh)
            mesh = uv_result['mesh']
        
        # Create a blank texture
        texture = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # In a real implementation, you would use proper texture projection
        # This is a simplified implementation for demonstration
        
        # For now, just fill with a gradient
        for y in range(texture.shape[0]):
            for x in range(texture.shape[1]):
                # Create a simple gradient
                r = int(255 * x / texture.shape[1])
                g = int(255 * y / texture.shape[0])
                b = int(128)
                texture[y, x] = [r, g, b]
        
        # Create a textured material
        material = trimesh.visual.material.SimpleMaterial(name='body_texture')
        
        # Create a PIL image from the texture
        texture_image = Image.fromarray(texture)
        
        # Create texture visuals
        texture_visual = trimesh.visual.texture.TextureVisuals(
            uv=mesh.visual.uv, 
            image=texture_image,
            material=material
        )
        
        # Apply texture to mesh
        textured_mesh = mesh.copy()
        textured_mesh.visual = texture_visual
        
        return {
            'textured_mesh': textured_mesh,
            'texture_image': texture
        }
    
    def create_texture_from_multiple_images(self, mesh: trimesh.Trimesh, 
                                          image_paths: List[str],
                                          pose_data: Optional[List[Dict]] = None) -> Dict:
        """
        Create a texture from multiple images showing different views.
        
        Args:
            mesh: Input mesh with UV coordinates
            image_paths: List of paths to images
            pose_data: Optional pose information for each image
            
        Returns:
            Dictionary containing the textured mesh and texture image
        """
        print(f"Creating texture from {len(image_paths)} images")
        
        # Ensure mesh has UV coordinates
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            uv_result = self.generate_uv_mapping(mesh)
            mesh = uv_result['mesh']
        
        # Create a blank texture
        texture = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # In a real implementation, you would properly project all images
        # and blend them intelligently
        # This is a simplified implementation
        
        # For now, just use the first image as the base
        if image_paths:
            base_result = self.project_texture_from_image(mesh, image_paths[0])
            texture = base_result['texture_image']
        
        # Create a textured material
        material = trimesh.visual.material.SimpleMaterial(name='body_texture')
        
        # Create a PIL image from the texture
        texture_image = Image.fromarray(texture)
        
        # Create texture visuals
        texture_visual = trimesh.visual.texture.TextureVisuals(
            uv=mesh.visual.uv, 
            image=texture_image,
            material=material
        )
        
        # Apply texture to mesh
        textured_mesh = mesh.copy()
        textured_mesh.visual = texture_visual
        
        return {
            'textured_mesh': textured_mesh,
            'texture_image': texture
        }
    
    def generate_procedural_texture(self, mesh: trimesh.Trimesh, 
                                  texture_type: str = "skin",
                                  params: Optional[Dict] = None) -> Dict:
        """
        Generate a procedural texture for the body.
        
        Args:
            mesh: Input mesh with UV coordinates
            texture_type: Type of texture to generate (skin, clothing, etc.)
            params: Parameters for texture generation
            
        Returns:
            Dictionary containing the textured mesh and texture image
        """
        print(f"Generating procedural {texture_type} texture")
        
        # Ensure mesh has UV coordinates
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            uv_result = self.generate_uv_mapping(mesh)
            mesh = uv_result['mesh']
        
        # Create a blank texture
        texture = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        if params is None:
            params = {}
        
        # Generate different types of textures based on type
        if texture_type == "skin":
            # Generate skin texture
            skin_color = params.get("skin_color", [255, 210, 180])
            
            # Create a base skin color
            texture[:, :] = skin_color
            
            # Add some noise for skin texture
            noise = np.random.normal(0, 10, texture.shape).astype(np.int32)
            texture = np.clip(texture + noise, 0, 255).astype(np.uint8)
            
        elif texture_type == "clothing":
            # Generate clothing texture
            color = params.get("color", [50, 50, 150])
            
            # Create a base color
            texture[:, :] = color
            
            # Add some pattern
            pattern_scale = params.get("pattern_scale", 50)
            for y in range(texture.shape[0]):
                for x in range(texture.shape[1]):
                    if (x // pattern_scale + y // pattern_scale) % 2 == 0:
                        texture[y, x] = [min(c + 30, 255) for c in color]
                        
        else:
            # Default texture
            for y in range(texture.shape[0]):
                for x in range(texture.shape[1]):
                    # Create a simple gradient
                    r = int(255 * x / texture.shape[1])
                    g = int(255 * y / texture.shape[0])
                    b = int(128)
                    texture[y, x] = [r, g, b]
        
        # Create a textured material
        material = trimesh.visual.material.SimpleMaterial(name=f'{texture_type}_texture')
        
        # Create a PIL image from the texture
        texture_image = Image.fromarray(texture)
        
        # Create texture visuals
        texture_visual = trimesh.visual.texture.TextureVisuals(
            uv=mesh.visual.uv, 
            image=texture_image,
            material=material
        )
        
        # Apply texture to mesh
        textured_mesh = mesh.copy()
        textured_mesh.visual = texture_visual
        
        return {
            'textured_mesh': textured_mesh,
            'texture_image': texture
        }
    
    def combine_face_body_textures(self, body_mesh: trimesh.Trimesh, 
                                 body_texture: np.ndarray,
                                 face_texture: np.ndarray,
                                 face_mask: np.ndarray) -> Dict:
        """
        Combine face and body textures seamlessly.
        
        Args:
            body_mesh: Body mesh with UV coordinates
            body_texture: Body texture image
            face_texture: Face texture image
            face_mask: Mask indicating face region in the UV map
            
        Returns:
            Dictionary containing the combined texture
        """
        print("Combining face and body textures")
        
        # Create a copy of the body texture
        combined_texture = body_texture.copy()
        
        # Resize face texture and mask if needed
        if face_texture.shape[:2] != combined_texture.shape[:2]:
            face_texture_resized = cv2.resize(
                face_texture, 
                (combined_texture.shape[1], combined_texture.shape[0])
            )
            face_mask_resized = cv2.resize(
                face_mask, 
                (combined_texture.shape[1], combined_texture.shape[0])
            )
        else:
            face_texture_resized = face_texture
            face_mask_resized = face_mask
        
        # Normalize mask to [0, 1]
        face_mask_normalized = face_mask_resized / 255.0
        
        # Expand mask dimensions if needed
        if len(face_mask_normalized.shape) == 2 and len(combined_texture.shape) == 3:
            face_mask_normalized = np.expand_dims(face_mask_normalized, axis=2)
            face_mask_normalized = np.repeat(face_mask_normalized, 3, axis=2)
        
        # Blend textures using the mask
        combined_texture = (
            combined_texture * (1 - face_mask_normalized) + 
            face_texture_resized * face_mask_normalized
        ).astype(np.uint8)
        
        # Create a textured material
        material = trimesh.visual.material.SimpleMaterial(name='combined_texture')
        
        # Create a PIL image from the texture
        texture_image = Image.fromarray(combined_texture)
        
        # Create texture visuals
        texture_visual = trimesh.visual.texture.TextureVisuals(
            uv=body_mesh.visual.uv, 
            image=texture_image,
            material=material
        )
        
        # Apply texture to mesh
        textured_mesh = body_mesh.copy()
        textured_mesh.visual = texture_visual
        
        return {
            'textured_mesh': textured_mesh,
            'texture_image': combined_texture
        }
    
    def enhance_texture(self, texture: np.ndarray) -> np.ndarray:
        """
        Enhance a texture with better colors and details.
        
        Args:
            texture: Input texture image
            
        Returns:
            Enhanced texture image
        """
        print("Enhancing texture")
        
        # This is a simplified implementation
        # In a real implementation, you would use more sophisticated image enhancement
        
        # Convert to float for processing
        texture_float = texture.astype(np.float32) / 255.0
        
        # Enhance contrast
        texture_enhanced = texture_float * 1.2
        texture_enhanced = np.clip(texture_enhanced, 0, 1)
        
        # Enhance saturation (simplified)
        hsv = cv2.cvtColor(texture_enhanced.astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.1  # Increase saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
        texture_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Convert back to uint8
        texture_enhanced = (texture_enhanced * 255).astype(np.uint8)
        
        return texture_enhanced
    
    def generate_normal_map(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Generate a normal map from a mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Normal map image
        """
        if not self.use_normal_maps:
            return None
        
        print("Generating normal map")
        
        # Ensure mesh has UV coordinates
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            uv_result = self.generate_uv_mapping(mesh)
            mesh = uv_result['mesh']
        
        # Create a blank normal map
        normal_map = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Get vertex normals
        vertex_normals = mesh.vertex_normals
        
        # Get UV coordinates
        uv_coords = mesh.visual.uv
        
        # For each face
        for face_idx, face in enumerate(mesh.faces):
            # Get face vertices
            vertices = mesh.vertices[face]
            
            # Get face vertex normals
            normals = vertex_normals[face]
            
            # Get face UV coordinates
            face_uvs = uv_coords[face]
            
            # Convert UV coordinates to pixel coordinates
            pixel_coords = np.floor(face_uvs * np.array([self.resolution[0], self.resolution[1]])).astype(np.int32)
            
            # Clip to image bounds
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, self.resolution[0] - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, self.resolution[1] - 1)
            
            # Simplified: just set the normal at each UV coordinate
            for i in range(3):
                px, py = pixel_coords[i]
                # Convert normal from [-1, 1] to [0, 255]
                normal = (normals[i] * 0.5 + 0.5) * 255
                normal_map[py, px] = normal
        
        # Fill in the gaps using a simple dilation
        kernel = np.ones((5, 5), np.uint8)
        mask = np.all(normal_map == 0, axis=2).astype(np.uint8)
        filled_normal_map = normal_map.copy()
        
        max_iterations = 10
        for _ in range(max_iterations):
            if np.sum(mask) == 0:
                break
                
            # Dilate the non-zero regions
            dilated_mask = cv2.dilate(1 - mask, kernel, iterations=1)
            new_mask = 1 - dilated_mask
            
            # For each channel
            for c in range(3):
                temp = filled_normal_map[:, :, c].copy()
                dilated = cv2.dilate(temp, kernel, iterations=1)
                filled_normal_map[:, :, c] = np.where(mask == 1, dilated, temp)
            
            mask = new_mask
        
        return filled_normal_map
    
    def export_textured_model(self, mesh: trimesh.Trimesh, 
                            texture: np.ndarray,
                            output_dir: str,
                            model_name: str = "textured_body") -> Dict:
        """
        Export a textured model to common 3D formats.
        
        Args:
            mesh: Textured mesh
            texture: Texture image
            output_dir: Output directory
            model_name: Name of the output files
            
        Returns:
            Dictionary with paths to the exported files
        """
        print(f"Exporting textured model to {output_dir}")
        
        # Create output directory
        ensure_directory(output_dir)
        
        # Save texture image
        texture_path = os.path.join(output_dir, f"{model_name}_texture.png")
        cv2.imwrite(texture_path, cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
        
        # Create a copy of the mesh with the texture
        textured_mesh = mesh.copy()
        
        # Make sure mesh has UV coordinates
        if not hasattr(textured_mesh, 'visual') or not hasattr(textured_mesh.visual, 'uv') or textured_mesh.visual.uv is None:
            uv_result = self.generate_uv_mapping(textured_mesh)
            textured_mesh = uv_result['mesh']
        
        # Create a textured material
        material = trimesh.visual.material.SimpleMaterial(name='exported_texture')
        
        # Create a PIL image from the texture
        texture_image = Image.fromarray(texture)
        
        # Create texture visuals
        texture_visual = trimesh.visual.texture.TextureVisuals(
            uv=textured_mesh.visual.uv, 
            image=texture_image,
            material=material
        )
        
        # Apply texture to mesh
        textured_mesh.visual = texture_visual
        
        # Export to OBJ format
        obj_path = os.path.join(output_dir, f"{model_name}.obj")
        mtl_path = os.path.join(output_dir, f"{model_name}.mtl")
        
        # Export the mesh with texture
        textured_mesh.export(obj_path)
        
        # Also export to GLB format if possible
        try:
            glb_path = os.path.join(output_dir, f"{model_name}.glb")
            textured_mesh.export(glb_path)
        except Exception as e:
            print(f"Error exporting to GLB: {e}")
            glb_path = None
        
        return {
            'obj_path': obj_path,
            'mtl_path': mtl_path,
            'texture_path': texture_path,
            'glb_path': glb_path
        } 