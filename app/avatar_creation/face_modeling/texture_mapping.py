import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
import open3d as o3d
import trimesh
import pyrender
from PIL import Image

from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    preprocess_image,
    tensor_to_image,
    image_to_tensor,
    get_device,
    ensure_directory
)

class TextureMapper:
    """
    Class for high-fidelity texture mapping of 3D face models.
    Supports 4K resolution textures and detailed UV mapping.
    """
    
    def __init__(self, resolution: Tuple[int, int] = (4096, 4096)):
        """
        Initialize the texture mapper.
        
        Args:
            resolution: Texture resolution (width, height)
        """
        self.device = get_device()
        self.resolution = resolution
    
    def create_uv_map(self, mesh: Union[o3d.geometry.TriangleMesh, trimesh.Trimesh]) -> Dict:
        """
        Create a UV map for the 3D mesh.
        
        Args:
            mesh: 3D face mesh
            
        Returns:
            Dictionary containing UV coordinates and mapping info
        """
        # Convert to trimesh if Open3D mesh
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        else:
            trimesh_mesh = mesh
        
        # Unwrap the mesh to create UV coordinates
        # This is a simplified approach - in production, use more sophisticated unwrapping
        unwrapped = trimesh_mesh.unwrap()
        
        # Extract UV coordinates
        uv_coords = unwrapped.visual.uv
        
        return {
            "mesh": unwrapped,
            "uv_coords": uv_coords
        }
    
    def project_texture(self, 
                        mesh: Union[o3d.geometry.TriangleMesh, trimesh.Trimesh], 
                        image: np.ndarray, 
                        landmarks: np.ndarray) -> Dict:
        """
        Project texture from a 2D image onto the 3D mesh.
        
        Args:
            mesh: 3D face mesh
            image: Input image for texture
            landmarks: Facial landmarks for alignment
            
        Returns:
            Dictionary containing the textured mesh and texture image
        """
        # Convert to trimesh if needed
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        else:
            trimesh_mesh = mesh
        
        # Create a UV map
        uv_data = self.create_uv_map(trimesh_mesh)
        unwrapped_mesh = uv_data["mesh"]
        uv_coords = uv_data["uv_coords"]
        
        # Create a texture image (4K resolution)
        texture_img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Create a renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1], 
                                             viewport_height=image.shape[0])
        
        # Create a scene
        scene = pyrender.Scene()
        
        # Add a camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=np.eye(4))
        
        # Add a mesh to the scene
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(1.0, 1.0, 1.0, 1.0),
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        
        render_mesh = pyrender.Mesh.from_trimesh(
            unwrapped_mesh,
            material=material
        )
        scene.add(render_mesh)
        
        # Render the mesh with the image as texture
        # This is a simplified approach - full implementation would use projective texturing
        # with vertex or face coloring based on the image
        color, _ = renderer.render(scene)
        
        # Map the image colors to the UV coordinates
        for i, uv in enumerate(uv_coords):
            u, v = uv
            x = int(u * self.resolution[0])
            y = int(v * self.resolution[1])
            
            # Ensure within bounds
            x = max(0, min(x, self.resolution[0] - 1))
            y = max(0, min(y, self.resolution[1] - 1))
            
            # Sample the image color for this vertex
            # In a real implementation, use proper texture projection and blending
            # This is a simplified approach
            texture_img[y, x] = color[i] if i < len(color) else [0, 0, 0]
        
        # Add the texture to the mesh
        unwrapped_mesh.visual.uv = uv_coords
        unwrapped_mesh.visual.material = trimesh.visual.material.SimpleMaterial(
            image=texture_img
        )
        
        return {
            "textured_mesh": unwrapped_mesh,
            "texture_image": texture_img,
            "uv_coords": uv_coords
        }
    
    def enhance_texture(self, texture_image: np.ndarray) -> np.ndarray:
        """
        Enhance the texture image for better quality.
        
        Args:
            texture_image: Input texture image
            
        Returns:
            Enhanced texture image
        """
        # Upscale to 4K if needed
        if texture_image.shape[0] != self.resolution[1] or texture_image.shape[1] != self.resolution[0]:
            texture_image = cv2.resize(texture_image, self.resolution, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply enhancement filters
        # Sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(texture_image, -1, kernel)
        
        # Color correction
        lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE on L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_image
    
    def apply_detail_textures(self, base_texture: np.ndarray, detail_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply detail textures for skin pores, wrinkles, etc.
        
        Args:
            base_texture: Base texture image
            detail_map: Optional detail normal map
            
        Returns:
            Texture with details applied
        """
        # If no detail map provided, generate a procedural one
        if detail_map is None:
            # Create a procedural detail map for skin pores and fine details
            detail_map = np.zeros_like(base_texture)
            
            # Generate noise for skin pores
            noise = np.random.normal(0, 1, (self.resolution[1]//4, self.resolution[0]//4)).astype(np.float32)
            noise = cv2.resize(noise, self.resolution, interpolation=cv2.INTER_CUBIC)
            
            # Normalize to 0-1 range
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            
            # Convert to detail normal map
            detail_map[:, :, 0] = 0.5  # X component
            detail_map[:, :, 1] = 0.5  # Y component
            detail_map[:, :, 2] = 1.0 - noise * 0.2  # Z component
        
        # Blend the detail map with the base texture
        # This is a simplified normal map blending approach
        # In production, use proper normal map blending techniques
        result = base_texture.copy().astype(np.float32) / 255.0
        detail_norm = detail_map.astype(np.float32) / 255.0
        
        # Simple multiplicative blending
        blended = result * detail_norm
        
        # Convert back to 0-255 range
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        
        return blended
    
    def save_texture(self, texture_image: np.ndarray, save_path: str) -> None:
        """
        Save texture image to file.
        
        Args:
            texture_image: Texture image as numpy array
            save_path: Path where to save the texture
        """
        ensure_directory(os.path.dirname(save_path))
        
        # Convert to PIL Image for better quality saving
        img = Image.fromarray(texture_image)
        
        # Save as PNG for lossless quality
        img.save(save_path, format='PNG')
    
    def export_textured_model(self, 
                             textured_mesh: trimesh.Trimesh, 
                             texture_image: np.ndarray, 
                             export_dir: str, 
                             model_name: str = "face_model") -> Dict:
        """
        Export the textured 3D model in standard formats.
        
        Args:
            textured_mesh: Textured 3D mesh
            texture_image: Texture image
            export_dir: Directory for export
            model_name: Base name for the exported files
            
        Returns:
            Dictionary of export paths
        """
        ensure_directory(export_dir)
        
        # Export paths
        obj_path = os.path.join(export_dir, f"{model_name}.obj")
        mtl_path = os.path.join(export_dir, f"{model_name}.mtl")
        texture_path = os.path.join(export_dir, f"{model_name}_texture.png")
        
        # Save the texture
        self.save_texture(texture_image, texture_path)
        
        # Export the model with texture reference
        textured_mesh.export(obj_path, include_texture=True)
        
        return {
            "obj_path": obj_path,
            "mtl_path": mtl_path,
            "texture_path": texture_path
        }
