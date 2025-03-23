import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
import open3d as o3d
import trimesh
from PIL import Image
import xatlas

from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    image_to_tensor,
    tensor_to_image,
    get_device,
    ensure_directory
)

from app.avatar_creation.face_modeling.texture_mapping import TextureMapper

class ImprovedTextureMapper(TextureMapper):
    """
    Enhanced texture mapping module with better UV unwrapping techniques.
    Extends the base TextureMapper with improved algorithms for texture projection.
    """
    
    def __init__(self, 
                resolution: Tuple[int, int] = (2048, 2048),
                seamless_boundary: bool = True,
                optimize_charts: bool = True):
        """
        Initialize the improved texture mapper.
        
        Args:
            resolution: Texture resolution (width, height)
            seamless_boundary: Whether to ensure seamless texture boundaries
            optimize_charts: Whether to optimize UV chart layout
        """
        super().__init__(resolution=resolution)
        self.seamless_boundary = seamless_boundary
        self.optimize_charts = optimize_charts
        
    def generate_optimized_uvs(self, mesh: Union[o3d.geometry.TriangleMesh, trimesh.Trimesh]) -> trimesh.Trimesh:
        """
        Generate optimized UV coordinates using xatlas.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Mesh with optimized UV coordinates
        """
        # Convert to trimesh if needed
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            tmesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            tmesh = mesh
        
        # Get vertices and faces
        vertices = np.array(tmesh.vertices, dtype=np.float32)
        faces = np.array(tmesh.faces, dtype=np.uint32)
        
        try:
            # Create xatlas atlas
            atlas = xatlas.Atlas()
            
            # Add mesh to atlas
            atlas.add_mesh(
                positions=vertices,
                indices=faces,
                normals=np.array(tmesh.vertex_normals, dtype=np.float32)
            )
            
            # Generate UV coordinates
            print("Generating optimized UV parameterization...")
            
            # Set xatlas options
            chart_options = xatlas.ChartOptions()
            if self.optimize_charts:
                chart_options.max_iterations = 4
                chart_options.max_cost = 2
                
            pack_options = xatlas.PackOptions()
            pack_options.resolution = max(self.resolution)
            pack_options.bruteForce = self.optimize_charts
            
            # Generate the atlas
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            
            # Get the results
            vmapping, indices, uvs = atlas[0]  # Assuming a single mesh
            
            # Create new mesh with optimized UVs
            new_vertices = vertices[vmapping]
            new_faces = np.array(indices).reshape(-1, 3)
            
            # Create the new mesh with UV coordinates
            new_mesh = trimesh.Trimesh(
                vertices=new_vertices,
                faces=new_faces,
                visual=trimesh.visual.TextureVisuals(
                    uv=np.array(uvs),
                    material=trimesh.visual.material.SimpleMaterial()
                )
            )
            
            return new_mesh
            
        except Exception as e:
            print(f"Error generating optimized UVs with xatlas: {e}")
            print("Falling back to default UV generation")
            
            # Fallback to base class method
            return super().generate_uvs(tmesh)
    
    def generate_uvs(self, mesh: Union[o3d.geometry.TriangleMesh, trimesh.Trimesh]) -> trimesh.Trimesh:
        """
        Generate UV coordinates using optimized algorithms.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Mesh with UV coordinates
        """
        # Try to use xatlas for optimized UV unwrapping
        try:
            return self.generate_optimized_uvs(mesh)
        except ImportError:
            print("xatlas module not available. Falling back to default UV generation.")
            # Fallback to base implementation
            return super().generate_uvs(mesh)
    
    def project_texture_from_multiple_views(self, 
                                          mesh: trimesh.Trimesh, 
                                          images: List[np.ndarray],
                                          landmarks_list: List[np.ndarray],
                                          view_weights: Optional[List[float]] = None) -> Dict:
        """
        Project texture from multiple views onto a mesh.
        
        Args:
            mesh: Input mesh
            images: List of input images
            landmarks_list: List of landmarks for each image
            view_weights: Optional weights for each view
            
        Returns:
            Dictionary containing textured mesh and texture image
        """
        # Set default weights if not provided
        if view_weights is None:
            view_weights = [1.0 / len(images)] * len(images)
        
        # Make sure mesh has UV coordinates
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            mesh = self.generate_uvs(mesh)
        
        # Create an empty texture image
        texture = np.zeros((self.resolution[1], self.resolution[0], 4), dtype=np.float32)
        
        # Weight accumulation buffer for blending
        weight_map = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        
        # For each view
        for idx, (image, landmarks, weight) in enumerate(zip(images, landmarks_list, view_weights)):
            print(f"Processing view {idx+1}/{len(images)}")
            
            # Project from this view
            result = self.project_texture(mesh, image, landmarks)
            view_texture = result["texture_image"]
            
            # Convert to RGBA float32
            if view_texture.shape[2] == 3:
                view_texture_rgba = np.ones((view_texture.shape[0], view_texture.shape[1], 4), dtype=np.float32)
                view_texture_rgba[:, :, :3] = view_texture.astype(np.float32) / 255.0
            else:
                view_texture_rgba = view_texture.astype(np.float32) / 255.0
            
            # Add to accumulated texture with weight
            texture += view_texture_rgba * weight
            weight_map += weight
        
        # Normalize by weights
        valid_mask = weight_map > 0
        texture[valid_mask] /= weight_map[valid_mask, np.newaxis]
        
        # Convert back to uint8
        texture_uint8 = (texture * 255).astype(np.uint8)
        
        # If no alpha channel, extract RGB
        if images[0].shape[2] == 3:
            texture_uint8 = texture_uint8[:, :, :3]
        
        # Apply texture to mesh
        textured_mesh = mesh.copy()
        textured_mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv,
            material=trimesh.visual.material.SimpleMaterial(),
            image=Image.fromarray(texture_uint8)
        )
        
        return {
            "textured_mesh": textured_mesh,
            "texture_image": texture_uint8
        }
    
    def fill_texture_holes(self, texture: np.ndarray, max_hole_size: int = 10) -> np.ndarray:
        """
        Fill small holes in the texture using inpainting.
        
        Args:
            texture: Input texture image
            max_hole_size: Maximum hole size to fill
            
        Returns:
            Texture with holes filled
        """
        # Create a mask for areas with no texture
        if texture.shape[2] == 4:
            # Use alpha channel as mask
            mask = texture[:, :, 3] == 0
            mask = mask.astype(np.uint8) * 255
        else:
            # For RGB textures, identify black areas as holes
            mask = np.all(texture < 5, axis=2).astype(np.uint8) * 255
        
        # Perform morphological operations to identify holes
        kernel = np.ones((3, 3), np.uint8)
        # Dilate to capture hole boundaries
        dilated = cv2.dilate(mask, kernel, iterations=1)
        # Erode to remove small holes
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Remove large holes (keep only small ones for filling)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_hole_mask = np.zeros_like(eroded)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max_hole_size * max_hole_size:
                cv2.drawContours(small_hole_mask, [contour], 0, 255, -1)
        
        # Perform inpainting only on small holes
        if texture.shape[2] == 4:
            # Process RGB channels separately
            filled_texture = texture.copy()
            for i in range(3):  # RGB channels
                channel = texture[:, :, i]
                filled_channel = cv2.inpaint(channel, small_hole_mask, 3, cv2.INPAINT_TELEA)
                filled_texture[:, :, i] = filled_channel
            
            # Update alpha channel
            filled_texture[:, :, 3] = np.where(small_hole_mask > 0, 255, texture[:, :, 3])
        else:
            # Process RGB texture
            filled_texture = cv2.inpaint(texture, small_hole_mask, 3, cv2.INPAINT_TELEA)
        
        return filled_texture
    
    def post_process_texture(self, texture: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to the texture for better quality.
        
        Args:
            texture: Input texture image
            
        Returns:
            Post-processed texture
        """
        # Fill texture holes
        filled_texture = self.fill_texture_holes(texture)
        
        # Apply edge-aware smoothing to reduce noise while preserving details
        if filled_texture.shape[2] == 4:
            # Process RGB channels
            rgb = filled_texture[:, :, :3]
            alpha = filled_texture[:, :, 3]
            
            # Apply bilateral filter for edge-aware smoothing
            smoothed_rgb = cv2.bilateralFilter(rgb, 5, 25, 25)
            
            # Combine with alpha
            result = np.dstack((smoothed_rgb, alpha))
        else:
            # Process RGB image
            result = cv2.bilateralFilter(filled_texture, 5, 25, 25)
        
        # Enhance contrast slightly
        if result.dtype != np.uint8:
            result = (result * 255).astype(np.uint8)
            
        lab = cv2.cvtColor(result[:, :, :3], cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Combine with alpha if present
        if result.shape[2] == 4:
            result = np.dstack((enhanced_rgb, result[:, :, 3]))
        else:
            result = enhanced_rgb
        
        return result
    
    def enhance_texture(self, texture: np.ndarray) -> np.ndarray:
        """
        Enhance texture with advanced techniques.
        
        Args:
            texture: Input texture image
            
        Returns:
            Enhanced texture
        """
        # Apply advanced post-processing
        processed_texture = self.post_process_texture(texture)
        
        # Apply additional super-resolution if needed
        if max(self.resolution) >= 4096:
            # For high-res textures, apply additional detail enhancement
            processed_texture = self.enhance_texture_details(processed_texture)
        
        return processed_texture
    
    def enhance_texture_details(self, texture: np.ndarray) -> np.ndarray:
        """
        Enhance fine details in the texture using frequency domain techniques.
        
        Args:
            texture: Input texture image
            
        Returns:
            Detail-enhanced texture
        """
        # Extract channels
        if texture.shape[2] == 4:
            rgb = texture[:, :, :3]
            alpha = texture[:, :, 3]
        else:
            rgb = texture
            alpha = None
        
        # Convert to YUV color space to process Y channel
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(yuv)
        
        # Apply unsharp masking to enhance details
        gaussian = cv2.GaussianBlur(y, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(y, 1.5, gaussian, -0.5, 0)
        
        # Combine channels
        enhanced_yuv = cv2.merge((unsharp_mask, u, v))
        enhanced_rgb = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2RGB)
        
        # Combine with alpha if present
        if alpha is not None:
            enhanced_texture = np.dstack((enhanced_rgb, alpha))
        else:
            enhanced_texture = enhanced_rgb
        
        return enhanced_texture
    
    def export_textured_model(self, 
                            mesh: trimesh.Trimesh, 
                            texture: np.ndarray, 
                            output_dir: str,
                            model_name: str = "textured_model") -> Dict:
        """
        Export a textured model to OBJ format with MTL and texture files.
        
        Args:
            mesh: Textured mesh
            texture: Texture image
            output_dir: Output directory
            model_name: Base name for output files
            
        Returns:
            Dictionary with paths to exported files
        """
        ensure_directory(output_dir)
        
        # Save texture image
        texture_path = os.path.join(output_dir, f"{model_name}_texture.png")
        if texture.shape[2] == 4:
            # Save with alpha channel
            Image.fromarray(texture).save(texture_path, format="PNG")
        else:
            # Save RGB image
            Image.fromarray(texture).save(texture_path, format="PNG")
        
        # Create a new mesh with the texture applied
        textured_mesh = mesh.copy()
        
        # Assign texture to mesh if not already assigned
        if not hasattr(textured_mesh.visual, 'uv') or textured_mesh.visual.uv is None:
            # Generate UVs if needed
            textured_mesh = self.generate_uvs(textured_mesh)
        
        # Set texture image
        textured_mesh.visual = trimesh.visual.TextureVisuals(
            uv=textured_mesh.visual.uv,
            material=trimesh.visual.material.SimpleMaterial(),
            image=Image.fromarray(texture)
        )
        
        # Save OBJ file with MTL and texture reference
        obj_path = os.path.join(output_dir, f"{model_name}.obj")
        mtl_path = os.path.join(output_dir, f"{model_name}.mtl")
        
        # Export with texture reference
        textured_mesh.export(
            obj_path,
            include_normals=True,
            include_texture=True
        )
        
        return {
            "obj_path": obj_path,
            "mtl_path": mtl_path,
            "texture_path": texture_path
        } 