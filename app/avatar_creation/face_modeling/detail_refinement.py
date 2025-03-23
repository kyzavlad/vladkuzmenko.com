import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.avatar_creation.face_modeling.utils import (
    tensor_to_image,
    image_to_tensor,
    get_device,
    ensure_directory
)

class DetailRefinement:
    """
    Class for high-resolution detail refinement of 3D face models.
    Adds realistic skin details like pores, wrinkles, and fine textures.
    """
    
    def __init__(self, detail_maps_dir: Optional[str] = None, 
                resolution: Tuple[int, int] = (4096, 4096)):
        """
        Initialize the detail refinement system.
        
        Args:
            detail_maps_dir: Directory containing detail maps (normal maps, etc.)
            resolution: Target resolution for detail maps
        """
        self.device = get_device()
        self.resolution = resolution
        self.detail_maps = {}
        
        # Load detail maps if directory is provided
        if detail_maps_dir and os.path.exists(detail_maps_dir):
            self._load_detail_maps(detail_maps_dir)
        else:
            # Generate procedural detail maps
            self._generate_procedural_maps()
    
    def _load_detail_maps(self, detail_maps_dir: str) -> None:
        """
        Load detail maps from directory.
        
        Args:
            detail_maps_dir: Directory containing detail maps
        """
        # Expected detail map types
        map_types = ['pores', 'wrinkles', 'fine_details', 'normal_map']
        
        for map_type in map_types:
            # Look for maps with common image extensions
            for ext in ['.png', '.jpg', '.exr']:
                map_path = os.path.join(detail_maps_dir, f"{map_type}{ext}")
                if os.path.exists(map_path):
                    # Load the map
                    if ext == '.exr':
                        # For EXR files (normal maps), use OpenCV with HDR support
                        detail_map = cv2.imread(map_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        detail_map = cv2.cvtColor(detail_map, cv2.COLOR_BGR2RGB)
                    else:
                        # For standard images
                        detail_map = cv2.imread(map_path)
                        detail_map = cv2.cvtColor(detail_map, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target resolution
                    detail_map = cv2.resize(detail_map, self.resolution, interpolation=cv2.INTER_LANCZOS4)
                    
                    # Store the map
                    self.detail_maps[map_type] = detail_map
                    break
    
    def _generate_procedural_maps(self) -> None:
        """
        Generate procedural detail maps when no maps are provided.
        """
        # Generate pore detail map
        pore_map = self._generate_pore_map()
        self.detail_maps['pores'] = pore_map
        
        # Generate wrinkle detail map
        wrinkle_map = self._generate_wrinkle_map()
        self.detail_maps['wrinkles'] = wrinkle_map
        
        # Generate fine details map
        fine_details = self._generate_fine_details_map()
        self.detail_maps['fine_details'] = fine_details
        
        # Generate normal map from combined details
        normal_map = self._generate_normal_map(pore_map, wrinkle_map, fine_details)
        self.detail_maps['normal_map'] = normal_map
    
    def _generate_pore_map(self) -> np.ndarray:
        """
        Generate a procedural pore detail map.
        
        Returns:
            Pore detail map as numpy array
        """
        # Create a noise-based pore map
        pore_map = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        
        # Generate multiple octaves of noise for realistic pores
        scale = 0.1
        for octave in range(3):
            # Generate noise at current scale
            noise_size = (int(self.resolution[1] * scale), int(self.resolution[0] * scale))
            noise = np.random.normal(0, 1, noise_size).astype(np.float32)
            
            # Resize to full resolution
            noise_resized = cv2.resize(noise, self.resolution, interpolation=cv2.INTER_CUBIC)
            
            # Add to pore map with decreasing weight for higher octaves
            weight = 1.0 / (2 ** octave)
            pore_map += noise_resized * weight
            
            # Double the scale for next octave
            scale *= 2
        
        # Normalize to 0-1 range
        pore_map = (pore_map - pore_map.min()) / (pore_map.max() - pore_map.min())
        
        # Apply threshold to create more defined pores
        pore_threshold = 0.6
        pore_map = np.where(pore_map > pore_threshold, 
                           1.0 - (pore_map - pore_threshold) / (1.0 - pore_threshold) * 0.5, 
                           1.0)
        
        # Convert to 3-channel image
        pore_map_rgb = np.stack([pore_map] * 3, axis=-1)
        
        return pore_map_rgb
    
    def _generate_wrinkle_map(self) -> np.ndarray:
        """
        Generate a procedural wrinkle detail map.
        
        Returns:
            Wrinkle detail map as numpy array
        """
        # Create a base canvas
        wrinkle_map = np.ones((self.resolution[1], self.resolution[0]), dtype=np.float32)
        
        # Number of wrinkles to generate
        num_wrinkles = np.random.randint(10, 30)
        
        for _ in range(num_wrinkles):
            # Random wrinkle parameters
            length = np.random.randint(50, 500)
            width = np.random.randint(2, 10)
            depth = np.random.uniform(0.1, 0.5)
            
            # Random position and orientation
            x = np.random.randint(0, self.resolution[0])
            y = np.random.randint(0, self.resolution[1])
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Generate wrinkle curve
            curve_points = []
            curve_length = np.random.randint(3, 10)
            
            for i in range(curve_length):
                t = i / (curve_length - 1)
                # Add some randomness to the curve
                dx = np.random.normal(0, 10)
                dy = np.random.normal(0, 10)
                
                # Calculate point along curve
                px = int(x + np.cos(angle) * length * t + dx)
                py = int(y + np.sin(angle) * length * t + dy)
                
                # Ensure within bounds
                px = max(0, min(px, self.resolution[0] - 1))
                py = max(0, min(py, self.resolution[1] - 1))
                
                curve_points.append((px, py))
            
            # Draw the wrinkle as a curve with varying width
            for i in range(len(curve_points) - 1):
                p1 = curve_points[i]
                p2 = curve_points[i + 1]
                
                # Draw line between points
                cv2.line(wrinkle_map, p1, p2, 1.0 - depth, width, cv2.LINE_AA)
        
        # Apply Gaussian blur to soften wrinkles
        wrinkle_map = cv2.GaussianBlur(wrinkle_map, (5, 5), 1.0)
        
        # Convert to 3-channel image
        wrinkle_map_rgb = np.stack([wrinkle_map] * 3, axis=-1)
        
        return wrinkle_map_rgb
    
    def _generate_fine_details_map(self) -> np.ndarray:
        """
        Generate a procedural fine details map.
        
        Returns:
            Fine details map as numpy array
        """
        # Create a multi-scale noise map for fine details
        fine_details = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        
        # Generate multiple octaves of noise
        scale = 0.05
        for octave in range(5):
            # Generate noise at current scale
            noise_size = (int(self.resolution[1] * scale), int(self.resolution[0] * scale))
            noise = np.random.normal(0, 1, noise_size).astype(np.float32)
            
            # Resize to full resolution
            noise_resized = cv2.resize(noise, self.resolution, interpolation=cv2.INTER_CUBIC)
            
            # Add to fine details map with decreasing weight for higher octaves
            weight = 1.0 / (1.5 ** octave)
            fine_details += noise_resized * weight
            
            # Double the scale for next octave
            scale *= 2
        
        # Normalize to 0-1 range
        fine_details = (fine_details - fine_details.min()) / (fine_details.max() - fine_details.min())
        
        # Apply subtle contrast enhancement
        fine_details = np.power(fine_details, 1.2)
        
        # Convert to 3-channel image
        fine_details_rgb = np.stack([fine_details] * 3, axis=-1)
        
        return fine_details_rgb
    
    def _generate_normal_map(self, 
                           pore_map: np.ndarray, 
                           wrinkle_map: np.ndarray, 
                           fine_details: np.ndarray) -> np.ndarray:
        """
        Generate a normal map from detail maps.
        
        Args:
            pore_map: Pore detail map
            wrinkle_map: Wrinkle detail map
            fine_details: Fine details map
            
        Returns:
            Normal map as numpy array
        """
        # Convert detail maps to grayscale if they're RGB
        if pore_map.ndim == 3:
            pore_gray = cv2.cvtColor(pore_map, cv2.COLOR_RGB2GRAY)
        else:
            pore_gray = pore_map
            
        if wrinkle_map.ndim == 3:
            wrinkle_gray = cv2.cvtColor(wrinkle_map, cv2.COLOR_RGB2GRAY)
        else:
            wrinkle_gray = wrinkle_map
            
        if fine_details.ndim == 3:
            fine_details_gray = cv2.cvtColor(fine_details, cv2.COLOR_RGB2GRAY)
        else:
            fine_details_gray = fine_details
        
        # Combine detail maps with different weights
        combined = (
            pore_gray * 0.4 + 
            wrinkle_gray * 0.4 + 
            fine_details_gray * 0.2
        )
        
        # Normalize to 0-1
        combined = (combined - combined.min()) / (combined.max() - combined.min())
        
        # Create a height map from combined details
        height_map = combined
        
        # Compute gradients for normal map
        sobelx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
        
        # Scale gradients
        scale = 5.0  # Adjust for normal map strength
        sobelx = sobelx * scale
        sobely = sobely * scale
        
        # Create normal map
        normal_map = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.float32)
        normal_map[:, :, 0] = -sobelx
        normal_map[:, :, 1] = -sobely
        normal_map[:, :, 2] = 1.0
        
        # Normalize vectors
        norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
        normal_map = normal_map / (norm + 1e-10)
        
        # Convert to 0-1 range for visualization
        normal_map = (normal_map + 1.0) / 2.0
        
        # Convert to 8-bit
        normal_map = (normal_map * 255).astype(np.uint8)
        
        return normal_map
    
    def apply_details_to_texture(self, 
                               texture: np.ndarray, 
                               detail_strength: float = 1.0,
                               detail_types: List[str] = ['pores', 'wrinkles', 'fine_details']) -> np.ndarray:
        """
        Apply detail maps to a base texture.
        
        Args:
            texture: Base texture image
            detail_strength: Strength of detail application (0-1)
            detail_types: List of detail types to apply
            
        Returns:
            Texture with details applied
        """
        # Resize texture to match detail map resolution if needed
        if texture.shape[0] != self.resolution[1] or texture.shape[1] != self.resolution[0]:
            texture = cv2.resize(texture, self.resolution, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to float for processing
        texture_float = texture.astype(np.float32) / 255.0
        
        # Create a detail mask (where to apply details)
        detail_mask = np.ones_like(texture_float)
        
        # Apply each detail type
        for detail_type in detail_types:
            if detail_type in self.detail_maps:
                detail_map = self.detail_maps[detail_type].astype(np.float32) / 255.0
                
                # Apply detail map with blending
                if detail_type == 'pores':
                    # Multiplicative blending for pores
                    texture_float = texture_float * (1.0 - detail_strength * 0.2) + \
                                   texture_float * detail_map * detail_strength * 0.2
                elif detail_type == 'wrinkles':
                    # Overlay blending for wrinkles
                    texture_float = texture_float * (1.0 - detail_strength * 0.3) + \
                                   detail_map * detail_strength * 0.3
                elif detail_type == 'fine_details':
                    # Add fine details with screen blending
                    texture_float = 1.0 - (1.0 - texture_float) * (1.0 - detail_map * detail_strength * 0.1)
        
        # Apply normal map if available
        if 'normal_map' in self.detail_maps and 'normal_map' in detail_types:
            normal_map = self.detail_maps['normal_map'].astype(np.float32) / 255.0
            
            # Create a lighting effect based on the normal map
            light_dir = np.array([0.5, 0.5, 1.0]) / np.sqrt(1.5)  # Normalized light direction
            
            # Convert normal map from 0-1 to -1 to 1 range
            normals = normal_map * 2.0 - 1.0
            
            # Calculate lighting (dot product of normal and light direction)
            lighting = np.sum(normals * light_dir, axis=2)
            lighting = np.clip(lighting, 0, 1)
            
            # Apply lighting to texture
            lighting = lighting.reshape(lighting.shape[0], lighting.shape[1], 1)
            texture_float = texture_float * (1.0 - detail_strength * 0.4) + \
                           texture_float * lighting * detail_strength * 0.4
        
        # Convert back to 8-bit
        detailed_texture = np.clip(texture_float * 255.0, 0, 255).astype(np.uint8)
        
        return detailed_texture
    
    def generate_displacement_map(self, 
                                detail_types: List[str] = ['pores', 'wrinkles'],
                                strength: float = 1.0) -> np.ndarray:
        """
        Generate a displacement map for 3D geometry detail.
        
        Args:
            detail_types: List of detail types to include
            strength: Strength of displacement
            
        Returns:
            Displacement map as numpy array
        """
        # Create a base displacement map
        displacement = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        
        # Add each detail type to the displacement
        for detail_type in detail_types:
            if detail_type in self.detail_maps:
                # Convert to grayscale if RGB
                if self.detail_maps[detail_type].ndim == 3:
                    detail_gray = cv2.cvtColor(self.detail_maps[detail_type], cv2.COLOR_RGB2GRAY)
                else:
                    detail_gray = self.detail_maps[detail_type]
                
                # Normalize to 0-1
                detail_gray = detail_gray.astype(np.float32) / 255.0
                
                # Add to displacement with type-specific weight
                if detail_type == 'pores':
                    displacement += (1.0 - detail_gray) * 0.3
                elif detail_type == 'wrinkles':
                    displacement += (1.0 - detail_gray) * 0.7
                elif detail_type == 'fine_details':
                    displacement += (detail_gray - 0.5) * 0.1
        
        # Normalize to 0-1 range
        displacement = (displacement - displacement.min()) / (displacement.max() - displacement.min())
        
        # Apply strength factor
        displacement = displacement * strength
        
        # Convert to 8-bit for storage
        displacement_map = (displacement * 255).astype(np.uint8)
        
        return displacement_map
    
    def save_detail_maps(self, output_dir: str) -> Dict[str, str]:
        """
        Save all detail maps to disk.
        
        Args:
            output_dir: Directory to save maps
            
        Returns:
            Dictionary of saved file paths
        """
        ensure_directory(output_dir)
        
        saved_paths = {}
        
        # Save each detail map
        for map_type, detail_map in self.detail_maps.items():
            save_path = os.path.join(output_dir, f"{map_type}.png")
            
            # Save as image
            cv2.imwrite(save_path, cv2.cvtColor(detail_map, cv2.COLOR_RGB2BGR))
            
            saved_paths[map_type] = save_path
        
        return saved_paths
    
    def create_detail_preview(self, size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """
        Create a preview image showing all detail maps.
        
        Args:
            size: Size of the preview image
            
        Returns:
            Preview image as numpy array
        """
        # Determine grid layout
        num_maps = len(self.detail_maps)
        grid_size = int(np.ceil(np.sqrt(num_maps)))
        
        # Calculate individual thumbnail size
        thumb_width = size[0] // grid_size
        thumb_height = size[1] // grid_size
        
        # Create canvas
        preview = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Add each detail map
        for i, (map_type, detail_map) in enumerate(self.detail_maps.items()):
            # Calculate position
            row = i // grid_size
            col = i % grid_size
            
            # Resize detail map to thumbnail size
            thumb = cv2.resize(detail_map, (thumb_width, thumb_height))
            
            # Place in grid
            y_start = row * thumb_height
            y_end = y_start + thumb_height
            x_start = col * thumb_width
            x_end = x_start + thumb_width
            
            # Ensure within bounds
            y_end = min(y_end, size[1])
            x_end = min(x_end, size[0])
            
            # Copy to preview
            preview[y_start:y_end, x_start:x_end] = thumb[:y_end-y_start, :x_end-x_start]
            
            # Add label
            cv2.putText(preview, map_type, (x_start + 5, y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return preview
