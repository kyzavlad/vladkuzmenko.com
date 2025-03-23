import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open3d as o3d
import trimesh

from app.avatar_creation.face_modeling.utils import (
    tensor_to_image,
    image_to_tensor,
    get_device,
    ensure_directory
)

class ExpressionCalibration:
    """
    Class for calibrating and generating facial expressions for 3D face models.
    Supports blendshape generation and expression transfer.
    """
    
    def __init__(self, blendshape_dir: Optional[str] = None):
        """
        Initialize the expression calibration system.
        
        Args:
            blendshape_dir: Directory containing base blendshapes (optional)
        """
        self.device = get_device()
        self.blendshapes = {}
        self.expression_params = {}
        
        # Load blendshapes if directory is provided
        if blendshape_dir and os.path.exists(blendshape_dir):
            self._load_blendshapes(blendshape_dir)
    
    def _load_blendshapes(self, blendshape_dir: str) -> None:
        """
        Load blendshapes from directory.
        
        Args:
            blendshape_dir: Directory containing blendshape meshes
        """
        # Common expression blendshapes
        expression_names = [
            'neutral', 'smile', 'frown', 'surprise', 'anger', 
            'jaw_open', 'jaw_left', 'jaw_right', 'eye_blink_left', 
            'eye_blink_right', 'brow_up', 'brow_down'
        ]
        
        for expr_name in expression_names:
            # Look for OBJ or PLY files
            for ext in ['.obj', '.ply']:
                mesh_path = os.path.join(blendshape_dir, f"{expr_name}{ext}")
                if os.path.exists(mesh_path):
                    # Load the mesh
                    if ext == '.obj':
                        mesh = trimesh.load(mesh_path)
                    else:  # .ply
                        mesh = o3d.io.read_triangle_mesh(mesh_path)
                        # Convert to trimesh for consistent handling
                        vertices = np.asarray(mesh.vertices)
                        triangles = np.asarray(mesh.triangles)
                        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                    
                    # Store the blendshape
                    self.blendshapes[expr_name] = mesh
                    break
    
    def create_basic_blendshapes(self, base_mesh: Union[o3d.geometry.TriangleMesh, trimesh.Trimesh]) -> Dict:
        """
        Create basic blendshapes from a neutral face mesh.
        
        Args:
            base_mesh: Neutral face mesh
            
        Returns:
            Dictionary of generated blendshapes
        """
        # Convert to trimesh if needed
        if isinstance(base_mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(base_mesh.vertices)
            triangles = np.asarray(base_mesh.triangles)
            base_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        else:
            base_trimesh = base_mesh
        
        # Store neutral expression
        self.blendshapes['neutral'] = base_trimesh
        
        # Define regions for different expressions
        # These are simplified approximations - a real implementation would use
        # more sophisticated facial region mapping
        
        # Get vertex positions
        vertices = np.array(base_trimesh.vertices)
        
        # Compute bounding box to help with region identification
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        center = (bbox_min + bbox_max) / 2
        
        # Approximate facial regions based on position
        # This is a simplified approach - a real implementation would use
        # semantic facial region mapping
        
        # Create smile blendshape
        smile_vertices = vertices.copy()
        for i, v in enumerate(vertices):
            # Mouth region (lower face)
            if v[1] < center[1] and abs(v[0] - center[0]) < (bbox_max[0] - bbox_min[0]) * 0.3:
                # Move corners up and out for smile
                if v[0] < center[0]:  # Left side
                    smile_vertices[i, 0] -= (center[1] - v[1]) * 0.1  # Move left
                else:  # Right side
                    smile_vertices[i, 0] += (center[1] - v[1]) * 0.1  # Move right
                
                # Move up
                smile_vertices[i, 1] += (center[1] - v[1]) * 0.2
        
        # Create smile mesh
        smile_mesh = trimesh.Trimesh(vertices=smile_vertices, faces=base_trimesh.faces)
        self.blendshapes['smile'] = smile_mesh
        
        # Create surprise blendshape
        surprise_vertices = vertices.copy()
        for i, v in enumerate(vertices):
            # Mouth region
            if v[1] < center[1] and abs(v[0] - center[0]) < (bbox_max[0] - bbox_min[0]) * 0.2:
                # Open mouth for surprise
                surprise_vertices[i, 1] -= (center[1] - v[1]) * 0.3
            
            # Eye region
            if v[1] > center[1] and abs(v[0] - center[0]) < (bbox_max[0] - bbox_min[0]) * 0.4:
                # Raise eyebrows
                surprise_vertices[i, 1] += (v[1] - center[1]) * 0.2
        
        # Create surprise mesh
        surprise_mesh = trimesh.Trimesh(vertices=surprise_vertices, faces=base_trimesh.faces)
        self.blendshapes['surprise'] = surprise_mesh
        
        # Create more blendshapes for other expressions
        # This is a simplified implementation - a real system would use
        # more sophisticated deformation techniques
        
        # Create jaw open blendshape
        jaw_open_vertices = vertices.copy()
        for i, v in enumerate(vertices):
            # Lower jaw region
            if v[1] < center[1]:
                # Move jaw down
                jaw_open_vertices[i, 1] -= (center[1] - v[1]) * 0.4
        
        # Create jaw open mesh
        jaw_open_mesh = trimesh.Trimesh(vertices=jaw_open_vertices, faces=base_trimesh.faces)
        self.blendshapes['jaw_open'] = jaw_open_mesh
        
        # Return the created blendshapes
        return self.blendshapes
    
    def blend_expressions(self, 
                        weights: Dict[str, float]) -> trimesh.Trimesh:
        """
        Blend multiple expressions based on weights.
        
        Args:
            weights: Dictionary mapping expression names to blend weights
            
        Returns:
            Blended mesh
        """
        if not self.blendshapes or 'neutral' not in self.blendshapes:
            raise ValueError("No blendshapes available. Load or create blendshapes first.")
        
        # Start with neutral expression
        neutral_vertices = np.array(self.blendshapes['neutral'].vertices)
        result_vertices = neutral_vertices.copy()
        
        # Normalize weights to sum to 1.0 (excluding neutral)
        non_neutral_weights = {k: v for k, v in weights.items() if k != 'neutral'}
        weight_sum = sum(non_neutral_weights.values())
        
        if weight_sum > 0:
            normalized_weights = {k: v / weight_sum for k, v in non_neutral_weights.items()}
            
            # Apply each blendshape
            for expr_name, weight in normalized_weights.items():
                if expr_name in self.blendshapes and weight > 0:
                    # Get the blendshape vertices
                    expr_vertices = np.array(self.blendshapes[expr_name].vertices)
                    
                    # Compute the displacement from neutral
                    displacement = expr_vertices - neutral_vertices
                    
                    # Add weighted displacement to result
                    result_vertices += displacement * weight
        
        # Create the blended mesh
        blended_mesh = trimesh.Trimesh(vertices=result_vertices, faces=self.blendshapes['neutral'].faces)
        
        return blended_mesh
    
    def create_expression_sequence(self, 
                                 start_expr: str, 
                                 end_expr: str, 
                                 num_frames: int = 10) -> List[trimesh.Trimesh]:
        """
        Create a sequence of meshes transitioning between expressions.
        
        Args:
            start_expr: Starting expression name
            end_expr: Ending expression name
            num_frames: Number of frames in the sequence
            
        Returns:
            List of meshes representing the expression sequence
        """
        if start_expr not in self.blendshapes or end_expr not in self.blendshapes:
            raise ValueError(f"Expression '{start_expr}' or '{end_expr}' not found in blendshapes")
        
        # Create sequence of meshes
        sequence = []
        
        for i in range(num_frames):
            # Calculate interpolation factor
            t = i / (num_frames - 1) if num_frames > 1 else 0
            
            # Create weights for blending
            weights = {
                start_expr: 1.0 - t,
                end_expr: t
            }
            
            # Blend expressions
            blended_mesh = self.blend_expressions(weights)
            sequence.append(blended_mesh)
        
        return sequence
    
    def detect_expression_from_image(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect facial expression from an image and estimate blendshape weights.
        
        Args:
            image: Input face image
            
        Returns:
            Dictionary of expression weights
        """
        # This is a placeholder for a real expression detection system
        # In a real implementation, use a facial expression recognition model
        
        # Default to neutral expression
        expression_weights = {
            'neutral': 1.0,
            'smile': 0.0,
            'surprise': 0.0,
            'jaw_open': 0.0
        }
        
        # Convert to grayscale for processing
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize for processing
            face_roi = cv2.resize(face_roi, (64, 64))
            
            # Simple feature extraction (this is a placeholder)
            # In a real implementation, use a proper facial expression recognition model
            
            # Check for smile (simple edge detection in lower face)
            lower_face = face_roi[32:, :]
            edges = cv2.Canny(lower_face, 100, 200)
            smile_score = np.sum(edges) / (lower_face.size * 255)
            
            # Check for surprise (look for open mouth and raised eyebrows)
            upper_face = face_roi[:32, :]
            upper_edges = cv2.Canny(upper_face, 100, 200)
            surprise_score = np.sum(upper_edges) / (upper_face.size * 255)
            
            # Check for open jaw
            jaw_open_score = np.mean(face_roi[48:, 16:48]) / 255.0
            
            # Update expression weights based on scores
            # These thresholds are arbitrary and would need calibration
            if smile_score > 0.1:
                expression_weights['neutral'] = 0.3
                expression_weights['smile'] = 0.7
                
            if surprise_score > 0.15:
                expression_weights['neutral'] = 0.2
                expression_weights['surprise'] = 0.8
                
            if jaw_open_score < 0.4:  # Darker region indicates open mouth
                expression_weights['jaw_open'] = 0.6
                expression_weights['neutral'] = 0.4
        
        return expression_weights
    
    def transfer_expression(self, 
                          source_mesh: trimesh.Trimesh, 
                          target_neutral_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Transfer expression from source mesh to target neutral mesh.
        
        Args:
            source_mesh: Source mesh with expression
            target_neutral_mesh: Target neutral mesh
            
        Returns:
            Target mesh with transferred expression
        """
        if 'neutral' not in self.blendshapes:
            raise ValueError("Neutral blendshape not available. Load or create blendshapes first.")
        
        # Get vertices
        source_vertices = np.array(source_mesh.vertices)
        source_neutral_vertices = np.array(self.blendshapes['neutral'].vertices)
        target_vertices = np.array(target_neutral_mesh.vertices)
        
        # Compute displacement from neutral to expression in source
        source_displacement = source_vertices - source_neutral_vertices
        
        # Apply displacement to target
        # This is a simplified approach - a real implementation would use
        # correspondence mapping between source and target
        target_result_vertices = target_vertices + source_displacement
        
        # Create result mesh
        result_mesh = trimesh.Trimesh(vertices=target_result_vertices, faces=target_neutral_mesh.faces)
        
        return result_mesh
    
    def calibrate_expressions(self, 
                            reference_images: Dict[str, np.ndarray], 
                            base_mesh: trimesh.Trimesh) -> Dict[str, trimesh.Trimesh]:
        """
        Calibrate expressions based on reference images.
        
        Args:
            reference_images: Dictionary mapping expression names to reference images
            base_mesh: Base neutral mesh
            
        Returns:
            Dictionary of calibrated expression blendshapes
        """
        # Store the neutral mesh
        self.blendshapes['neutral'] = base_mesh
        
        # Process each reference image
        for expr_name, image in reference_images.items():
            if expr_name == 'neutral':
                continue  # Skip neutral, we already have it
            
            # Detect facial landmarks in the image
            # This is a placeholder - a real implementation would use
            # a proper facial landmark detection system
            
            # For demonstration, we'll create a simple deformation
            # based on the expression name
            vertices = np.array(base_mesh.vertices)
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            center = (bbox_min + bbox_max) / 2
            
            # Apply different deformations based on expression
            if expr_name == 'smile':
                # Create smile deformation
                for i, v in enumerate(vertices):
                    # Mouth region
                    if v[1] < center[1] and abs(v[0] - center[0]) < (bbox_max[0] - bbox_min[0]) * 0.3:
                        # Move corners up and out for smile
                        if v[0] < center[0]:  # Left side
                            vertices[i, 0] -= (center[1] - v[1]) * 0.1
                        else:  # Right side
                            vertices[i, 0] += (center[1] - v[1]) * 0.1
                        
                        # Move up
                        vertices[i, 1] += (center[1] - v[1]) * 0.2
            
            elif expr_name == 'surprise':
                # Create surprise deformation
                for i, v in enumerate(vertices):
                    # Mouth region
                    if v[1] < center[1] and abs(v[0] - center[0]) < (bbox_max[0] - bbox_min[0]) * 0.2:
                        # Open mouth for surprise
                        vertices[i, 1] -= (center[1] - v[1]) * 0.3
                    
                    # Eye region
                    if v[1] > center[1] and abs(v[0] - center[0]) < (bbox_max[0] - bbox_min[0]) * 0.4:
                        # Raise eyebrows
                        vertices[i, 1] += (v[1] - center[1]) * 0.2
            
            # Create the expression mesh
            expr_mesh = trimesh.Trimesh(vertices=vertices, faces=base_mesh.faces)
            self.blendshapes[expr_name] = expr_mesh
        
        return self.blendshapes
    
    def save_blendshapes(self, output_dir: str) -> Dict[str, str]:
        """
        Save all blendshapes to disk.
        
        Args:
            output_dir: Directory to save blendshapes
            
        Returns:
            Dictionary of saved file paths
        """
        ensure_directory(output_dir)
        
        saved_paths = {}
        
        # Save each blendshape
        for expr_name, mesh in self.blendshapes.items():
            save_path = os.path.join(output_dir, f"{expr_name}.obj")
            
            # Save as OBJ
            mesh.export(save_path)
            
            saved_paths[expr_name] = save_path
        
        return saved_paths
    
    def create_expression_visualization(self, 
                                      expression_name: str, 
                                      size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Create a visualization of an expression blendshape.
        
        Args:
            expression_name: Name of the expression to visualize
            size: Size of the visualization image
            
        Returns:
            Visualization image
        """
        if expression_name not in self.blendshapes:
            raise ValueError(f"Expression '{expression_name}' not found in blendshapes")
        
        # Get the mesh
        mesh = self.blendshapes[expression_name]
        
        # Create a simple visualization
        # In a real implementation, use a proper 3D renderer
        
        # Create a blank image
        image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        
        # Project 3D vertices to 2D
        vertices = np.array(mesh.vertices)
        
        # Simple orthographic projection
        # Scale and center the projection
        x_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        scale = min(size[0] / x_range, size[1] / y_range) * 0.8
        
        x_offset = size[0] / 2 - np.mean(vertices[:, 0]) * scale
        y_offset = size[1] / 2 - np.mean(vertices[:, 1]) * scale
        
        points_2d = []
        for v in vertices:
            x = int(v[0] * scale + x_offset)
            y = int(v[1] * scale + y_offset)
            points_2d.append((x, y))
        
        # Draw edges
        for edge in mesh.edges:
            p1 = points_2d[edge[0]]
            p2 = points_2d[edge[1]]
            cv2.line(image, p1, p2, (0, 0, 0), 1)
        
        # Add label
        cv2.putText(image, expression_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return image
