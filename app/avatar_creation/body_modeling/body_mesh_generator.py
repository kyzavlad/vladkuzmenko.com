import os
import numpy as np
import torch
import trimesh
from typing import Dict, List, Tuple, Union, Optional
import json

from app.avatar_creation.face_modeling.utils import (
    get_device,
    ensure_directory
)

class BodyMeshGenerator:
    """
    Class for generating 3D body meshes based on measurements.
    Supports SMPL model integration if available, or fallback procedural generation.
    """
    
    def __init__(self, 
                model_path: Optional[str] = None,
                gender: str = 'neutral',
                use_smpl: bool = True,
                high_quality: bool = False):
        """
        Initialize the body mesh generator.
        
        Args:
            model_path: Path to SMPL model files
            gender: Gender for body model ('male', 'female', or 'neutral')
            use_smpl: Whether to use SMPL model
            high_quality: Whether to generate high-quality meshes
        """
        self.device = get_device()
        self.model_path = model_path
        self.gender = gender.lower()
        self.use_smpl = use_smpl
        self.high_quality = high_quality
        
        # Validate gender
        if self.gender not in ['male', 'female', 'neutral']:
            print(f"Invalid gender '{gender}', defaulting to 'neutral'")
            self.gender = 'neutral'
        
        # Initialize body model
        self.body_model = self._initialize_body_model()
        
        # Default shape parameters
        self.default_shape_params = torch.zeros(10, device=self.device)
        
        print(f"Initialized BodyMeshGenerator (gender={gender}, use_smpl={use_smpl})")
    
    def _initialize_body_model(self):
        """
        Initialize the body model (SMPL or alternative).
        
        Returns:
            Initialized model or None if not available
        """
        # Check if SMPL should be used
        if self.use_smpl:
            try:
                # Try to import SMPL
                try:
                    # Try importing official SMPL first
                    import smplx
                    
                    # Set model type based on gender
                    model_type = 'smpl'
                    
                    # Set gender
                    gender = self.gender
                    if gender == 'neutral':
                        gender = 'neutral'  # SMPL uses 'neutral' for ungendered model
                    
                    # Check if model path exists
                    if self.model_path and os.path.exists(self.model_path):
                        # Initialize SMPL model
                        model = smplx.create(
                            self.model_path,
                            model_type=model_type,
                            gender=gender,
                            use_face_contour=True,
                            batch_size=1
                        ).to(self.device)
                        
                        print(f"Loaded SMPL model (gender={gender})")
                        return model
                    else:
                        print(f"SMPL model path not found: {self.model_path}")
                        return None
                
                except ImportError:
                    # If official SMPL fails, try simplified version
                    print("Official SMPL not available, trying simplified implementation")
                    
                    return self._initialize_simplified_body_model()
            
            except Exception as e:
                print(f"Error initializing SMPL: {e}")
                return None
        
        # If SMPL is not used or failed to load, use simplified model
        return self._initialize_simplified_body_model()
    
    def _initialize_simplified_body_model(self):
        """
        Initialize a simplified body model when SMPL is not available.
        
        Returns:
            Simplified body model
        """
        print("Using simplified body model")
        
        # This is a placeholder for a simplified body model
        # In a real implementation, you might use a simpler parametric model
        
        # Create a simple dictionary-based model
        model = {
            'type': 'simplified',
            'gender': self.gender,
            'vertices': None,  # Will be filled on generation
            'faces': None,     # Will be filled on generation
            'shape_params': self.default_shape_params.cpu().numpy()
        }
        
        return model
    
    def generate_mesh_from_measurements(self, 
                                     measurements: Dict,
                                     pose_params: Optional[np.ndarray] = None) -> trimesh.Trimesh:
        """
        Generate a body mesh based on measurement data.
        
        Args:
            measurements: Dictionary containing body measurements
            pose_params: Optional pose parameters for the model
            
        Returns:
            Body mesh as a trimesh.Trimesh object
        """
        # Convert measurements to shape parameters
        shape_params = self._measurements_to_shape_params(measurements)
        
        # Default pose (T-pose)
        if pose_params is None:
            pose_params = np.zeros(72)  # SMPL uses 72 pose parameters for 24 joints
            pose_params[0] = np.pi  # Global rotation
        
        # Generate the mesh
        if self.use_smpl and self.body_model is not None and not isinstance(self.body_model, dict):
            # Generate mesh using SMPL
            mesh = self._generate_mesh_smpl(shape_params, pose_params)
        else:
            # Generate mesh using simplified approach
            mesh = self._generate_mesh_simplified(shape_params, pose_params, measurements)
        
        return mesh
    
    def _measurements_to_shape_params(self, measurements: Dict) -> np.ndarray:
        """
        Convert measurements to shape parameters for the body model.
        
        Args:
            measurements: Dictionary containing body measurements
            
        Returns:
            Shape parameters (beta) for the model
        """
        # Default shape parameters
        shape_params = self.default_shape_params.clone().detach().cpu().numpy()
        
        # This is a simplified implementation
        # In a real implementation, you would use a proper regression model
        
        # Check if we have necessary measurements
        if not all(key in measurements for key in 
                 ['height', 'chest_circumference', 'waist_circumference', 'hip_circumference']):
            print("Missing essential measurements, using default shape")
            return shape_params
        
        try:
            # Extract key measurements
            height = measurements.get('height', 170.0)  # cm
            chest = measurements.get('chest_circumference', 90.0)  # cm
            waist = measurements.get('waist_circumference', 80.0)  # cm
            hip = measurements.get('hip_circumference', 95.0)  # cm
            shoulder = measurements.get('shoulder_width', 40.0)  # cm
            
            # Normalize measurements (based on standard average values)
            height_norm = (height - 170.0) / 15.0
            chest_norm = (chest - 90.0) / 10.0
            waist_norm = (waist - 80.0) / 10.0
            hip_norm = (hip - 95.0) / 10.0
            shoulder_norm = (shoulder - 40.0) / 5.0
            
            # Set shape parameters based on normalized measurements
            # These are simplified approximations and would need proper calibration
            # in a real implementation
            
            # Height primarily affects parameter 0
            shape_params[0] = height_norm
            
            # Weight distribution affects parameters 1, 2
            weight_factor = (chest_norm + waist_norm + hip_norm) / 3.0
            shape_params[1] = weight_factor * 1.5
            
            # Chest/hip ratio affects parameter 2
            shape_params[2] = (chest_norm - hip_norm) * 1.2
            
            # Waist affects parameter 3
            shape_params[3] = waist_norm * 1.5
            
            # Shoulder width affects parameter 4
            shape_params[4] = shoulder_norm
            
            # Gender-specific adjustments
            if self.gender == 'male':
                # Males tend to have broader shoulders, narrower hips
                shape_params[2] += 0.5
                shape_params[4] += 0.3
            elif self.gender == 'female':
                # Females tend to have narrower shoulders, wider hips
                shape_params[2] -= 0.5
                shape_params[3] += 0.3
            
            # Limb lengths
            if 'arm_length' in measurements:
                arm_norm = (measurements['arm_length'] - (height * 0.33)) / 5.0
                shape_params[5] = arm_norm
            
            if 'inseam' in measurements:
                leg_norm = (measurements['inseam'] - (height * 0.45)) / 5.0
                shape_params[6] = leg_norm
            
        except Exception as e:
            print(f"Error converting measurements to shape parameters: {e}")
        
        # Ensure parameters are within reasonable bounds
        shape_params = np.clip(shape_params, -3.0, 3.0)
        
        return shape_params
    
    def _generate_mesh_smpl(self, 
                          shape_params: np.ndarray,
                          pose_params: np.ndarray) -> trimesh.Trimesh:
        """
        Generate a body mesh using the SMPL model.
        
        Args:
            shape_params: Shape parameters (beta) for the model
            pose_params: Pose parameters (theta) for the model
            
        Returns:
            Body mesh as a trimesh.Trimesh object
        """
        try:
            # Convert numpy arrays to torch tensors
            betas = torch.tensor(shape_params, dtype=torch.float32, device=self.device).unsqueeze(0)
            pose = torch.tensor(pose_params, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Generate model output
            output = self.body_model(betas=betas, body_pose=pose[:, 3:], global_orient=pose[:, :3])
            
            # Extract vertices and faces
            vertices = output.vertices.detach().cpu().numpy()[0]
            faces = self.body_model.faces
            
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            return mesh
            
        except Exception as e:
            print(f"Error generating SMPL mesh: {e}")
            
            # Fall back to simplified method
            return self._generate_mesh_simplified(shape_params, pose_params)
    
    def _generate_mesh_simplified(self, 
                                shape_params: np.ndarray,
                                pose_params: np.ndarray,
                                measurements: Optional[Dict] = None) -> trimesh.Trimesh:
        """
        Generate a simplified body mesh when SMPL is not available.
        
        Args:
            shape_params: Shape parameters
            pose_params: Pose parameters
            measurements: Optional raw measurements for backup
            
        Returns:
            Body mesh as a trimesh.Trimesh object
        """
        # This is a placeholder implementation that creates a simple parametric mesh
        # In a real implementation, you would use a more sophisticated approach
        
        # Load a base mesh or create a simple one
        try:
            # Try to load a base mesh if available
            base_mesh_path = os.path.join(os.path.dirname(__file__), "assets", "base_mesh.obj")
            if os.path.exists(base_mesh_path):
                base_mesh = trimesh.load(base_mesh_path)
                print(f"Loaded base mesh from {base_mesh_path}")
            else:
                print("Base mesh not found, creating primitive")
                base_mesh = self._create_primitive_body()
        except Exception as e:
            print(f"Error loading base mesh: {e}")
            base_mesh = self._create_primitive_body()
        
        # Apply shape deformations based on shape parameters
        mesh = self._apply_shape_deformations(base_mesh, shape_params, measurements)
        
        # Apply pose (simplified)
        mesh = self._apply_simplified_pose(mesh, pose_params)
        
        return mesh
    
    def _create_primitive_body(self) -> trimesh.Trimesh:
        """
        Create a primitive body mesh as a fallback.
        
        Returns:
            Primitive body mesh
        """
        # Create a very simplified body using primitives
        # This is just a placeholder - a real implementation would use a proper base mesh
        
        # Create body parts
        torso = trimesh.creation.cylinder(radius=0.15, height=0.6)
        head = trimesh.creation.icosphere(radius=0.1)
        
        # Position head
        head.vertices += [0, 0.4, 0]
        
        # Create limbs
        arm_right = trimesh.creation.cylinder(radius=0.05, height=0.6)
        arm_right.vertices += [0.25, 0.2, 0]
        arm_right.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
        
        arm_left = trimesh.creation.cylinder(radius=0.05, height=0.6)
        arm_left.vertices += [-0.25, 0.2, 0]
        arm_left.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 0, 1]))
        
        leg_right = trimesh.creation.cylinder(radius=0.07, height=0.7)
        leg_right.vertices += [0.1, -0.55, 0]
        
        leg_left = trimesh.creation.cylinder(radius=0.07, height=0.7)
        leg_left.vertices += [-0.1, -0.55, 0]
        
        # Combine into a single mesh
        body = trimesh.util.concatenate([torso, head, arm_right, arm_left, leg_right, leg_left])
        
        return body
    
    def _apply_shape_deformations(self, 
                               mesh: trimesh.Trimesh, 
                               shape_params: np.ndarray,
                               measurements: Optional[Dict] = None) -> trimesh.Trimesh:
        """
        Apply shape deformations to a base mesh.
        
        Args:
            mesh: Input mesh
            shape_params: Shape parameters
            measurements: Optional raw measurements
            
        Returns:
            Deformed mesh
        """
        # Make a copy to avoid modifying the original
        mesh = mesh.copy()
        
        # Get mesh bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        height = bounds[1, 1] - bounds[0, 1]
        width = bounds[1, 0] - bounds[0, 0]
        depth = bounds[1, 2] - bounds[0, 2]
        
        # Center of the mesh
        center = np.mean(bounds, axis=0)
        
        # Scale factors based on shape parameters
        height_scale = 1.0 + shape_params[0] * 0.15  # Height
        width_scale = 1.0 + shape_params[1] * 0.1    # Overall width
        depth_scale = 1.0 + shape_params[1] * 0.1    # Overall depth
        
        # Apply global scaling (keep centered)
        mesh.vertices = (mesh.vertices - center) * [width_scale, height_scale, depth_scale] + center
        
        # Gender-specific shape adjustments
        if self.gender == 'male':
            # For male: broader shoulders, narrower hips
            self._adjust_shoulder_hip_ratio(mesh, 1.1, 0.9)
        elif self.gender == 'female':
            # For female: narrower shoulders, wider hips
            self._adjust_shoulder_hip_ratio(mesh, 0.9, 1.1)
        
        # Weight distribution based on parameters
        if abs(shape_params[1]) > 0.5:
            # Adjust torso width based on weight parameter
            self._adjust_torso_width(mesh, 1.0 + shape_params[1] * 0.1)
        
        # Waist adjustment
        if abs(shape_params[3]) > 0.3:
            self._adjust_waist(mesh, 1.0 - shape_params[3] * 0.1)
        
        # Arm length
        if abs(shape_params[5]) > 0.3:
            self._adjust_arm_length(mesh, 1.0 + shape_params[5] * 0.1)
        
        # Leg length
        if abs(shape_params[6]) > 0.3:
            self._adjust_leg_length(mesh, 1.0 + shape_params[6] * 0.1)
        
        return mesh
    
    def _adjust_shoulder_hip_ratio(self, 
                                mesh: trimesh.Trimesh, 
                                shoulder_factor: float, 
                                hip_factor: float) -> None:
        """
        Adjust the shoulder-to-hip ratio of a mesh.
        
        Args:
            mesh: Input mesh to modify in-place
            shoulder_factor: Scale factor for shoulders
            hip_factor: Scale factor for hips
        """
        # Get mesh bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        height = bounds[1, 1] - bounds[0, 1]
        
        # Shoulder height (approximately)
        shoulder_y = bounds[0, 1] + height * 0.75
        hip_y = bounds[0, 1] + height * 0.45
        
        # Adjust vertices
        for i, v in enumerate(mesh.vertices):
            y = v[1]
            
            # Shoulder region
            if abs(y - shoulder_y) < height * 0.1:
                # Scale x-coordinate (width)
                dist_factor = 1.0 - abs(y - shoulder_y) / (height * 0.1)
                scale = 1.0 + (shoulder_factor - 1.0) * dist_factor
                mesh.vertices[i, 0] *= scale
            
            # Hip region
            if abs(y - hip_y) < height * 0.1:
                # Scale x-coordinate (width)
                dist_factor = 1.0 - abs(y - hip_y) / (height * 0.1)
                scale = 1.0 + (hip_factor - 1.0) * dist_factor
                mesh.vertices[i, 0] *= scale
    
    def _adjust_torso_width(self, mesh: trimesh.Trimesh, factor: float) -> None:
        """
        Adjust the width of the torso.
        
        Args:
            mesh: Input mesh to modify in-place
            factor: Scale factor for torso width
        """
        # Get mesh bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        height = bounds[1, 1] - bounds[0, 1]
        
        # Torso region (approximately)
        torso_bottom = bounds[0, 1] + height * 0.45  # Hip level
        torso_top = bounds[0, 1] + height * 0.75     # Shoulder level
        
        # Adjust vertices in torso region
        for i, v in enumerate(mesh.vertices):
            y = v[1]
            
            # Check if vertex is in torso region
            if y >= torso_bottom and y <= torso_top:
                # Scale x and z coordinates (width and depth)
                mesh.vertices[i, 0] *= factor
                mesh.vertices[i, 2] *= factor
    
    def _adjust_waist(self, mesh: trimesh.Trimesh, factor: float) -> None:
        """
        Adjust the waist circumference.
        
        Args:
            mesh: Input mesh to modify in-place
            factor: Scale factor for waist width
        """
        # Get mesh bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        height = bounds[1, 1] - bounds[0, 1]
        
        # Waist height (approximately)
        waist_y = bounds[0, 1] + height * 0.55
        
        # Adjust vertices in waist region
        for i, v in enumerate(mesh.vertices):
            y = v[1]
            
            # Check if vertex is in waist region
            dist = abs(y - waist_y)
            if dist < height * 0.07:
                # Scale based on distance from waist
                dist_factor = 1.0 - dist / (height * 0.07)
                scale = 1.0 + (factor - 1.0) * dist_factor
                
                # Scale x and z coordinates (width and depth)
                mesh.vertices[i, 0] *= scale
                mesh.vertices[i, 2] *= scale
    
    def _adjust_arm_length(self, mesh: trimesh.Trimesh, factor: float) -> None:
        """
        Adjust the arm length.
        
        Args:
            mesh: Input mesh to modify in-place
            factor: Scale factor for arm length
        """
        # This is a simplified implementation
        # In a real implementation, you would use proper segmentation
        
        # Get mesh bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        height = bounds[1, 1] - bounds[0, 1]
        width = bounds[1, 0] - bounds[0, 0]
        
        # Arm regions (approximately)
        arm_y = bounds[0, 1] + height * 0.7  # Shoulder level
        
        # Right arm
        right_x = bounds[1, 0] - width * 0.1  # Right side
        # Left arm
        left_x = bounds[0, 0] + width * 0.1   # Left side
        
        # Adjust arm vertices
        for i, v in enumerate(mesh.vertices):
            x, y = v[0], v[1]
            
            # Right arm
            if x > right_x and y < arm_y:
                # Scale y relative to shoulder joint
                mesh.vertices[i, 1] = arm_y + (y - arm_y) * factor
            
            # Left arm
            if x < left_x and y < arm_y:
                # Scale y relative to shoulder joint
                mesh.vertices[i, 1] = arm_y + (y - arm_y) * factor
    
    def _adjust_leg_length(self, mesh: trimesh.Trimesh, factor: float) -> None:
        """
        Adjust the leg length.
        
        Args:
            mesh: Input mesh to modify in-place
            factor: Scale factor for leg length
        """
        # This is a simplified implementation
        # In a real implementation, you would use proper segmentation
        
        # Get mesh bounds
        bounds = np.vstack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))
        height = bounds[1, 1] - bounds[0, 1]
        
        # Leg region (approximately)
        leg_y = bounds[0, 1] + height * 0.45  # Hip level
        
        # Adjust leg vertices
        for i, v in enumerate(mesh.vertices):
            y = v[1]
            
            # Check if vertex is in leg region
            if y < leg_y:
                # Scale y relative to hip joint
                mesh.vertices[i, 1] = leg_y + (y - leg_y) * factor
    
    def _apply_simplified_pose(self, 
                            mesh: trimesh.Trimesh, 
                            pose_params: np.ndarray) -> trimesh.Trimesh:
        """
        Apply a simplified pose to a mesh.
        
        Args:
            mesh: Input mesh
            pose_params: Pose parameters
            
        Returns:
            Posed mesh
        """
        # This is a placeholder for a simplified posing system
        # In a real implementation, you would use a proper skinning algorithm
        
        # For now, just apply global rotation
        if len(pose_params) >= 3:
            global_orient = pose_params[:3]
            
            # Apply global rotation
            rotation = trimesh.transformations.euler_matrix(
                global_orient[0], global_orient[1], global_orient[2], 'sxyz')
            mesh.apply_transform(rotation)
        
        return mesh
    
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
        
        # Try to use xatlas if available
        try:
            import xatlas
            import numpy as np
            
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
        
        # Normalize height
        y_normalized = (vertices[:, 1] - bounds[0, 1]) / height
        
        # Calculate angles for cylindrical mapping
        x = vertices[:, 0]
        z = vertices[:, 2]
        
        # Calculate angle from center
        angles = np.arctan2(z, x)
        
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
    
    def save_mesh(self, mesh: trimesh.Trimesh, output_path: str, texture_path: Optional[str] = None) -> str:
        """
        Save a mesh to disk in OBJ format.
        
        Args:
            mesh: Body mesh to save
            output_path: Path to save the mesh
            texture_path: Optional path to texture image
            
        Returns:
            Path to the saved mesh
        """
        # Create output directory if needed
        ensure_directory(os.path.dirname(output_path))
        
        # Check file extension and convert if needed
        if not output_path.lower().endswith('.obj'):
            base_path = os.path.splitext(output_path)[0]
            output_path = f"{base_path}.obj"
        
        # If texture is provided, make sure the mesh has UV coordinates
        if texture_path is not None:
            if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                print("Adding UV coordinates for textured export")
                mesh = self.generate_uv_coordinates(mesh)
            
            # Apply texture to mesh
            if os.path.exists(texture_path):
                try:
                    material = trimesh.visual.material.SimpleMaterial(image=texture_path)
                    mesh.visual.material = material
                except Exception as e:
                    print(f"Error applying texture: {e}")
        
        # Save mesh
        try:
            mesh.export(output_path)
            print(f"Mesh saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving mesh: {e}")
            return ""
    
    def save_model_params(self, 
                       shape_params: np.ndarray, 
                       pose_params: np.ndarray, 
                       output_path: str) -> None:
        """
        Save model parameters to a JSON file.
        
        Args:
            shape_params: Shape parameters
            pose_params: Pose parameters
            output_path: Path to save the JSON file
        """
        # Create output directory if needed
        ensure_directory(os.path.dirname(output_path))
        
        # Create parameters dictionary
        params = {
            'shape': shape_params.tolist(),
            'pose': pose_params.tolist(),
            'gender': self.gender,
            'model_type': 'smpl' if self.use_smpl else 'simplified'
        }
        
        # Save parameters
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"Model parameters saved to: {output_path}")
    
    def load_model_params(self, input_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load model parameters from a JSON file.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            Tuple of (shape_params, pose_params)
        """
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Parameter file not found: {input_path}")
            return (self.default_shape_params.cpu().numpy(), np.zeros(72))
        
        try:
            # Load parameters
            with open(input_path, 'r') as f:
                params = json.load(f)
            
            # Extract parameters
            shape_params = np.array(params.get('shape', [0] * 10))
            pose_params = np.array(params.get('pose', [0] * 72))
            
            # Check shape parameter size
            if len(shape_params) != 10:
                print(f"Invalid shape parameters size: {len(shape_params)}, expected 10")
                shape_params = np.zeros(10)
            
            # Check pose parameter size
            if len(pose_params) != 72:
                print(f"Invalid pose parameters size: {len(pose_params)}, expected 72")
                pose_params = np.zeros(72)
            
            return (shape_params, pose_params)
            
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return (self.default_shape_params.cpu().numpy(), np.zeros(72)) 