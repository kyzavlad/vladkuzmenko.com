import os
import numpy as np
import torch
import trimesh
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any

from app.avatar_creation.face_modeling.utils import get_device, ensure_directory

class AvatarAnimator:
    """
    Class for real-time animation of 3D avatars.
    Supports animation of face and body with blendshapes and skeletal animation.
    """
    
    def __init__(self, 
                model_path: str,
                blendshapes_dir: Optional[str] = None,
                skeleton_path: Optional[str] = None,
                texture_path: Optional[str] = None,
                use_gpu: bool = True):
        """
        Initialize the avatar animator.
        
        Args:
            model_path: Path to the 3D avatar model (OBJ/FBX/GLB)
            blendshapes_dir: Directory containing blendshape meshes
            skeleton_path: Path to skeleton/rig information
            texture_path: Path to the texture file
            use_gpu: Whether to use GPU acceleration
        """
        self.device = get_device() if use_gpu else torch.device("cpu")
        self.model_path = model_path
        self.blendshapes_dir = blendshapes_dir
        self.skeleton_path = skeleton_path
        self.texture_path = texture_path
        
        # Load the avatar model
        self.avatar_model = self._load_model(model_path)
        
        # Load blendshapes if available
        self.blendshapes = {}
        if blendshapes_dir and os.path.exists(blendshapes_dir):
            self.blendshapes = self._load_blendshapes(blendshapes_dir)
        
        # Load skeleton if available
        self.skeleton = None
        if skeleton_path and os.path.exists(skeleton_path):
            self.skeleton = self._load_skeleton(skeleton_path)
        
        # Current animation state
        self.current_state = {
            'blendshape_weights': {},
            'bone_rotations': {},
            'bone_translations': {},
            'global_transform': np.eye(4)
        }
        
        # Animation history for smoothing
        self.animation_history = []
        self.max_history_length = 10  # frames
        
        # Performance metrics
        self.last_update_time = time.time()
        self.frame_times = []
        
        print(f"Avatar animator initialized with model: {model_path}")
        print(f"  - Blendshapes: {len(self.blendshapes)}")
        print(f"  - Skeleton: {'Loaded' if self.skeleton else 'None'}")
    
    def _load_model(self, model_path: str) -> Any:
        """
        Load the avatar 3D model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded 3D model
        """
        try:
            # Try loading with trimesh
            model = trimesh.load(model_path)
            print(f"Loaded model with {len(model.vertices)} vertices and {len(model.faces)} faces")
            return model
        except Exception as e:
            print(f"Error loading model with trimesh: {e}")
            
            # Try alternative loading methods based on file extension
            ext = os.path.splitext(model_path)[1].lower()
            
            if ext == '.fbx':
                # For FBX files, we would need a specialized loader
                try:
                    import pyfbx
                    # This is a placeholder, pyfbx usage would depend on the actual package
                    return pyfbx.load(model_path)
                except ImportError:
                    print("pyfbx not available for loading FBX files")
            
            elif ext == '.glb' or ext == '.gltf':
                # For glTF files
                try:
                    from pygltflib import GLTF2
                    model = GLTF2().load(model_path)
                    return model
                except ImportError:
                    print("pygltflib not available for loading glTF files")
            
            # If all else fails, return a simple trimesh cube as placeholder
            print("Using placeholder cube model")
            return trimesh.creation.box()
    
    def _load_blendshapes(self, blendshapes_dir: str) -> Dict[str, Any]:
        """
        Load blendshape meshes from directory.
        
        Args:
            blendshapes_dir: Directory containing blendshape meshes
            
        Returns:
            Dictionary of blendshapes
        """
        blendshapes = {}
        
        # Get list of files in the blendshapes directory
        try:
            files = os.listdir(blendshapes_dir)
            
            for file in files:
                if file.endswith(('.obj', '.ply')):
                    # Extract blendshape name from filename (e.g., "smile.obj" -> "smile")
                    name = os.path.splitext(file)[0]
                    
                    # Load the blendshape mesh
                    mesh_path = os.path.join(blendshapes_dir, file)
                    try:
                        mesh = trimesh.load(mesh_path)
                        blendshapes[name] = mesh
                        print(f"Loaded blendshape: {name}")
                    except Exception as e:
                        print(f"Error loading blendshape {name}: {e}")
        
        except Exception as e:
            print(f"Error loading blendshapes: {e}")
        
        return blendshapes
    
    def _load_skeleton(self, skeleton_path: str) -> Dict:
        """
        Load skeleton/rig information from file.
        
        Args:
            skeleton_path: Path to skeleton file
            
        Returns:
            Skeleton data structure
        """
        skeleton = {
            'bones': [],
            'hierarchy': {},
            'inverse_binds': []
        }
        
        try:
            # Check file extension
            ext = os.path.splitext(skeleton_path)[1].lower()
            
            if ext == '.json':
                # Load from JSON format
                with open(skeleton_path, 'r') as f:
                    skeleton_data = json.load(f)
                
                # Process the skeleton data
                if 'bones' in skeleton_data:
                    skeleton['bones'] = skeleton_data['bones']
                if 'hierarchy' in skeleton_data:
                    skeleton['hierarchy'] = skeleton_data['hierarchy']
                if 'inverse_binds' in skeleton_data:
                    skeleton['inverse_binds'] = skeleton_data['inverse_binds']
            
            elif ext == '.fbx':
                # FBX skeleton loading would require specialized handling
                print("FBX skeleton loading not fully implemented")
                # Placeholder for FBX skeleton extraction
            
            else:
                print(f"Unsupported skeleton file format: {ext}")
        
        except Exception as e:
            print(f"Error loading skeleton: {e}")
        
        return skeleton
    
    def set_blendshape_weight(self, blendshape_name: str, weight: float) -> None:
        """
        Set the weight of a specific blendshape.
        
        Args:
            blendshape_name: Name of the blendshape
            weight: Weight value (typically 0.0-1.0)
        """
        weight = max(0.0, min(1.0, weight))  # Clamp to [0, 1]
        
        if blendshape_name in self.blendshapes:
            self.current_state['blendshape_weights'][blendshape_name] = weight
        else:
            print(f"Warning: Blendshape '{blendshape_name}' not found")
    
    def set_bone_rotation(self, bone_name: str, rotation: np.ndarray) -> None:
        """
        Set the rotation of a specific bone.
        
        Args:
            bone_name: Name of the bone
            rotation: Rotation as a 3x3 matrix or quaternion
        """
        if self.skeleton is None:
            print("Warning: No skeleton loaded")
            return
            
        # Add to current state
        self.current_state['bone_rotations'][bone_name] = rotation
    
    def set_global_transform(self, transform: np.ndarray) -> None:
        """
        Set the global transform of the avatar.
        
        Args:
            transform: 4x4 transformation matrix
        """
        self.current_state['global_transform'] = transform
    
    def update(self) -> Dict:
        """
        Update the avatar based on current animation state.
        
        Returns:
            Updated avatar data
        """
        start_time = time.time()
        
        # Apply blendshape deformations
        deformed_vertices = self._apply_blendshapes(
            self.avatar_model.vertices, 
            self.current_state['blendshape_weights']
        )
        
        # Apply skeletal animation if skeleton exists
        if self.skeleton:
            deformed_vertices = self._apply_skeletal_animation(
                deformed_vertices,
                self.current_state['bone_rotations'],
                self.current_state['bone_translations']
            )
        
        # Apply global transform
        transformed_vertices = self._apply_global_transform(
            deformed_vertices,
            self.current_state['global_transform']
        )
        
        # Create updated mesh
        updated_mesh = self.avatar_model.copy()
        updated_mesh.vertices = transformed_vertices
        
        # Add entry to animation history
        self.animation_history.append({
            'timestamp': time.time(),
            'state': self.current_state.copy()
        })
        
        # Trim history if needed
        if len(self.animation_history) > self.max_history_length:
            self.animation_history.pop(0)
        
        # Calculate performance metrics
        end_time = time.time()
        update_time = end_time - self.last_update_time
        self.last_update_time = end_time
        
        self.frame_times.append(end_time - start_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        return {
            'mesh': updated_mesh,
            'update_time': update_time,
            'avg_frame_time': avg_frame_time
        }
    
    def _apply_blendshapes(self, base_vertices: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """
        Apply blendshape deformations to base vertices.
        
        Args:
            base_vertices: Base vertex positions
            weights: Dictionary of blendshape weights
            
        Returns:
            Deformed vertices
        """
        # Start with the base vertices
        result = base_vertices.copy()
        
        # Apply each active blendshape
        for name, weight in weights.items():
            if weight > 0 and name in self.blendshapes:
                blendshape = self.blendshapes[name]
                
                # Ensure the blendshape has the same number of vertices
                if len(blendshape.vertices) == len(base_vertices):
                    # Calculate the displacement
                    displacement = (blendshape.vertices - base_vertices) * weight
                    
                    # Apply the displacement
                    result += displacement
                else:
                    print(f"Warning: Blendshape '{name}' vertex count mismatch")
        
        return result
    
    def _apply_skeletal_animation(self, 
                               vertices: np.ndarray, 
                               bone_rotations: Dict[str, np.ndarray],
                               bone_translations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply skeletal animation to vertices.
        
        Args:
            vertices: Vertex positions
            bone_rotations: Dictionary of bone rotations
            bone_translations: Dictionary of bone translations
            
        Returns:
            Animated vertices
        """
        # This is a simplified implementation
        # A full implementation would use proper skinning weights
        
        # If no skeleton, return unchanged vertices
        if not self.skeleton:
            return vertices
        
        # Placeholder for a real skeletal animation system
        # In a real system, you would:
        # 1. Apply bone transformations in hierarchy order
        # 2. Compute skin deformation using vertex weights
        # 3. Handle additional deformation like dual quaternion skinning
        
        return vertices  # Placeholder: no change
    
    def _apply_global_transform(self, vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply global transformation to vertices.
        
        Args:
            vertices: Vertex positions
            transform: 4x4 transformation matrix
            
        Returns:
            Transformed vertices
        """
        # Check if transform is a 4x4 matrix
        if transform.shape != (4, 4):
            print("Warning: Global transform must be a 4x4 matrix")
            return vertices
        
        # Convert vertices to homogeneous coordinates
        homogeneous = np.ones((len(vertices), 4))
        homogeneous[:, :3] = vertices
        
        # Apply transformation
        transformed = np.dot(homogeneous, transform.T)
        
        # Convert back to 3D coordinates
        return transformed[:, :3]
    
    def smoothed_update(self, smoothing_factor: float = 0.5) -> Dict:
        """
        Update with temporal smoothing for more stable animation.
        
        Args:
            smoothing_factor: Factor for smoothing (0 = no smoothing, 1 = max smoothing)
            
        Returns:
            Updated avatar data
        """
        # If no history, do a regular update
        if not self.animation_history:
            return self.update()
        
        # Save current state
        original_state = self.current_state.copy()
        
        # Apply smoothing to the current state
        if self.animation_history:
            # Get previous state
            prev_state = self.animation_history[-1]['state']
            
            # Blend blendshape weights
            for name, weight in original_state['blendshape_weights'].items():
                prev_weight = prev_state['blendshape_weights'].get(name, 0.0)
                self.current_state['blendshape_weights'][name] = (
                    weight * (1 - smoothing_factor) + prev_weight * smoothing_factor
                )
            
            # Blend bone rotations (simplified, real impl would use quaternions)
            for name, rot in original_state['bone_rotations'].items():
                if name in prev_state['bone_rotations']:
                    prev_rot = prev_state['bone_rotations'][name]
                    self.current_state['bone_rotations'][name] = (
                        rot * (1 - smoothing_factor) + prev_rot * smoothing_factor
                    )
        
        # Do the update with smoothed state
        result = self.update()
        
        # Restore original state
        self.current_state = original_state
        
        return result
    
    def get_available_blendshapes(self) -> List[str]:
        """
        Get list of available blendshapes.
        
        Returns:
            List of blendshape names
        """
        return list(self.blendshapes.keys())
    
    def get_available_bones(self) -> List[str]:
        """
        Get list of available bones.
        
        Returns:
            List of bone names
        """
        if self.skeleton:
            return [bone.get('name', f"bone_{i}") for i, bone in enumerate(self.skeleton.get('bones', []))]
        return []
    
    def set_expression(self, expression_name: str, intensity: float = 1.0) -> None:
        """
        Set a predefined facial expression.
        
        Args:
            expression_name: Name of the expression (e.g., "smile", "surprise")
            intensity: Intensity of the expression (0.0-1.0)
        """
        # Reset facial blendshapes
        for name in self.get_available_blendshapes():
            if name.startswith("face_"):
                self.set_blendshape_weight(name, 0.0)
        
        # Apply the specified expression
        if expression_name == "smile":
            self.set_blendshape_weight("face_smile", intensity)
            self.set_blendshape_weight("face_cheek_squint", intensity * 0.5)
        
        elif expression_name == "surprise":
            self.set_blendshape_weight("face_brow_raise", intensity)
            self.set_blendshape_weight("face_mouth_open", intensity)
        
        elif expression_name == "angry":
            self.set_blendshape_weight("face_brow_lower", intensity)
            self.set_blendshape_weight("face_nose_wrinkle", intensity * 0.7)
        
        elif expression_name == "sad":
            self.set_blendshape_weight("face_frown", intensity)
            self.set_blendshape_weight("face_mouth_down", intensity * 0.5)
        
        else:
            print(f"Unknown expression: {expression_name}")
    
    def set_pose(self, pose_name: str) -> None:
        """
        Set a predefined body pose.
        
        Args:
            pose_name: Name of the pose (e.g., "t_pose", "a_pose")
        """
        if self.skeleton is None:
            print("Warning: No skeleton loaded, cannot set pose")
            return
            
        # Set pose based on name
        if pose_name == "t_pose":
            # Reset all rotations to T-pose
            for bone in self.get_available_bones():
                self.set_bone_rotation(bone, np.eye(3))  # Identity rotation
        
        elif pose_name == "a_pose":
            # A-pose: arms slightly down
            for bone in self.get_available_bones():
                if "arm" in bone.lower():
                    # Simple rotation matrix for arms down about 15 degrees
                    angle = np.radians(15)
                    rotation = np.array([
                        [1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]
                    ])
                    self.set_bone_rotation(bone, rotation)
                else:
                    self.set_bone_rotation(bone, np.eye(3))
        
        else:
            print(f"Unknown pose: {pose_name}")
    
    def save_current_frame(self, output_path: str) -> str:
        """
        Save the current animation frame as a 3D model.
        
        Args:
            output_path: Path to save the model
            
        Returns:
            Path to the saved file
        """
        # Update the mesh
        result = self.update()
        mesh = result['mesh']
        
        # Create output directory if it doesn't exist
        ensure_directory(os.path.dirname(output_path))
        
        # Save the mesh
        try:
            mesh.export(output_path)
            print(f"Saved frame to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving frame: {e}")
            return ""
    
    def create_animation_sequence(self, 
                               animation_data: List[Dict], 
                               output_dir: str,
                               fps: int = 30) -> List[str]:
        """
        Create an animation sequence by interpolating between keyframes.
        
        Args:
            animation_data: List of keyframes with blendshape and bone data
            output_dir: Directory to save animation frames
            fps: Frames per second
            
        Returns:
            List of paths to saved frames
        """
        # Create output directory
        ensure_directory(output_dir)
        
        frame_paths = []
        
        # If no keyframes, return empty list
        if not animation_data:
            return frame_paths
        
        # Calculate total animation duration
        total_time = animation_data[-1].get('time', len(animation_data) - 1) - animation_data[0].get('time', 0)
        
        # Calculate number of frames
        num_frames = int(total_time * fps) + 1
        
        # Process each frame
        for frame in range(num_frames):
            # Calculate time for this frame
            time_point = animation_data[0].get('time', 0) + frame / fps
            
            # Find keyframes before and after this time
            next_idx = 0
            while next_idx < len(animation_data) and animation_data[next_idx].get('time', next_idx) < time_point:
                next_idx += 1
            
            prev_idx = max(0, next_idx - 1)
            
            # Handle boundary cases
            if next_idx >= len(animation_data):
                next_idx = len(animation_data) - 1
            
            # Get keyframes
            prev_keyframe = animation_data[prev_idx]
            next_keyframe = animation_data[next_idx]
            
            # Calculate interpolation factor
            prev_time = prev_keyframe.get('time', prev_idx)
            next_time = next_keyframe.get('time', next_idx)
            
            # Avoid division by zero
            if next_time == prev_time:
                factor = 0
            else:
                factor = (time_point - prev_time) / (next_time - prev_time)
            
            # Interpolate blendshape weights
            for name in set(list(prev_keyframe.get('blendshapes', {}).keys()) + 
                           list(next_keyframe.get('blendshapes', {}).keys())):
                prev_weight = prev_keyframe.get('blendshapes', {}).get(name, 0.0)
                next_weight = next_keyframe.get('blendshapes', {}).get(name, 0.0)
                
                weight = prev_weight * (1 - factor) + next_weight * factor
                self.set_blendshape_weight(name, weight)
            
            # Interpolate bone rotations (simplified)
            if self.skeleton:
                for name in set(list(prev_keyframe.get('bones', {}).keys()) + 
                               list(next_keyframe.get('bones', {}).keys())):
                    prev_rot = prev_keyframe.get('bones', {}).get(name, np.eye(3))
                    next_rot = next_keyframe.get('bones', {}).get(name, np.eye(3))
                    
                    # Linear interpolation (better would be quaternion slerp)
                    rot = prev_rot * (1 - factor) + next_rot * factor
                    self.set_bone_rotation(name, rot)
            
            # Update the mesh
            self.update()
            
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{frame:04d}.obj")
            saved_path = self.save_current_frame(frame_path)
            
            if saved_path:
                frame_paths.append(saved_path)
        
        return frame_paths
    
    def reset(self) -> None:
        """
        Reset the animator to default state.
        """
        # Clear blendshape weights
        self.current_state['blendshape_weights'] = {}
        
        # Reset bone rotations
        self.current_state['bone_rotations'] = {}
        self.current_state['bone_translations'] = {}
        
        # Reset global transform
        self.current_state['global_transform'] = np.eye(4)
        
        # Clear animation history
        self.animation_history = []
        
        print("Animator reset to default state") 