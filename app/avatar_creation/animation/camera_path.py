#!/usr/bin/env python3
"""
Camera Path Editor

This module provides tools for creating and editing camera paths for avatar
animations, allowing for smooth camera movements between keyframes.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


class CameraKeyframe:
    """Camera keyframe with position, target, and other properties."""
    
    def __init__(self,
                time: float,
                position: List[float],
                target: List[float],
                up: Optional[List[float]] = None,
                fov: float = 45.0,
                roll: float = 0.0,
                transition_smoothness: float = 0.5,
                ease_type: str = "ease_in_out"):
        """
        Initialize a camera keyframe.
        
        Args:
            time: Time in seconds
            position: Camera position [x, y, z]
            target: Camera target/look-at point [x, y, z]
            up: Up vector [x, y, z], default is [0, 1, 0]
            fov: Field of view in degrees
            roll: Camera roll in degrees
            transition_smoothness: Smoothness of transition (0-1)
            ease_type: Easing function type
        """
        self.time = time
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up or [0, 1, 0], dtype=np.float32)
        self.fov = fov
        self.roll = roll
        self.transition_smoothness = max(0.0, min(1.0, transition_smoothness))
        self.ease_type = ease_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert keyframe to dictionary."""
        return {
            "time": self.time,
            "position": self.position.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist(),
            "fov": self.fov,
            "roll": self.roll,
            "transition_smoothness": self.transition_smoothness,
            "ease_type": self.ease_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraKeyframe':
        """Create keyframe from dictionary."""
        return cls(
            time=data["time"],
            position=data["position"],
            target=data["target"],
            up=data.get("up", [0, 1, 0]),
            fov=data.get("fov", 45.0),
            roll=data.get("roll", 0.0),
            transition_smoothness=data.get("transition_smoothness", 0.5),
            ease_type=data.get("ease_type", "ease_in_out")
        )


class CameraPath:
    """
    Camera path with keyframes and interpolation methods.
    """
    
    def __init__(self, name: str = "default_path"):
        """
        Initialize a camera path.
        
        Args:
            name: Name of the camera path
        """
        self.name = name
        self.keyframes: List[CameraKeyframe] = []
        self.loop = False
        self.duration = 0.0
    
    def add_keyframe(self, keyframe: CameraKeyframe) -> None:
        """
        Add a keyframe to the path.
        
        Args:
            keyframe: Camera keyframe to add
        """
        # Insert in sorted order by time
        inserted = False
        for i, kf in enumerate(self.keyframes):
            if keyframe.time < kf.time:
                self.keyframes.insert(i, keyframe)
                inserted = True
                break
        
        if not inserted:
            self.keyframes.append(keyframe)
        
        # Update duration
        if self.keyframes:
            self.duration = max(self.duration, keyframe.time)
    
    def remove_keyframe(self, time: float, tolerance: float = 0.001) -> bool:
        """
        Remove a keyframe at the specified time.
        
        Args:
            time: Time in seconds
            tolerance: Time tolerance for matching
            
        Returns:
            True if keyframe was removed, False otherwise
        """
        for i, kf in enumerate(self.keyframes):
            if abs(kf.time - time) < tolerance:
                self.keyframes.pop(i)
                
                # Update duration if needed
                if self.keyframes:
                    self.duration = max(kf.time for kf in self.keyframes)
                else:
                    self.duration = 0.0
                    
                return True
        
        return False
    
    def get_keyframe_at_time(self, time: float, tolerance: float = 0.001) -> Optional[CameraKeyframe]:
        """
        Get keyframe at the specified time.
        
        Args:
            time: Time in seconds
            tolerance: Time tolerance for matching
            
        Returns:
            Keyframe if found, None otherwise
        """
        for kf in self.keyframes:
            if abs(kf.time - time) < tolerance:
                return kf
        
        return None
    
    def get_camera_at_time(self, time: float) -> Dict[str, Any]:
        """
        Get interpolated camera properties at the specified time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Dictionary with camera properties
        """
        if not self.keyframes:
            # Default camera if no keyframes
            return {
                "position": [0.0, 0.0, 1.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 45.0,
                "roll": 0.0
            }
        
        # Handle looping
        if self.loop and time > self.duration and self.duration > 0:
            time = time % self.duration
        
        # Find surrounding keyframes
        prev_kf = None
        next_kf = None
        
        for i, kf in enumerate(self.keyframes):
            if abs(kf.time - time) < 0.001:
                # Exact match
                return {
                    "position": kf.position.tolist(),
                    "target": kf.target.tolist(),
                    "up": kf.up.tolist(),
                    "fov": kf.fov,
                    "roll": kf.roll
                }
            
            if kf.time < time:
                prev_kf = kf
            elif kf.time > time and next_kf is None:
                next_kf = kf
        
        # Handle edge cases
        if prev_kf is None:
            # Before first keyframe
            first_kf = self.keyframes[0]
            return {
                "position": first_kf.position.tolist(),
                "target": first_kf.target.tolist(),
                "up": first_kf.up.tolist(),
                "fov": first_kf.fov,
                "roll": first_kf.roll
            }
        
        if next_kf is None:
            # After last keyframe
            last_kf = self.keyframes[-1]
            return {
                "position": last_kf.position.tolist(),
                "target": last_kf.target.tolist(),
                "up": last_kf.up.tolist(),
                "fov": last_kf.fov,
                "roll": last_kf.roll
            }
        
        # Interpolate between keyframes
        t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)
        
        # Apply easing function
        t = self._apply_easing(t, prev_kf.ease_type, prev_kf.transition_smoothness)
        
        # Interpolate camera properties
        position = self._interpolate_vectors(prev_kf.position, next_kf.position, t)
        target = self._interpolate_vectors(prev_kf.target, next_kf.target, t)
        up = self._interpolate_vectors(prev_kf.up, next_kf.up, t)
        fov = prev_kf.fov + t * (next_kf.fov - prev_kf.fov)
        
        # Special handling for roll to avoid jumps
        roll_diff = next_kf.roll - prev_kf.roll
        if abs(roll_diff) > 180:
            if roll_diff > 0:
                roll_diff -= 360
            else:
                roll_diff += 360
        roll = prev_kf.roll + t * roll_diff
        
        return {
            "position": position.tolist(),
            "target": target.tolist(),
            "up": up.tolist(),
            "fov": fov,
            "roll": roll
        }
    
    def _interpolate_vectors(self, v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation between two vectors."""
        return v1 + t * (v2 - v1)
    
    def _apply_easing(self, t: float, ease_type: str, smoothness: float) -> float:
        """
        Apply easing function to interpolation factor.
        
        Args:
            t: Interpolation factor (0-1)
            ease_type: Type of easing function
            smoothness: Smoothness factor (0-1)
            
        Returns:
            Eased interpolation factor
        """
        if ease_type == "linear":
            return t
        
        elif ease_type == "ease_in":
            return t ** (1 + smoothness * 2)
        
        elif ease_type == "ease_out":
            return 1 - (1 - t) ** (1 + smoothness * 2)
        
        elif ease_type == "ease_in_out":
            if t < 0.5:
                return 0.5 * (2 * t) ** (1 + smoothness * 2)
            else:
                return 1 - 0.5 * (2 * (1 - t)) ** (1 + smoothness * 2)
        
        elif ease_type == "bounce":
            # Simple bounce easing
            k = 1 - smoothness * 0.5
            return (1 - np.cos(t * np.pi * (1 + 2 * smoothness))) * k
        
        # Default to linear
        return t
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export camera path to dictionary.
        
        Returns:
            Dictionary representation of camera path
        """
        return {
            "name": self.name,
            "loop": self.loop,
            "duration": self.duration,
            "keyframes": [kf.to_dict() for kf in self.keyframes]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraPath':
        """
        Create camera path from dictionary.
        
        Args:
            data: Dictionary representation of camera path
            
        Returns:
            CameraPath instance
        """
        path = cls(data.get("name", "imported_path"))
        path.loop = data.get("loop", False)
        
        for kf_data in data.get("keyframes", []):
            keyframe = CameraKeyframe.from_dict(kf_data)
            path.add_keyframe(keyframe)
        
        return path
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save camera path to JSON file.
        
        Args:
            file_path: Path to save the file
        """
        with open(file_path, 'w') as f:
            json.dump(self.export_to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'CameraPath':
        """
        Load camera path from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            CameraPath instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def create_orbit_path(self, 
                        center: List[float], 
                        radius: float, 
                        height: float, 
                        duration: float,
                        num_keyframes: int = 24,
                        fov: float = 45.0) -> None:
        """
        Create an orbit camera path around a center point.
        
        Args:
            center: Center point to orbit around [x, y, z]
            radius: Orbit radius
            height: Height above center
            duration: Total duration in seconds
            num_keyframes: Number of keyframes in the orbit
            fov: Field of view in degrees
        """
        self.keyframes = []
        self.duration = duration
        self.loop = True
        
        center_array = np.array(center, dtype=np.float32)
        
        for i in range(num_keyframes):
            angle = 2 * np.pi * i / num_keyframes
            time = duration * i / num_keyframes
            
            # Calculate position on orbit
            x = center[0] + radius * np.cos(angle)
            z = center[2] + radius * np.sin(angle)
            y = center[1] + height
            
            position = [x, y, z]
            
            # Add keyframe
            kf = CameraKeyframe(
                time=time,
                position=position,
                target=center,
                fov=fov,
                transition_smoothness=0.5,
                ease_type="linear"  # Linear for smooth orbits
            )
            
            self.add_keyframe(kf)
    
    def create_flyby_path(self,
                        start_position: List[float],
                        end_position: List[float],
                        target: List[float],
                        duration: float,
                        num_keyframes: int = 5,
                        curve_height: float = 0.0,
                        ease_type: str = "ease_in_out") -> None:
        """
        Create a flyby camera path from start to end position.
        
        Args:
            start_position: Starting camera position [x, y, z]
            end_position: Ending camera position [x, y, z]
            target: Camera target/look-at point [x, y, z]
            duration: Total duration in seconds
            num_keyframes: Number of keyframes
            curve_height: Height of curve above straight line
            ease_type: Easing function type
        """
        self.keyframes = []
        self.duration = duration
        self.loop = False
        
        start_array = np.array(start_position, dtype=np.float32)
        end_array = np.array(end_position, dtype=np.float32)
        
        # Create path with curved trajectory if curve_height > 0
        for i in range(num_keyframes):
            t = i / (num_keyframes - 1)
            time = duration * t
            
            # Linear position
            position = start_array + t * (end_array - start_array)
            
            # Add curve if requested
            if curve_height > 0 and i > 0 and i < num_keyframes - 1:
                # Parabolic height adjustment (max at t=0.5)
                height_factor = 4 * t * (1 - t)
                position[1] += curve_height * height_factor
            
            # FOV animation (wider in the middle for dramatic effect)
            fov = 45.0
            if num_keyframes > 2 and i > 0 and i < num_keyframes - 1:
                fov_factor = 4 * t * (1 - t)  # Same as height factor
                fov = 45.0 + 15.0 * fov_factor
            
            # Add keyframe
            kf = CameraKeyframe(
                time=time,
                position=position.tolist(),
                target=target,
                fov=fov,
                transition_smoothness=0.5,
                ease_type=ease_type
            )
            
            self.add_keyframe(kf)


class CameraPathLibrary:
    """
    Library for managing multiple camera paths.
    """
    
    def __init__(self):
        """Initialize the camera path library."""
        self.paths: Dict[str, CameraPath] = {}
        self.presets: Dict[str, Dict[str, Any]] = {
            "front": {
                "position": [0.0, 0.0, 1.0],
                "target": [0.0, 0.0, 0.0]
            },
            "side": {
                "position": [1.0, 0.0, 0.0],
                "target": [0.0, 0.0, 0.0]
            },
            "three-quarter": {
                "position": [0.7, 0.0, 0.7],
                "target": [0.0, 0.0, 0.0]
            },
            "top": {
                "position": [0.0, 1.0, 0.0],
                "target": [0.0, 0.0, 0.0]
            },
            "dramatic-low": {
                "position": [0.5, -0.3, 0.8],
                "target": [0.0, 0.2, 0.0]
            }
        }
    
    def add_path(self, path: CameraPath) -> None:
        """Add a camera path to the library."""
        self.paths[path.name] = path
    
    def remove_path(self, name: str) -> bool:
        """Remove a camera path from the library."""
        if name in self.paths:
            del self.paths[name]
            return True
        return False
    
    def get_path(self, name: str) -> Optional[CameraPath]:
        """Get a camera path by name."""
        return self.paths.get(name)
    
    def add_preset(self, name: str, position: List[float], target: List[float]) -> None:
        """Add a camera preset to the library."""
        self.presets[name] = {
            "position": position,
            "target": target
        }
    
    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a camera preset by name."""
        return self.presets.get(name)
    
    def save_library(self, file_path: str) -> None:
        """Save the entire library to a JSON file."""
        data = {
            "paths": {name: path.export_to_dict() for name, path in self.paths.items()},
            "presets": self.presets
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_library(cls, file_path: str) -> 'CameraPathLibrary':
        """Load a library from a JSON file."""
        library = cls()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Load paths
        for name, path_data in data.get("paths", {}).items():
            library.paths[name] = CameraPath.from_dict(path_data)
        
        # Load presets
        library.presets.update(data.get("presets", {}))
        
        return library
    
    def create_common_paths(self, target: List[float] = [0, 0, 0]) -> None:
        """
        Create a set of common camera paths.
        
        Args:
            target: Target to look at
        """
        # Orbit path
        orbit_path = CameraPath("orbit")
        orbit_path.create_orbit_path(
            center=target,
            radius=1.0,
            height=0.2,
            duration=10.0,
            num_keyframes=24
        )
        self.add_path(orbit_path)
        
        # Flyby path
        flyby_path = CameraPath("flyby")
        flyby_path.create_flyby_path(
            start_position=[-1.0, 0.3, 1.0],
            end_position=[1.0, 0.3, 1.0],
            target=target,
            duration=5.0,
            curve_height=0.2
        )
        self.add_path(flyby_path)
        
        # Dramatic reveal path
        reveal_path = CameraPath("dramatic_reveal")
        
        # Start far away, looking down
        kf1 = CameraKeyframe(
            time=0.0,
            position=[0.0, 1.5, 3.0],
            target=target,
            fov=35.0,
            ease_type="ease_in_out"
        )
        
        # Move closer, eye level
        kf2 = CameraKeyframe(
            time=2.0,
            position=[0.0, 0.0, 1.2],
            target=target,
            fov=45.0,
            ease_type="ease_in_out"
        )
        
        # Slight angle change for visual interest
        kf3 = CameraKeyframe(
            time=3.5,
            position=[0.5, 0.1, 1.0],
            target=target,
            fov=50.0,
            ease_type="ease_out"
        )
        
        reveal_path.add_keyframe(kf1)
        reveal_path.add_keyframe(kf2)
        reveal_path.add_keyframe(kf3)
        
        self.add_path(reveal_path)


def main():
    """Example usage of camera path tools."""
    # Create a sample camera path
    path = CameraPath("example_path")
    
    # Add keyframes
    kf1 = CameraKeyframe(
        time=0.0,
        position=[0.0, 0.0, 1.0],
        target=[0.0, 0.0, 0.0],
        fov=45.0
    )
    
    kf2 = CameraKeyframe(
        time=2.0,
        position=[1.0, 0.5, 0.5],
        target=[0.0, 0.0, 0.0],
        fov=50.0,
        ease_type="ease_in_out"
    )
    
    kf3 = CameraKeyframe(
        time=5.0,
        position=[0.0, 1.0, 0.0],
        target=[0.0, 0.0, 0.0],
        fov=35.0,
        ease_type="ease_out"
    )
    
    path.add_keyframe(kf1)
    path.add_keyframe(kf2)
    path.add_keyframe(kf3)
    
    # Sample the path at various times
    for t in [0.0, 1.0, 2.0, 3.5, 5.0]:
        camera = path.get_camera_at_time(t)
        print(f"Time {t}s: Position {camera['position']}, FOV {camera['fov']:.1f}°")
    
    # Create a camera path library
    library = CameraPathLibrary()
    library.add_path(path)
    
    # Create some common paths
    library.create_common_paths()
    
    # Add a custom preset
    library.add_preset("low_angle", [0.5, -0.3, 1.0], [0.0, 0.0, 0.0])
    
    print("\nAvailable paths:")
    for path_name in library.paths:
        print(f" - {path_name}")
    
    print("\nAvailable presets:")
    for preset_name in library.presets:
        print(f" - {preset_name}")


if __name__ == "__main__":
    main() 