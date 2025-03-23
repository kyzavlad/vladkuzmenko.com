#!/usr/bin/env python3
"""
Gesture Builder Utility

This module provides tools for creating, editing, and exporting custom gestures
for the avatar animation system's gesture library.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class GestureBuilder:
    """
    Utility for building and editing custom gestures for the avatar animation system.
    """
    
    def __init__(self):
        """Initialize the gesture builder."""
        self.current_gesture = None
        self.gesture_name = ""
        
        # Default gesture templates
        self.templates = {
            "wave": {
                "type": "hand",
                "description": "Wave hand greeting",
                "duration": 2.0,
                "joints": ["shoulder_r", "elbow_r", "wrist_r"],
                "keyframes": [
                    {"time": 0.0, "pose": {"shoulder_r": [0, 0, 0], "elbow_r": [0, 0, 0], "wrist_r": [0, 0, 0]}},
                    {"time": 0.5, "pose": {"shoulder_r": [0, 30, 0], "elbow_r": [0, 0, 80], "wrist_r": [0, 0, 20]}},
                    {"time": 1.0, "pose": {"shoulder_r": [0, 30, 0], "elbow_r": [0, 0, 80], "wrist_r": [0, 0, -20]}},
                    {"time": 1.5, "pose": {"shoulder_r": [0, 30, 0], "elbow_r": [0, 0, 80], "wrist_r": [0, 0, 20]}},
                    {"time": 2.0, "pose": {"shoulder_r": [0, 0, 0], "elbow_r": [0, 0, 0], "wrist_r": [0, 0, 0]}}
                ],
                "context_triggers": ["greeting", "goodbye"]
            },
            "point": {
                "type": "hand",
                "description": "Pointing gesture",
                "duration": 1.0,
                "joints": ["shoulder_r", "elbow_r", "wrist_r", "index_finger_r"],
                "keyframes": [
                    {"time": 0.0, "pose": {"shoulder_r": [0, 0, 0], "elbow_r": [0, 0, 0], "wrist_r": [0, 0, 0], "index_finger_r": [0, 0, 0]}},
                    {"time": 0.5, "pose": {"shoulder_r": [0, 45, 0], "elbow_r": [0, 0, 40], "wrist_r": [0, 0, 0], "index_finger_r": [0, 0, 20]}},
                    {"time": 1.0, "pose": {"shoulder_r": [0, 0, 0], "elbow_r": [0, 0, 0], "wrist_r": [0, 0, 0], "index_finger_r": [0, 0, 0]}}
                ],
                "context_triggers": ["indicating", "direction", "reference"]
            },
            "nod": {
                "type": "head",
                "description": "Head nodding (yes)",
                "duration": 1.5,
                "joints": ["head"],
                "keyframes": [
                    {"time": 0.0, "pose": {"head": [0, 0, 0]}},
                    {"time": 0.3, "pose": {"head": [20, 0, 0]}},
                    {"time": 0.6, "pose": {"head": [0, 0, 0]}},
                    {"time": 0.9, "pose": {"head": [10, 0, 0]}},
                    {"time": 1.2, "pose": {"head": [0, 0, 0]}}
                ],
                "context_triggers": ["agreement", "acknowledgment", "approval"]
            }
        }
        
        # Common gesture joints
        self.common_joints = {
            "head": "Head rotation",
            "neck": "Neck rotation",
            "shoulder_l": "Left shoulder rotation",
            "shoulder_r": "Right shoulder rotation",
            "elbow_l": "Left elbow rotation",
            "elbow_r": "Right elbow rotation",
            "wrist_l": "Left wrist rotation",
            "wrist_r": "Right wrist rotation",
            "hip_l": "Left hip rotation",
            "hip_r": "Right hip rotation",
            "knee_l": "Left knee rotation",
            "knee_r": "Right knee rotation",
            "ankle_l": "Left ankle rotation",
            "ankle_r": "Right ankle rotation",
            "spine_lower": "Lower spine rotation",
            "spine_middle": "Middle spine rotation",
            "spine_upper": "Upper spine rotation",
            "index_finger_l": "Left index finger",
            "index_finger_r": "Right index finger",
            "thumb_l": "Left thumb",
            "thumb_r": "Right thumb"
        }
    
    def create_new_gesture(self, name: str, gesture_type: str, description: str, 
                         duration: float = 1.0) -> Dict:
        """
        Create a new empty gesture with the given parameters.
        
        Args:
            name: Name of the gesture
            gesture_type: Type of gesture (e.g., "hand", "head", "full_body")
            description: Description of the gesture
            duration: Duration in seconds
            
        Returns:
            The created gesture data dictionary
        """
        self.gesture_name = name
        
        self.current_gesture = {
            "type": gesture_type,
            "description": description,
            "duration": duration,
            "joints": [],
            "keyframes": [
                {"time": 0.0, "pose": {}},  # Start keyframe (neutral pose)
                {"time": duration, "pose": {}}  # End keyframe (return to neutral)
            ],
            "context_triggers": []
        }
        
        return self.current_gesture
    
    def use_template(self, template_name: str, new_name: Optional[str] = None) -> Optional[Dict]:
        """
        Use a predefined gesture template as starting point.
        
        Args:
            template_name: Name of the template to use
            new_name: Optional new name for the gesture
            
        Returns:
            The gesture data dictionary or None if template not found
        """
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        self.current_gesture = {key: value.copy() if isinstance(value, list) else value 
                             for key, value in template.items()}
        
        # Deep copy keyframes
        self.current_gesture["keyframes"] = [
            {
                "time": kf["time"],
                "pose": kf["pose"].copy()
            }
            for kf in template["keyframes"]
        ]
        
        self.gesture_name = new_name or template_name
        
        return self.current_gesture
    
    def add_joint(self, joint_name: str) -> None:
        """
        Add a joint to the current gesture.
        
        Args:
            joint_name: Name of the joint to add
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        if joint_name not in self.current_gesture["joints"]:
            self.current_gesture["joints"].append(joint_name)
            
            # Initialize the joint in all keyframes with zero rotation
            for keyframe in self.current_gesture["keyframes"]:
                if joint_name not in keyframe["pose"]:
                    keyframe["pose"][joint_name] = [0, 0, 0]
    
    def remove_joint(self, joint_name: str) -> None:
        """
        Remove a joint from the current gesture.
        
        Args:
            joint_name: Name of the joint to remove
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        if joint_name in self.current_gesture["joints"]:
            self.current_gesture["joints"].remove(joint_name)
            
            # Remove the joint from all keyframes
            for keyframe in self.current_gesture["keyframes"]:
                if joint_name in keyframe["pose"]:
                    del keyframe["pose"][joint_name]
    
    def add_keyframe(self, time: float) -> Dict:
        """
        Add a new keyframe at the specified time.
        
        Args:
            time: Time in seconds
            
        Returns:
            The created keyframe dictionary
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        # Check if a keyframe already exists at this time
        for kf in self.current_gesture["keyframes"]:
            if abs(kf["time"] - time) < 0.001:
                return kf
        
        # Create new keyframe with neutral pose for all joints
        new_keyframe = {
            "time": time,
            "pose": {joint: [0, 0, 0] for joint in self.current_gesture["joints"]}
        }
        
        # Insert in the correct position (sorted by time)
        inserted = False
        for i, kf in enumerate(self.current_gesture["keyframes"]):
            if time < kf["time"]:
                self.current_gesture["keyframes"].insert(i, new_keyframe)
                inserted = True
                break
        
        if not inserted:
            self.current_gesture["keyframes"].append(new_keyframe)
        
        return new_keyframe
    
    def remove_keyframe(self, time: float) -> bool:
        """
        Remove the keyframe at the specified time.
        
        Args:
            time: Time in seconds
            
        Returns:
            True if a keyframe was removed, False otherwise
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        # Don't allow removing the first or last keyframe
        if len(self.current_gesture["keyframes"]) <= 2:
            return False
        
        for i, kf in enumerate(self.current_gesture["keyframes"]):
            if abs(kf["time"] - time) < 0.001:
                # Don't remove first or last keyframe
                if i == 0 or i == len(self.current_gesture["keyframes"]) - 1:
                    return False
                
                self.current_gesture["keyframes"].pop(i)
                return True
        
        return False
    
    def set_joint_rotation(self, time: float, joint: str, rotation: List[float]) -> bool:
        """
        Set the rotation for a joint at a specific keyframe.
        
        Args:
            time: Keyframe time in seconds
            joint: Joint name
            rotation: Rotation values [x, y, z] in degrees
            
        Returns:
            True if successful, False otherwise
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        if joint not in self.current_gesture["joints"]:
            return False
        
        # Find the keyframe
        for kf in self.current_gesture["keyframes"]:
            if abs(kf["time"] - time) < 0.001:
                kf["pose"][joint] = rotation.copy()
                return True
        
        return False
    
    def add_context_trigger(self, trigger: str) -> None:
        """
        Add a context trigger to the current gesture.
        
        Args:
            trigger: Context trigger word/phrase
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        if trigger not in self.current_gesture["context_triggers"]:
            self.current_gesture["context_triggers"].append(trigger)
    
    def remove_context_trigger(self, trigger: str) -> bool:
        """
        Remove a context trigger from the current gesture.
        
        Args:
            trigger: Context trigger to remove
            
        Returns:
            True if removed, False if not found
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        if trigger in self.current_gesture["context_triggers"]:
            self.current_gesture["context_triggers"].remove(trigger)
            return True
        
        return False
    
    def set_duration(self, duration: float) -> None:
        """
        Set the duration of the gesture and adjust keyframe times accordingly.
        
        Args:
            duration: New duration in seconds
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        if duration <= 0:
            raise ValueError("Duration must be positive")
        
        old_duration = self.current_gesture["duration"]
        scale_factor = duration / old_duration
        
        # Scale all keyframe times
        for kf in self.current_gesture["keyframes"]:
            kf["time"] *= scale_factor
        
        # Set the last keyframe to exactly the new duration
        self.current_gesture["keyframes"][-1]["time"] = duration
        
        # Update duration
        self.current_gesture["duration"] = duration
    
    def mirror_gesture(self, new_name: Optional[str] = None) -> Dict:
        """
        Create a mirrored version of the current gesture (left to right, right to left).
        
        Args:
            new_name: Optional name for the mirrored gesture
            
        Returns:
            The mirrored gesture data dictionary
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        mirrored = {key: value.copy() if isinstance(value, list) else value 
                  for key, value in self.current_gesture.items()}
        
        # Mirror the joints (left to right, right to left)
        mirrored["joints"] = []
        for joint in self.current_gesture["joints"]:
            if joint.endswith("_l"):
                mirrored_joint = joint[:-2] + "_r"
            elif joint.endswith("_r"):
                mirrored_joint = joint[:-2] + "_l"
            else:
                mirrored_joint = joint
            
            mirrored["joints"].append(mirrored_joint)
        
        # Mirror the keyframes
        mirrored["keyframes"] = []
        for kf in self.current_gesture["keyframes"]:
            mirrored_kf = {"time": kf["time"], "pose": {}}
            
            for joint, rotation in kf["pose"].items():
                if joint.endswith("_l"):
                    mirrored_joint = joint[:-2] + "_r"
                elif joint.endswith("_r"):
                    mirrored_joint = joint[:-2] + "_l"
                else:
                    mirrored_joint = joint
                
                # Mirror the rotation values
                # For many joints, this means negating the Y and Z rotations
                mirrored_rotation = rotation.copy()
                if len(rotation) >= 3:
                    mirrored_rotation[1] = -rotation[1]  # Y-axis
                    mirrored_rotation[2] = -rotation[2]  # Z-axis
                
                mirrored_kf["pose"][mirrored_joint] = mirrored_rotation
            
            mirrored["keyframes"].append(mirrored_kf)
        
        # Set a new name if provided
        if new_name:
            mirrored_name = new_name
        else:
            # Generate a name based on the original
            if self.gesture_name.startswith("left_"):
                mirrored_name = "right_" + self.gesture_name[5:]
            elif self.gesture_name.startswith("right_"):
                mirrored_name = "left_" + self.gesture_name[6:]
            else:
                mirrored_name = "mirrored_" + self.gesture_name
        
        return mirrored
    
    def export_gesture(self, output_path: Optional[str] = None) -> Dict:
        """
        Export the current gesture to a JSON file or return as a dictionary.
        
        Args:
            output_path: Optional path to save the gesture to a JSON file
            
        Returns:
            The gesture data dictionary
        """
        if self.current_gesture is None:
            raise ValueError("No current gesture. Create a new gesture first.")
        
        gesture_data = {self.gesture_name: self.current_gesture}
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(gesture_data, f, indent=2)
            print(f"Gesture exported to {output_path}")
        
        return gesture_data
    
    def import_gesture(self, file_path: str) -> Dict:
        """
        Import a gesture from a JSON file.
        
        Args:
            file_path: Path to the gesture JSON file
            
        Returns:
            Dictionary of imported gestures
        """
        with open(file_path, 'r') as f:
            gesture_data = json.load(f)
        
        # If there's only one gesture, set it as the current gesture
        if len(gesture_data) == 1:
            self.gesture_name = list(gesture_data.keys())[0]
            self.current_gesture = gesture_data[self.gesture_name]
        
        return gesture_data
    
    def get_available_templates(self) -> List[str]:
        """
        Get a list of available gesture templates.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def get_common_joints(self) -> Dict[str, str]:
        """
        Get a dictionary of common joint names and descriptions.
        
        Returns:
            Dictionary of joint names and descriptions
        """
        return self.common_joints


def main():
    """Example usage of the GestureBuilder."""
    builder = GestureBuilder()
    
    # Create a new gesture
    builder.create_new_gesture(
        name="thinking_pose",
        gesture_type="hand",
        description="Thinking pose with hand on chin",
        duration=3.0
    )
    
    # Add joints
    for joint in ["shoulder_r", "elbow_r", "wrist_r", "head"]:
        builder.add_joint(joint)
    
    # Add keyframes
    builder.add_keyframe(1.0)
    builder.add_keyframe(2.0)
    
    # Set joint rotations
    builder.set_joint_rotation(1.0, "shoulder_r", [0, 15, 0])
    builder.set_joint_rotation(1.0, "elbow_r", [0, 0, 90])
    builder.set_joint_rotation(1.0, "wrist_r", [0, 0, 30])
    builder.set_joint_rotation(1.0, "head", [5, -10, 0])
    
    builder.set_joint_rotation(2.0, "shoulder_r", [0, 15, 0])
    builder.set_joint_rotation(2.0, "elbow_r", [0, 0, 90])
    builder.set_joint_rotation(2.0, "wrist_r", [0, 0, 30])
    builder.set_joint_rotation(2.0, "head", [5, 10, 0])
    
    # Add context triggers
    builder.add_context_trigger("thinking")
    builder.add_context_trigger("contemplating")
    builder.add_context_trigger("considering")
    
    # Export the gesture
    gesture_data = builder.export_gesture()
    print(json.dumps(gesture_data, indent=2))


if __name__ == "__main__":
    main() 