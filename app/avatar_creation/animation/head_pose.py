import os
import numpy as np
import time
import random
import math
from typing import Dict, List, Tuple, Optional, Union, Any

class HeadPoseController:
    """
    Class for controlling realistic head pose variations for avatars.
    Implements natural head movement patterns with physiologically plausible constraints.
    """
    
    def __init__(self, 
                movement_scale: float = 0.5,
                movement_frequency: float = 0.2,
                natural_motion: bool = True,
                follow_gaze: bool = True):
        """
        Initialize the head pose controller.
        
        Args:
            movement_scale: Scale of head movements (0-1)
            movement_frequency: Frequency of head movements (0-1)
            natural_motion: Whether to add subtle natural head motion
            follow_gaze: Whether head should follow gaze for large eye movements
        """
        self.movement_scale = min(max(movement_scale, 0.0), 1.0)
        self.movement_frequency = min(max(movement_frequency, 0.0), 1.0)
        self.natural_motion = natural_motion
        self.follow_gaze = follow_gaze
        
        # Current pose state (degrees)
        self.current_pose = np.zeros(3)  # (pitch, yaw, roll)
        self.target_pose = np.zeros(3)
        self.last_movement_time = time.time()
        self.pose_velocity = np.zeros(3)  # Degrees per second
        
        # Physical limits and constraints
        self.pitch_limits = (-30.0 * self.movement_scale, 30.0 * self.movement_scale)  # Up/down
        self.yaw_limits = (-60.0 * self.movement_scale, 60.0 * self.movement_scale)    # Left/right
        self.roll_limits = (-20.0 * self.movement_scale, 20.0 * self.movement_scale)   # Tilt
        
        # Movement parameters
        self.max_velocity = 90.0  # Degrees per second
        self.natural_drift_scale = 1.0 if self.natural_motion else 0.0
        
        # Pose history
        self.pose_history = []
        self.max_history_length = 60  # frames
        
        # Typical head postures and their probabilities
        self.typical_poses = {
            'neutral': {
                'pose': np.array([0.0, 0.0, 0.0]),
                'probability': 0.4
            },
            'slight_right': {
                'pose': np.array([0.0, 15.0, 0.0]),
                'probability': 0.1
            },
            'slight_left': {
                'pose': np.array([0.0, -15.0, 0.0]),
                'probability': 0.1
            },
            'slight_down': {
                'pose': np.array([15.0, 0.0, 0.0]),
                'probability': 0.1
            },
            'slight_up': {
                'pose': np.array([-10.0, 0.0, 0.0]),
                'probability': 0.05
            },
            'down_right': {
                'pose': np.array([10.0, 10.0, 0.0]),
                'probability': 0.05
            },
            'down_left': {
                'pose': np.array([10.0, -10.0, 0.0]),
                'probability': 0.05
            },
            'up_right': {
                'pose': np.array([-10.0, 10.0, 0.0]),
                'probability': 0.05
            },
            'up_left': {
                'pose': np.array([-10.0, -10.0, 0.0]),
                'probability': 0.05
            },
            'tilt_right': {
                'pose': np.array([0.0, 0.0, 8.0]),
                'probability': 0.025
            },
            'tilt_left': {
                'pose': np.array([0.0, 0.0, -8.0]),
                'probability': 0.025
            }
        }
        
        # Scale typical poses by movement scale
        for posture in self.typical_poses.values():
            posture['pose'] = posture['pose'] * self.movement_scale
        
        # Blendshape mappings for head pose
        self.head_pose_blendshapes = {
            'pitch_negative': 'face_head_down',   # Looking down (positive pitch)
            'pitch_positive': 'face_head_up',     # Looking up (negative pitch)
            'yaw_negative': 'face_head_left',     # Looking left (negative yaw)
            'yaw_positive': 'face_head_right',    # Looking right (positive yaw)
            'roll_negative': 'face_head_tilt_left',  # Tilting left (negative roll)
            'roll_positive': 'face_head_tilt_right'  # Tilting right (positive roll)
        }
        
        # Neck/head motion parameters (for natural movement)
        if self.natural_motion:
            self.micro_motion_scale = 0.1
            self.micro_motion_frequency = 0.3
            self.last_micro_motion_time = time.time()
        
        print(f"Head Pose Controller initialized")
        print(f"  - Movement scale: {self.movement_scale}")
        print(f"  - Movement frequency: {self.movement_frequency}")
        print(f"  - Natural motion: {self.natural_motion}")
        print(f"  - Follow gaze: {self.follow_gaze}")
    
    def update(self, delta_time: float, gaze_direction: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Update head pose based on elapsed time and optionally follow gaze.
        
        Args:
            delta_time: Time since last update (seconds)
            gaze_direction: Current gaze direction as (horizontal, vertical) in degrees
            
        Returns:
            Dictionary of head pose blendshape weights
        """
        current_time = time.time()
        
        # Check if we should change pose target
        if (current_time - self.last_movement_time > 3.0 and 
            random.random() < self.movement_frequency * delta_time):
            
            # Generate new target pose
            self.target_pose = self._generate_target_pose()
            
            # Update timing
            self.last_movement_time = current_time
            
            # Calculate velocity based on distance
            pose_distance = np.linalg.norm(self.target_pose - self.current_pose)
            
            # Set velocity based on distance
            if pose_distance > 0:
                self.pose_velocity = (self.target_pose - self.current_pose) / pose_distance * self.max_velocity * random.uniform(0.3, 1.0)
            else:
                self.pose_velocity = np.zeros(3)
        
        # Handle gaze following
        if self.follow_gaze and gaze_direction is not None:
            horizontal_gaze, vertical_gaze = gaze_direction
            
            # Only follow gaze for large eye movements
            h_threshold = 20.0  # degrees
            v_threshold = 15.0  # degrees
            
            # Calculate how much the head should follow gaze
            # (head follows eyes partially, not 1:1)
            follow_ratio = 0.3  # 30% of eye movement
            
            # Horizontal (yaw) following
            if abs(horizontal_gaze) > h_threshold:
                # Calculate target yaw to follow gaze
                target_yaw = horizontal_gaze * follow_ratio
                
                # Blend with current target
                self.target_pose[1] = 0.7 * self.target_pose[1] + 0.3 * target_yaw
                
                # Ensure within limits
                self.target_pose[1] = np.clip(self.target_pose[1], *self.yaw_limits)
            
            # Vertical (pitch) following
            if abs(vertical_gaze) > v_threshold:
                # Calculate target pitch to follow gaze
                # Note: negative vertical gaze (looking down) = positive pitch
                target_pitch = -vertical_gaze * follow_ratio
                
                # Blend with current target
                self.target_pose[0] = 0.7 * self.target_pose[0] + 0.3 * target_pitch
                
                # Ensure within limits
                self.target_pose[0] = np.clip(self.target_pose[0], *self.pitch_limits)
        
        # Add natural micro-motion
        if self.natural_motion:
            if current_time - self.last_micro_motion_time > 0.2:  # 5 Hz update rate
                # Add small natural drift
                drift = np.random.normal(0, self.micro_motion_scale, 3) * delta_time * self.natural_drift_scale
                self.current_pose += drift
                
                # Ensure within limits
                self.current_pose[0] = np.clip(self.current_pose[0], *self.pitch_limits)
                self.current_pose[1] = np.clip(self.current_pose[1], *self.yaw_limits)
                self.current_pose[2] = np.clip(self.current_pose[2], *self.roll_limits)
                
                self.last_micro_motion_time = current_time
        
        # Compute distance to target
        to_target = self.target_pose - self.current_pose
        distance = np.linalg.norm(to_target)
        
        if distance > 0.1:  # If we're not already at the target
            # Calculate step size
            max_step = delta_time * self.max_velocity
            step_size = min(distance, max_step)
            
            # Move towards target
            if distance > 0:
                self.current_pose += to_target / distance * step_size
            
            # Apply damping to slow down as we approach target
            damping_radius = 5.0  # degrees
            if distance < damping_radius:
                damping = distance / damping_radius
                self.current_pose = self.current_pose * damping + self.target_pose * (1 - damping)
        
        # Ensure pose stays within physiological limits
        self.current_pose[0] = np.clip(self.current_pose[0], *self.pitch_limits)
        self.current_pose[1] = np.clip(self.current_pose[1], *self.yaw_limits)
        self.current_pose[2] = np.clip(self.current_pose[2], *self.roll_limits)
        
        # Add to history
        self.pose_history.append({
            'timestamp': current_time,
            'pose': self.current_pose.copy(),
            'target': self.target_pose.copy()
        })
        
        # Trim history if needed
        if len(self.pose_history) > self.max_history_length:
            self.pose_history.pop(0)
        
        # Convert pose angles to blendshape weights
        blendshape_weights = self._pose_to_blendshapes(self.current_pose)
        
        return blendshape_weights
    
    def _generate_target_pose(self) -> np.ndarray:
        """
        Generate a new target head pose.
        
        Returns:
            New target pose as (pitch, yaw, roll) angles in degrees
        """
        # Either select a typical pose or generate a random one
        if random.random() < 0.8:  # 80% chance to use a typical pose
            # Sample a typical pose based on probabilities
            poses = list(self.typical_poses.values())
            probabilities = [p['probability'] for p in poses]
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob > 0:
                normalized_probs = [p / total_prob for p in probabilities]
                chosen_idx = np.random.choice(len(poses), p=normalized_probs)
                
                # Get the chosen pose and add small randomization
                chosen_pose = poses[chosen_idx]['pose'].copy()
                randomization = np.random.normal(0, 2.0, 3) * self.movement_scale
                chosen_pose += randomization
                
                # Ensure within limits
                chosen_pose[0] = np.clip(chosen_pose[0], *self.pitch_limits)
                chosen_pose[1] = np.clip(chosen_pose[1], *self.yaw_limits)
                chosen_pose[2] = np.clip(chosen_pose[2], *self.roll_limits)
                
                return chosen_pose
        
        # Generate random pose
        pitch = random.uniform(*self.pitch_limits)
        yaw = random.uniform(*self.yaw_limits)
        roll = random.uniform(*self.roll_limits) * 0.5  # Less roll variation
        
        # Center bias for more natural distribution
        center_bias = 0.4
        pitch = pitch * (1 - center_bias)
        yaw = yaw * (1 - center_bias)
        roll = roll * (1 - center_bias)
        
        return np.array([pitch, yaw, roll])
    
    def _pose_to_blendshapes(self, pose: np.ndarray) -> Dict[str, float]:
        """
        Convert head pose to blendshape weights.
        
        Args:
            pose: Head pose as (pitch, yaw, roll) angles in degrees
            
        Returns:
            Dictionary of blendshape weights
        """
        pitch, yaw, roll = pose
        
        # Convert angles to normalized weights (0-1)
        # Pitch: positive = down, negative = up
        if pitch > 0:  # Looking down
            pitch_down = min(pitch / self.pitch_limits[1], 1.0)
            pitch_up = 0.0
        else:  # Looking up
            pitch_down = 0.0
            pitch_up = min(abs(pitch) / abs(self.pitch_limits[0]), 1.0)
        
        # Yaw: positive = right, negative = left
        if yaw > 0:  # Looking right
            yaw_right = min(yaw / self.yaw_limits[1], 1.0)
            yaw_left = 0.0
        else:  # Looking left
            yaw_right = 0.0
            yaw_left = min(abs(yaw) / abs(self.yaw_limits[0]), 1.0)
        
        # Roll: positive = right tilt, negative = left tilt
        if roll > 0:  # Tilting right
            roll_right = min(roll / self.roll_limits[1], 1.0)
            roll_left = 0.0
        else:  # Tilting left
            roll_right = 0.0
            roll_left = min(abs(roll) / abs(self.roll_limits[0]), 1.0)
        
        # Create blendshape weights dictionary
        weights = {}
        
        # Add weights for pitch
        if pitch_down > 0:
            weights[self.head_pose_blendshapes['pitch_negative']] = pitch_down
        if pitch_up > 0:
            weights[self.head_pose_blendshapes['pitch_positive']] = pitch_up
        
        # Add weights for yaw
        if yaw_left > 0:
            weights[self.head_pose_blendshapes['yaw_negative']] = yaw_left
        if yaw_right > 0:
            weights[self.head_pose_blendshapes['yaw_positive']] = yaw_right
        
        # Add weights for roll
        if roll_left > 0:
            weights[self.head_pose_blendshapes['roll_negative']] = roll_left
        if roll_right > 0:
            weights[self.head_pose_blendshapes['roll_positive']] = roll_right
        
        return weights
    
    def set_pose_target(self, pitch: float, yaw: float, roll: float = 0.0, immediate: bool = False) -> None:
        """
        Set a specific head pose target.
        
        Args:
            pitch: Pitch angle in degrees (positive = down, negative = up)
            yaw: Yaw angle in degrees (positive = right, negative = left)
            roll: Roll angle in degrees (positive = right tilt, negative = left tilt)
            immediate: Whether to move immediately or animate
        """
        # Clamp to limits
        p = np.clip(pitch, *self.pitch_limits)
        y = np.clip(yaw, *self.yaw_limits)
        r = np.clip(roll, *self.roll_limits)
        
        # Set target
        self.target_pose = np.array([p, y, r])
        
        # If immediate, set current pose to target
        if immediate:
            self.current_pose = np.array([p, y, r])
    
    def set_movement_scale(self, scale: float) -> None:
        """
        Set the scale of head movements.
        
        Args:
            scale: Movement scale (0-1)
        """
        old_scale = self.movement_scale
        self.movement_scale = min(max(scale, 0.0), 1.0)
        
        # Scale ratio between old and new
        if old_scale > 0:
            scale_ratio = self.movement_scale / old_scale
            
            # Update limits
            self.pitch_limits = (self.pitch_limits[0] * scale_ratio, self.pitch_limits[1] * scale_ratio)
            self.yaw_limits = (self.yaw_limits[0] * scale_ratio, self.yaw_limits[1] * scale_ratio)
            self.roll_limits = (self.roll_limits[0] * scale_ratio, self.roll_limits[1] * scale_ratio)
            
            # Update current and target poses
            self.current_pose = self.current_pose * scale_ratio
            self.target_pose = self.target_pose * scale_ratio
            
            # Update typical poses
            for posture in self.typical_poses.values():
                posture['pose'] = posture['pose'] * scale_ratio
    
    def look_at_point(self, horizontal_angle: float, vertical_angle: float) -> None:
        """
        Orient the head to look at a point in space.
        
        Args:
            horizontal_angle: Horizontal angle in degrees
            vertical_angle: Vertical angle in degrees
        """
        # Convert gaze angles to head pose
        # Head follows gaze partially
        follow_ratio = 0.7  # 70% of gaze angle
        
        # Calculate target pose angles
        target_yaw = horizontal_angle * follow_ratio
        target_pitch = -vertical_angle * follow_ratio  # Negative because looking down = positive pitch
        
        # Set the pose target
        self.set_pose_target(target_pitch, target_yaw)
    
    def add_natural_nod(self, intensity: float = 1.0) -> None:
        """
        Add a natural head nod (vertical movement).
        
        Args:
            intensity: Intensity of the nod (0-1)
        """
        # Define a natural nod sequence
        nod_duration = 1.0  # seconds
        nod_angle = 15.0 * self.movement_scale * intensity
        
        # Save current target
        original_target = self.target_pose.copy()
        
        # Set initial nod down target
        self.target_pose[0] = nod_angle  # Positive pitch = down
        self.pose_velocity[0] = nod_angle / (nod_duration * 0.3)  # 30% of duration for down movement
        
        # Schedule a timer to complete the nod sequence
        # Since we can't use actual timers in this implementation, in practice
        # you would integrate this with your animation system's update loop
        
        # For demonstration purposes, we'll set up the nod pattern
        # In a real implementation, you would trigger these pose changes on a timer
        nod_sequence = [
            {'pose': np.array([nod_angle, original_target[1], original_target[2]]), 'time': 0.3 * nod_duration},
            {'pose': np.array([-nod_angle * 0.7, original_target[1], original_target[2]]), 'time': 0.6 * nod_duration},
            {'pose': np.array([nod_angle * 0.3, original_target[1], original_target[2]]), 'time': 0.8 * nod_duration},
            {'pose': original_target, 'time': 1.0 * nod_duration}
        ]
        
        # In a real implementation, you would execute this sequence over time
    
    def add_natural_shake(self, intensity: float = 1.0) -> None:
        """
        Add a natural head shake (horizontal movement).
        
        Args:
            intensity: Intensity of the shake (0-1)
        """
        # Define a natural shake sequence
        shake_duration = 1.2  # seconds
        shake_angle = 20.0 * self.movement_scale * intensity
        
        # Save current target
        original_target = self.target_pose.copy()
        
        # Set initial shake target
        self.target_pose[1] = -shake_angle  # Negative yaw = left
        self.pose_velocity[1] = shake_angle / (shake_duration * 0.25)  # 25% of duration for initial movement
        
        # For demonstration purposes, we'll set up the shake pattern
        # In a real implementation, you would trigger these pose changes on a timer
        shake_sequence = [
            {'pose': np.array([original_target[0], -shake_angle, original_target[2]]), 'time': 0.25 * shake_duration},
            {'pose': np.array([original_target[0], shake_angle, original_target[2]]), 'time': 0.5 * shake_duration},
            {'pose': np.array([original_target[0], -shake_angle * 0.7, original_target[2]]), 'time': 0.75 * shake_duration},
            {'pose': np.array([original_target[0], shake_angle * 0.4, original_target[2]]), 'time': 0.9 * shake_duration},
            {'pose': original_target, 'time': 1.0 * shake_duration}
        ]
        
        # In a real implementation, you would execute this sequence over time
    
    def get_current_pose(self) -> Tuple[float, float, float]:
        """
        Get the current head pose.
        
        Returns:
            Tuple of (pitch, yaw, roll) angles in degrees
        """
        return tuple(self.current_pose)
    
    def reset(self) -> None:
        """
        Reset the head pose controller to initial state.
        """
        self.current_pose = np.zeros(3)
        self.target_pose = np.zeros(3)
        self.pose_velocity = np.zeros(3)
        self.last_movement_time = time.time()
        self.pose_history = []
        
        print("Head pose controller reset to initial state") 