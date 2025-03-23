import os
import numpy as np
import time
import random
import math
from typing import Dict, List, Tuple, Optional, Union, Any

class GazeController:
    """
    Class for controlling and modeling realistic eye gaze for avatars.
    Implements natural eye movement patterns, including saccades, fixations,
    and smooth pursuit with physiologically plausible constraints.
    """
    
    def __init__(self, 
                saccade_frequency: float = 0.15,
                fixation_duration: float = 0.3,
                max_gaze_angle: float = 30.0,
                natural_movement: bool = True):
        """
        Initialize the gaze controller.
        
        Args:
            saccade_frequency: Frequency of saccadic eye movements (0-1)
            fixation_duration: Average fixation duration in seconds
            max_gaze_angle: Maximum gaze angle in degrees
            natural_movement: Whether to use natural eye movement patterns
        """
        self.saccade_frequency = min(max(saccade_frequency, 0.0), 1.0)
        self.fixation_duration = max(fixation_duration, 0.1)
        self.max_gaze_angle = max(max_gaze_angle, 5.0)
        self.natural_movement = natural_movement
        
        # Current gaze state
        self.current_gaze = np.zeros(2)  # (horizontal, vertical) in degrees
        self.target_gaze = np.zeros(2)
        self.last_saccade_time = time.time()
        self.current_fixation_duration = self.fixation_duration
        
        # Movement parameters
        self.gaze_velocity = np.zeros(2)  # Degrees per second
        self.max_velocity = 300.0  # Degrees per second (for saccades)
        self.pursuit_speed = 40.0  # Degrees per second (for smooth pursuit)
        
        # Physiological constraints
        self.horizontal_limits = (-self.max_gaze_angle, self.max_gaze_angle)
        self.vertical_limits = (-self.max_gaze_angle * 0.7, self.max_gaze_angle * 0.5)  # Asymmetric vertical limits
        
        # Attention points and weights
        self.attention_points = []  # List of (position, weight) tuples
        
        # Gaze history
        self.gaze_history = []
        self.max_history_length = 60  # frames
        
        # Eye blendshape mapping
        self.eye_blendshapes = {
            'left_horizontal': {
                'negative': 'face_eye_look_left',  # Negative = left
                'positive': 'face_eye_look_right'  # Positive = right
            },
            'right_horizontal': {
                'negative': 'face_eye_look_left',
                'positive': 'face_eye_look_right'
            },
            'left_vertical': {
                'negative': 'face_eye_look_down',  # Negative = down
                'positive': 'face_eye_look_up'     # Positive = up
            },
            'right_vertical': {
                'negative': 'face_eye_look_down',
                'positive': 'face_eye_look_up'
            }
        }
        
        # Natural movement patterns
        if self.natural_movement:
            # Physiological microsaccade parameters
            self.microsaccade_amplitude = 0.5  # degrees
            self.microsaccade_frequency = 1.0  # Hz
            self.last_microsaccade_time = time.time()
        
        print(f"Gaze Controller initialized")
        print(f"  - Saccade frequency: {self.saccade_frequency}")
        print(f"  - Fixation duration: {self.fixation_duration}s")
        print(f"  - Max gaze angle: {self.max_gaze_angle}°")
        print(f"  - Natural movement: {self.natural_movement}")
    
    def update(self, delta_time: float) -> Dict[str, float]:
        """
        Update gaze direction based on elapsed time.
        
        Args:
            delta_time: Time since last update (seconds)
            
        Returns:
            Dictionary of eye blendshape weights
        """
        current_time = time.time()
        
        # Check if we should start a new fixation/saccade
        if (current_time - self.last_saccade_time > self.current_fixation_duration and 
            random.random() < self.saccade_frequency * delta_time):
            
            # Generate new target gaze
            self.target_gaze = self._generate_target_gaze()
            
            # Update timing
            self.last_saccade_time = current_time
            self.current_fixation_duration = self._random_fixation_duration()
            
            # Calculate saccade velocity based on distance
            gaze_distance = np.linalg.norm(self.target_gaze - self.current_gaze)
            
            # Apply main sequence relationship (amplitude-velocity relationship)
            # Higher amplitude saccades have higher peak velocities
            peak_velocity = min(gaze_distance * 80, self.max_velocity)
            
            # Set velocity vector
            if gaze_distance > 0:
                self.gaze_velocity = (self.target_gaze - self.current_gaze) / gaze_distance * peak_velocity
            else:
                self.gaze_velocity = np.zeros(2)
        
        # Handle microsaccades for natural eye movement
        if self.natural_movement:
            if current_time - self.last_microsaccade_time > 1.0 / self.microsaccade_frequency:
                # Add small random movement
                microsaccade = np.random.normal(0, self.microsaccade_amplitude, 2)
                self.target_gaze += microsaccade
                
                # Ensure within limits
                self.target_gaze[0] = np.clip(self.target_gaze[0], *self.horizontal_limits)
                self.target_gaze[1] = np.clip(self.target_gaze[1], *self.vertical_limits)
                
                self.last_microsaccade_time = current_time
        
        # Compute distance to target
        to_target = self.target_gaze - self.current_gaze
        distance = np.linalg.norm(to_target)
        
        if distance > 0.1:  # If we're not already at the target
            # Determine movement type based on distance
            if distance > 5.0:
                # Saccadic movement (fast)
                max_step = delta_time * self.max_velocity
            else:
                # Smooth pursuit or slow correction (slower)
                max_step = delta_time * self.pursuit_speed
            
            # Limit step size
            step_size = min(distance, max_step)
            
            # Move towards target
            if distance > 0:
                self.current_gaze += to_target / distance * step_size
        else:
            # We've reached the target, small drift
            if self.natural_movement:
                drift = np.random.normal(0, 0.05, 2) * delta_time
                self.current_gaze += drift
        
        # Ensure gaze stays within physiological limits
        self.current_gaze[0] = np.clip(self.current_gaze[0], *self.horizontal_limits)
        self.current_gaze[1] = np.clip(self.current_gaze[1], *self.vertical_limits)
        
        # Add to history
        self.gaze_history.append({
            'timestamp': current_time,
            'gaze': self.current_gaze.copy(),
            'target': self.target_gaze.copy()
        })
        
        # Trim history if needed
        if len(self.gaze_history) > self.max_history_length:
            self.gaze_history.pop(0)
        
        # Convert gaze angles to blendshape weights
        blendshape_weights = self._gaze_to_blendshapes(self.current_gaze)
        
        return blendshape_weights
    
    def _generate_target_gaze(self) -> np.ndarray:
        """
        Generate a new target gaze direction.
        
        Returns:
            New target gaze as (horizontal, vertical) angles in degrees
        """
        # Check if we have attention points
        if self.attention_points and random.random() < 0.7:  # 70% chance to look at an attention point
            # Sample an attention point based on weights
            weights = [w for _, w in self.attention_points]
            total_weight = sum(weights)
            
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                chosen_idx = np.random.choice(len(self.attention_points), p=normalized_weights)
                chosen_point, _ = self.attention_points[chosen_idx]
                
                # Return the chosen attention point
                return np.array(chosen_point)
        
        # Otherwise generate a random gaze target
        horizontal = random.uniform(*self.horizontal_limits)
        vertical = random.uniform(*self.vertical_limits)
        
        # Bias towards center (more natural)
        center_bias = 0.3
        horizontal = horizontal * (1 - center_bias)
        vertical = vertical * (1 - center_bias)
        
        return np.array([horizontal, vertical])
    
    def _random_fixation_duration(self) -> float:
        """
        Generate a random fixation duration.
        
        Returns:
            Fixation duration in seconds
        """
        # Log-normal distribution for more natural timing
        mean = math.log(self.fixation_duration)
        sigma = 0.4  # Controls spread
        
        # Sample from log-normal distribution
        duration = random.lognormvariate(mean, sigma)
        
        # Ensure reasonable limits
        return min(max(duration, 0.1), 5.0)
    
    def _gaze_to_blendshapes(self, gaze: np.ndarray) -> Dict[str, float]:
        """
        Convert gaze direction to eye blendshape weights.
        
        Args:
            gaze: Gaze direction as (horizontal, vertical) angles in degrees
            
        Returns:
            Dictionary of blendshape weights
        """
        horizontal_angle, vertical_angle = gaze
        
        # Scale to 0-1 range for blendshapes
        h_range = self.horizontal_limits[1] - self.horizontal_limits[0]
        v_range = self.vertical_limits[1] - self.vertical_limits[0]
        
        # Handle horizontal movement
        if horizontal_angle < 0:  # Looking left
            h_left = abs(horizontal_angle) / abs(self.horizontal_limits[0])
            h_right = 0.0
        else:  # Looking right
            h_left = 0.0
            h_right = horizontal_angle / self.horizontal_limits[1]
        
        # Handle vertical movement
        if vertical_angle < 0:  # Looking down
            v_down = abs(vertical_angle) / abs(self.vertical_limits[0])
            v_up = 0.0
        else:  # Looking up
            v_down = 0.0
            v_up = vertical_angle / self.vertical_limits[1]
        
        # Cap values
        h_left = min(h_left, 1.0)
        h_right = min(h_right, 1.0)
        v_down = min(v_down, 1.0)
        v_up = min(v_up, 1.0)
        
        # Create blendshape weights dictionary
        weights = {}
        
        # Left eye
        if h_left > 0:
            weights[self.eye_blendshapes['left_horizontal']['negative']] = h_left
        if h_right > 0:
            weights[self.eye_blendshapes['left_horizontal']['positive']] = h_right
        
        if v_down > 0:
            weights[self.eye_blendshapes['left_vertical']['negative']] = v_down
        if v_up > 0:
            weights[self.eye_blendshapes['left_vertical']['positive']] = v_up
        
        # Right eye (same weights as left eye for conjugate eye movements)
        if h_left > 0:
            weights[self.eye_blendshapes['right_horizontal']['negative']] = h_left
        if h_right > 0:
            weights[self.eye_blendshapes['right_horizontal']['positive']] = h_right
        
        if v_down > 0:
            weights[self.eye_blendshapes['right_vertical']['negative']] = v_down
        if v_up > 0:
            weights[self.eye_blendshapes['right_vertical']['positive']] = v_up
        
        return weights
    
    def set_gaze_target(self, horizontal: float, vertical: float, immediate: bool = False) -> None:
        """
        Set a specific gaze target.
        
        Args:
            horizontal: Horizontal angle in degrees (negative = left, positive = right)
            vertical: Vertical angle in degrees (negative = down, positive = up)
            immediate: Whether to move immediately or animate
        """
        # Clamp to limits
        h = np.clip(horizontal, *self.horizontal_limits)
        v = np.clip(vertical, *self.vertical_limits)
        
        # Set target
        self.target_gaze = np.array([h, v])
        
        # If immediate, set current gaze to target
        if immediate:
            self.current_gaze = np.array([h, v])
    
    def add_attention_point(self, horizontal: float, vertical: float, weight: float = 1.0) -> int:
        """
        Add an attention point for the gaze to focus on.
        
        Args:
            horizontal: Horizontal angle in degrees
            vertical: Vertical angle in degrees
            weight: Attention weight (higher = more likely to look)
            
        Returns:
            Index of the added attention point
        """
        # Clamp to limits
        h = np.clip(horizontal, *self.horizontal_limits)
        v = np.clip(vertical, *self.vertical_limits)
        
        # Add to list
        self.attention_points.append(((h, v), max(weight, 0.0)))
        
        return len(self.attention_points) - 1
    
    def clear_attention_points(self) -> None:
        """
        Clear all attention points.
        """
        self.attention_points = []
    
    def set_attention_point_weight(self, index: int, weight: float) -> bool:
        """
        Update the weight of an attention point.
        
        Args:
            index: Index of the attention point
            weight: New weight
            
        Returns:
            True if successful, False otherwise
        """
        if 0 <= index < len(self.attention_points):
            point, _ = self.attention_points[index]
            self.attention_points[index] = (point, max(weight, 0.0))
            return True
        return False
    
    def follow_target(self, horizontal: float, vertical: float, speed: Optional[float] = None) -> None:
        """
        Follow a moving target with smooth pursuit.
        
        Args:
            horizontal: Horizontal angle in degrees
            vertical: Vertical angle in degrees
            speed: Pursuit speed in degrees per second (None for default)
        """
        # Clamp to limits
        h = np.clip(horizontal, *self.horizontal_limits)
        v = np.clip(vertical, *self.vertical_limits)
        
        # Set target
        self.target_gaze = np.array([h, v])
        
        # Set pursuit speed if specified
        if speed is not None:
            self.pursuit_speed = max(speed, 5.0)
    
    def perform_saccade(self, horizontal: float, vertical: float) -> None:
        """
        Perform an immediate saccade to a target.
        
        Args:
            horizontal: Horizontal angle in degrees
            vertical: Vertical angle in degrees
        """
        # Clamp to limits
        h = np.clip(horizontal, *self.horizontal_limits)
        v = np.clip(vertical, *self.vertical_limits)
        
        # Set target
        self.target_gaze = np.array([h, v])
        
        # Reset saccade timer
        self.last_saccade_time = time.time()
        self.current_fixation_duration = self._random_fixation_duration()
        
        # Calculate saccade velocity based on distance
        gaze_distance = np.linalg.norm(self.target_gaze - self.current_gaze)
        
        # Apply main sequence relationship
        peak_velocity = min(gaze_distance * 80, self.max_velocity)
        
        # Set velocity vector
        if gaze_distance > 0:
            self.gaze_velocity = (self.target_gaze - self.current_gaze) / gaze_distance * peak_velocity
        else:
            self.gaze_velocity = np.zeros(2)
    
    def get_current_gaze(self) -> Tuple[float, float]:
        """
        Get the current gaze direction.
        
        Returns:
            Tuple of (horizontal, vertical) angles in degrees
        """
        return (self.current_gaze[0], self.current_gaze[1])
    
    def reset(self) -> None:
        """
        Reset the gaze controller to initial state.
        """
        self.current_gaze = np.zeros(2)
        self.target_gaze = np.zeros(2)
        self.gaze_velocity = np.zeros(2)
        self.last_saccade_time = time.time()
        self.gaze_history = []
        self.attention_points = []
        
        print("Gaze controller reset to initial state") 