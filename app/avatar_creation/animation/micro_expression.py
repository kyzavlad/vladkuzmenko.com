import os
import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Union, Any

class MicroExpressionSynthesizer:
    """
    Class for synthesizing natural micro-expressions for avatars.
    Generates subtle facial movements like eye blinks, small mouth movements,
    and other micro-expressions that make avatars appear more lifelike.
    """
    
    def __init__(self, 
                blink_frequency: float = 0.1,
                micro_movement_scale: float = 0.3,
                randomness: float = 0.5):
        """
        Initialize the micro-expression synthesizer.
        
        Args:
            blink_frequency: Frequency of eye blinks (0-1, higher means more frequent)
            micro_movement_scale: Scale of micro-movements (0-1)
            randomness: Randomness factor for expressions (0-1)
        """
        self.blink_frequency = min(max(blink_frequency, 0.0), 1.0)
        self.micro_movement_scale = min(max(micro_movement_scale, 0.0), 1.0)
        self.randomness = min(max(randomness, 0.0), 1.0)
        
        # State tracking
        self.last_blink_time = time.time()
        self.current_micro_expressions = {}
        self.expression_history = []
        self.max_history_length = 60  # frames
        
        # Base micro-expressions dictionary
        self.micro_expressions = {
            'eye_blink': {
                'blendshapes': {
                    'face_eye_blink_left': 1.0,
                    'face_eye_blink_right': 1.0
                },
                'duration': (0.1, 0.3),  # seconds (min, max)
                'probability': self.blink_frequency,
                'cooldown': (2.0, 5.0)  # seconds (min, max)
            },
            'eye_squint': {
                'blendshapes': {
                    'face_eye_squint_left': 0.3,
                    'face_eye_squint_right': 0.3
                },
                'duration': (0.5, 2.0),
                'probability': 0.03,
                'cooldown': (3.0, 10.0)
            },
            'micro_smile': {
                'blendshapes': {
                    'face_smile': 0.15
                },
                'duration': (0.5, 1.5),
                'probability': 0.05,
                'cooldown': (5.0, 15.0)
            },
            'micro_mouth_movement': {
                'blendshapes': {
                    'face_mouth_stretch': 0.1,
                    'face_mouth_sideways': 0.1
                },
                'duration': (0.2, 0.8),
                'probability': 0.04,
                'cooldown': (3.0, 8.0)
            },
            'eyebrow_raise': {
                'blendshapes': {
                    'face_brow_raise_left': 0.2,
                    'face_brow_raise_right': 0.2
                },
                'duration': (0.3, 1.2),
                'probability': 0.03,
                'cooldown': (4.0, 12.0)
            },
            'nose_wrinkle': {
                'blendshapes': {
                    'face_nose_wrinkle': 0.15
                },
                'duration': (0.2, 0.7),
                'probability': 0.02,
                'cooldown': (8.0, 20.0)
            },
            'jaw_clench': {
                'blendshapes': {
                    'face_jaw_clench': 0.2
                },
                'duration': (0.3, 0.9),
                'probability': 0.02,
                'cooldown': (10.0, 25.0)
            }
        }
        
        # Track active expressions
        self.active_expressions = {}  # name -> info dictionary
        
        print(f"Micro-Expression Synthesizer initialized")
        print(f"  - Blink frequency: {self.blink_frequency}")
        print(f"  - Micro-movement scale: {self.micro_movement_scale}")
        print(f"  - Randomness: {self.randomness}")
    
    def update(self, delta_time: float) -> Dict[str, float]:
        """
        Update micro-expressions based on elapsed time.
        
        Args:
            delta_time: Time since last update (seconds)
            
        Returns:
            Dictionary of blendshape weights
        """
        current_time = time.time()
        
        # Initialize output blendshape weights
        blendshape_weights = {}
        
        # Update active expressions and remove completed ones
        expressions_to_remove = []
        
        for expr_name, expr_info in self.active_expressions.items():
            # Update elapsed time
            expr_info['elapsed_time'] += delta_time
            
            # Check if expression is completed
            if expr_info['elapsed_time'] >= expr_info['duration']:
                expressions_to_remove.append(expr_name)
                # Update last activation time
                self.micro_expressions[expr_name]['last_activation'] = current_time
                continue
            
            # Calculate expression progress (0 to 1)
            progress = expr_info['elapsed_time'] / expr_info['duration']
            
            # Apply easing for natural movement
            weight = self._ease_in_out(progress)
            
            # Apply weights for this expression
            for bs_name, bs_max_weight in expr_info['blendshapes'].items():
                # Scale by weight progress
                bs_weight = bs_max_weight * weight
                
                # Update the output weights dictionary
                if bs_name in blendshape_weights:
                    # Take maximum value if already present
                    blendshape_weights[bs_name] = max(blendshape_weights[bs_name], bs_weight)
                else:
                    blendshape_weights[bs_name] = bs_weight
        
        # Remove completed expressions
        for expr_name in expressions_to_remove:
            del self.active_expressions[expr_name]
        
        # Try to trigger new expressions
        for expr_name, expr_data in self.micro_expressions.items():
            # Skip if already active
            if expr_name in self.active_expressions:
                continue
            
            # Check cooldown
            last_activation = expr_data.get('last_activation', 0)
            min_cooldown, max_cooldown = expr_data['cooldown']
            actual_cooldown = min_cooldown + (max_cooldown - min_cooldown) * (1 - self.randomness)
            
            if current_time - last_activation < actual_cooldown:
                continue
            
            # Determine if expression should trigger
            base_probability = expr_data['probability']
            # Scale by randomness (higher randomness = more likely to trigger)
            adjusted_probability = base_probability * (1 + self.randomness)
            # Convert to per-second to per-frame probability
            frame_probability = adjusted_probability * delta_time
            
            if random.random() < frame_probability:
                # Trigger the expression
                min_duration, max_duration = expr_data['duration']
                duration = min_duration + random.random() * (max_duration - min_duration)
                
                # Create modified copy of blendshapes with micro-movement scale applied
                scaled_blendshapes = {}
                for bs_name, bs_weight in expr_data['blendshapes'].items():
                    # Apply micro-movement scale and some randomness
                    scale_factor = self.micro_movement_scale * (1.0 + (random.random() - 0.5) * self.randomness)
                    scaled_blendshapes[bs_name] = bs_weight * scale_factor
                
                # Add to active expressions
                self.active_expressions[expr_name] = {
                    'blendshapes': scaled_blendshapes,
                    'duration': duration,
                    'elapsed_time': 0.0,
                    'start_time': current_time
                }
                
                # Special handling for eye blinks
                if expr_name == 'eye_blink':
                    self.last_blink_time = current_time
        
        # Store expression state in history
        self.expression_history.append({
            'timestamp': current_time,
            'active_expressions': {name: info.copy() for name, info in self.active_expressions.items()},
            'blendshape_weights': blendshape_weights.copy()
        })
        
        # Trim history if needed
        if len(self.expression_history) > self.max_history_length:
            self.expression_history.pop(0)
        
        return blendshape_weights
    
    def _ease_in_out(self, t: float) -> float:
        """
        Apply smooth easing function to make movements more natural.
        
        Args:
            t: Progress value (0 to 1)
            
        Returns:
            Eased value (0 to 1)
        """
        # Cubic ease in-out
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def trigger_expression(self, expression_name: str, intensity: float = 1.0) -> bool:
        """
        Manually trigger a specific micro-expression.
        
        Args:
            expression_name: Name of the expression to trigger
            intensity: Intensity factor (0-1)
            
        Returns:
            True if expression was triggered, False otherwise
        """
        # Check if expression exists
        if expression_name not in self.micro_expressions:
            print(f"Expression not found: {expression_name}")
            return False
        
        # Check if already active
        if expression_name in self.active_expressions:
            # Just update the intensity
            for bs_name in self.active_expressions[expression_name]['blendshapes']:
                orig_weight = self.micro_expressions[expression_name]['blendshapes'][bs_name]
                self.active_expressions[expression_name]['blendshapes'][bs_name] = orig_weight * intensity
            return True
        
        # Get expression data
        expr_data = self.micro_expressions[expression_name]
        
        # Set duration
        min_duration, max_duration = expr_data['duration']
        duration = min_duration + random.random() * (max_duration - min_duration)
        
        # Scale blendshapes by intensity
        scaled_blendshapes = {}
        for bs_name, bs_weight in expr_data['blendshapes'].items():
            scaled_blendshapes[bs_name] = bs_weight * intensity
        
        # Add to active expressions
        self.active_expressions[expression_name] = {
            'blendshapes': scaled_blendshapes,
            'duration': duration,
            'elapsed_time': 0.0,
            'start_time': time.time()
        }
        
        return True
    
    def set_expression_parameter(self, expression_name: str, parameter: str, value: Any) -> bool:
        """
        Set a parameter for a specific micro-expression.
        
        Args:
            expression_name: Name of the expression
            parameter: Parameter name to set
            value: New parameter value
            
        Returns:
            True if parameter was set, False otherwise
        """
        # Check if expression exists
        if expression_name not in self.micro_expressions:
            print(f"Expression not found: {expression_name}")
            return False
        
        # Set the parameter
        try:
            if parameter in self.micro_expressions[expression_name]:
                self.micro_expressions[expression_name][parameter] = value
                return True
            else:
                print(f"Parameter not found: {parameter}")
                return False
        except Exception as e:
            print(f"Error setting parameter: {e}")
            return False
    
    def adjust_frequencies(self, scale_factor: float) -> None:
        """
        Globally adjust the frequency of all micro-expressions.
        
        Args:
            scale_factor: Factor to scale all probabilities (> 1 = more frequent)
        """
        for expr_name in self.micro_expressions:
            # Scale probability but keep between 0 and 1
            orig_prob = self.micro_expressions[expr_name]['probability']
            new_prob = min(max(orig_prob * scale_factor, 0.0), 1.0)
            self.micro_expressions[expr_name]['probability'] = new_prob
    
    def add_custom_expression(self, name: str, blendshapes: Dict[str, float], 
                           duration: Tuple[float, float], probability: float,
                           cooldown: Tuple[float, float]) -> bool:
        """
        Add a custom micro-expression.
        
        Args:
            name: Expression name
            blendshapes: Dictionary of blendshape name to max weight
            duration: Tuple of (min_duration, max_duration) in seconds
            probability: Base probability of triggering
            cooldown: Tuple of (min_cooldown, max_cooldown) in seconds
            
        Returns:
            True if added successfully, False otherwise
        """
        # Validate inputs
        if not name or not blendshapes:
            return False
        
        # Add the expression
        self.micro_expressions[name] = {
            'blendshapes': blendshapes,
            'duration': duration,
            'probability': min(max(probability, 0.0), 1.0),
            'cooldown': cooldown,
            'last_activation': 0
        }
        
        return True
    
    def get_active_expressions(self) -> Dict[str, Dict]:
        """
        Get currently active expressions.
        
        Returns:
            Dictionary of active expressions
        """
        return {name: info.copy() for name, info in self.active_expressions.items()}
    
    def reset(self) -> None:
        """
        Reset the synthesizer to initial state.
        """
        self.last_blink_time = time.time()
        self.active_expressions = {}
        self.expression_history = []
        
        # Reset last activation times
        for expr_name in self.micro_expressions:
            if 'last_activation' in self.micro_expressions[expr_name]:
                self.micro_expressions[expr_name]['last_activation'] = 0
        
        print("Micro-expression synthesizer reset to initial state") 