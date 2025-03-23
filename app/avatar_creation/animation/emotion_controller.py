import os
import numpy as np
import time
import random
import math
from typing import Dict, List, Tuple, Optional, Union, Any

class EmotionController:
    """
    Class for controlling and blending facial emotions and expressions.
    Provides natural transitions between emotional states and intensity adjustment.
    """
    
    def __init__(self, 
                transition_speed: float = 0.5,
                idle_variation: float = 0.2,
                personality_bias: Dict[str, float] = None):
        """
        Initialize the emotion controller.
        
        Args:
            transition_speed: Speed of transitions between emotions (0-1)
            idle_variation: Amount of subtle variation during idle state (0-1)
            personality_bias: Dictionary of emotion biases for the character's personality
        """
        self.transition_speed = min(max(transition_speed, 0.01), 1.0)
        self.idle_variation = min(max(idle_variation, 0.0), 1.0)
        
        # Initialize current emotional state (normalized weights)
        self.current_emotions = {}
        self.target_emotions = {}
        
        # Emotion history for analysis
        self.emotion_history = []
        self.max_history_length = 300  # ~5 seconds at 60fps
        
        # Personality bias (makes character more prone to certain expressions)
        self.personality_bias = personality_bias or {
            'neutral': 0.1,
            'happy': 0.05,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'disgusted': 0.0,
            'fearful': 0.0
        }
        
        # Basic emotion to blendshape mapping
        self.emotion_blendshapes = {
            'neutral': ['face_neutral'],
            'happy': ['face_smile', 'face_cheeks_raised', 'face_mouth_smile'],
            'sad': ['face_sad', 'face_mouth_sad', 'face_brows_sad'],
            'angry': ['face_angry', 'face_brows_angry', 'face_mouth_narrow'],
            'surprised': ['face_surprised', 'face_brows_raised', 'face_eyes_wide', 'face_mouth_open'],
            'disgusted': ['face_disgusted', 'face_nose_wrinkle', 'face_mouth_press'],
            'fearful': ['face_afraid', 'face_brows_worried', 'face_eyes_wide', 'face_mouth_open_wide']
        }
        
        # Blendshape weights for each emotion at 100% intensity
        self.emotion_blendshape_weights = {
            'neutral': {'face_neutral': 1.0},
            'happy': {
                'face_smile': 1.0, 
                'face_cheeks_raised': 0.8, 
                'face_mouth_smile': 0.9
            },
            'sad': {
                'face_sad': 1.0, 
                'face_mouth_sad': 0.8, 
                'face_brows_sad': 0.7
            },
            'angry': {
                'face_angry': 1.0, 
                'face_brows_angry': 0.9, 
                'face_mouth_narrow': 0.7
            },
            'surprised': {
                'face_surprised': 1.0, 
                'face_brows_raised': 0.8, 
                'face_eyes_wide': 0.7, 
                'face_mouth_open': 0.6
            },
            'disgusted': {
                'face_disgusted': 1.0, 
                'face_nose_wrinkle': 0.7, 
                'face_mouth_press': 0.6
            },
            'fearful': {
                'face_afraid': 1.0, 
                'face_brows_worried': 0.8, 
                'face_eyes_wide': 0.7, 
                'face_mouth_open_wide': 0.5
            }
        }
        
        # Complex emotional states (blends of basic emotions)
        self.complex_emotions = {
            'curious': {
                'surprised': 0.3,
                'happy': 0.2
            },
            'anxious': {
                'fearful': 0.5,
                'sad': 0.2
            },
            'excited': {
                'happy': 0.7,
                'surprised': 0.3
            },
            'confused': {
                'surprised': 0.4,
                'sad': 0.1,
                'angry': 0.1
            },
            'disappointed': {
                'sad': 0.6,
                'angry': 0.2
            },
            'content': {
                'happy': 0.4,
                'neutral': 0.6
            },
            'bored': {
                'neutral': 0.7,
                'sad': 0.1
            },
            'amused': {
                'happy': 0.6,
                'surprised': 0.2
            },
            'concerned': {
                'sad': 0.3,
                'fearful': 0.2
            }
        }
        
        # Initialize with neutral expression
        self.set_emotion('neutral', 1.0, immediate=True)
        
        print(f"Emotion Controller initialized")
        print(f"  - Transition speed: {self.transition_speed}")
        print(f"  - Idle variation: {self.idle_variation}")
        if personality_bias:
            print(f"  - Personality biases: {personality_bias}")
    
    def update(self, delta_time: float) -> Dict[str, float]:
        """
        Update emotional state based on elapsed time.
        
        Args:
            delta_time: Time since last update (seconds)
            
        Returns:
            Dictionary of blendshape weights
        """
        current_time = time.time()
        
        # Determine interpolation factor based on transition speed
        interp_factor = min(self.transition_speed * 2.0 * delta_time, 1.0)
        
        # Get current set of emotions (keys)
        current_emotion_set = set(self.current_emotions.keys())
        target_emotion_set = set(self.target_emotions.keys())
        
        # Add keys that are in target but not in current (with weight 0)
        for emotion in target_emotion_set - current_emotion_set:
            self.current_emotions[emotion] = 0.0
        
        # Update current emotion weights towards target
        for emotion in self.current_emotions:
            target_weight = self.target_emotions.get(emotion, 0.0)
            current_weight = self.current_emotions[emotion]
            
            # Interpolate towards target
            new_weight = current_weight + (target_weight - current_weight) * interp_factor
            
            # If weight becomes very small, remove it
            if new_weight < 0.01 and target_weight == 0.0:
                self.current_emotions.pop(emotion, None)
            else:
                self.current_emotions[emotion] = new_weight
        
        # Add subtle variations to active emotions when idle
        if self.idle_variation > 0.0 and len(self.target_emotions) > 0:
            for emotion in self.current_emotions:
                # Only apply variation if emotion is active in target
                if emotion in self.target_emotions and self.target_emotions[emotion] > 0.1:
                    # Calculate subtle random variation based on time
                    t = current_time * 0.5  # slower variation
                    variation = math.sin(t + hash(emotion) % 100) * 0.05 * self.idle_variation
                    self.current_emotions[emotion] = max(0.0, min(1.0, self.current_emotions[emotion] + variation))
        
        # Add to history
        self.emotion_history.append({
            'timestamp': current_time,
            'emotions': self.current_emotions.copy()
        })
        
        # Trim history if needed
        if len(self.emotion_history) > self.max_history_length:
            self.emotion_history.pop(0)
        
        # Convert emotion state to blendshape weights
        blendshape_weights = self._emotions_to_blendshapes(self.current_emotions)
        
        return blendshape_weights
    
    def _emotions_to_blendshapes(self, emotion_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Convert emotion weights to blendshape weights.
        
        Args:
            emotion_weights: Dictionary of emotion weights
            
        Returns:
            Dictionary of blendshape weights
        """
        blendshape_weights = {}
        
        # Process each active emotion
        for emotion, weight in emotion_weights.items():
            if emotion in self.emotion_blendshape_weights:
                # Get blendshape weights for this emotion
                emotion_blendshapes = self.emotion_blendshape_weights[emotion]
                
                # Apply emotion weight to each blendshape
                for blendshape, bs_weight in emotion_blendshapes.items():
                    # Calculate final weight, adding to existing value if present
                    final_weight = weight * bs_weight
                    blendshape_weights[blendshape] = max(
                        blendshape_weights.get(blendshape, 0.0),
                        final_weight
                    )
        
        return blendshape_weights
    
    def set_emotion(self, emotion: str, intensity: float = 1.0, immediate: bool = False) -> None:
        """
        Set a basic or complex emotion.
        
        Args:
            emotion: Name of the emotion
            intensity: Intensity of the emotion (0-1)
            immediate: Whether to apply immediately or transition smoothly
        """
        intensity = min(max(intensity, 0.0), 1.0)
        
        # Clear previous target emotions
        self.target_emotions = {}
        
        if emotion in self.emotion_blendshape_weights:
            # Set a basic emotion
            self.target_emotions[emotion] = intensity
        elif emotion in self.complex_emotions:
            # Set a complex emotion (blend of basic emotions)
            for basic_emotion, ratio in self.complex_emotions[emotion].items():
                self.target_emotions[basic_emotion] = ratio * intensity
        else:
            print(f"Warning: Unknown emotion '{emotion}'. Defaulting to neutral.")
            self.target_emotions['neutral'] = 1.0
        
        # Add personality bias
        for emotion, bias in self.personality_bias.items():
            if emotion in self.target_emotions and bias > 0:
                # Only boost with positive bias
                self.target_emotions[emotion] = min(1.0, self.target_emotions[emotion] + bias)
        
        # If immediate, set current emotions to target
        if immediate:
            self.current_emotions = self.target_emotions.copy()
    
    def blend_emotions(self, emotions: Dict[str, float], intensity: float = 1.0, immediate: bool = False) -> None:
        """
        Set multiple emotions with custom weights.
        
        Args:
            emotions: Dictionary of emotions and their weights
            intensity: Overall intensity scaling factor (0-1)
            immediate: Whether to apply immediately or transition smoothly
        """
        intensity = min(max(intensity, 0.0), 1.0)
        
        # Clear previous target emotions
        self.target_emotions = {}
        
        # Process each emotion
        for emotion, weight in emotions.items():
            # Scale by intensity
            scaled_weight = weight * intensity
            
            if emotion in self.emotion_blendshape_weights:
                # Set a basic emotion
                self.target_emotions[emotion] = scaled_weight
            elif emotion in self.complex_emotions:
                # Set a complex emotion (blend of basic emotions)
                for basic_emotion, ratio in self.complex_emotions[emotion].items():
                    # Add or update the basic emotion weight
                    current = self.target_emotions.get(basic_emotion, 0.0)
                    self.target_emotions[basic_emotion] = min(1.0, current + (ratio * scaled_weight))
        
        # Normalize weights if they sum to more than 1.0
        total_weight = sum(self.target_emotions.values())
        if total_weight > 1.0:
            for emotion in self.target_emotions:
                self.target_emotions[emotion] /= total_weight
        
        # Add personality bias
        for emotion, bias in self.personality_bias.items():
            if emotion in self.target_emotions and bias > 0:
                # Only boost with positive bias
                self.target_emotions[emotion] = min(1.0, self.target_emotions[emotion] + bias)
        
        # If immediate, set current emotions to target
        if immediate:
            self.current_emotions = self.target_emotions.copy()
    
    def add_emotion(self, emotion: str, intensity: float = 0.5, blend_mode: str = 'additive') -> None:
        """
        Add an emotion to the current mix without replacing others.
        
        Args:
            emotion: Name of the emotion to add
            intensity: Intensity of the emotion (0-1)
            blend_mode: How to blend with existing emotions ('additive', 'multiply', 'replace')
        """
        intensity = min(max(intensity, 0.0), 1.0)
        
        # Copy current target emotions
        new_targets = self.target_emotions.copy()
        
        if emotion in self.emotion_blendshape_weights:
            # Add a basic emotion
            if blend_mode == 'additive':
                current = new_targets.get(emotion, 0.0)
                new_targets[emotion] = min(1.0, current + intensity)
            elif blend_mode == 'multiply':
                if emotion in new_targets:
                    new_targets[emotion] *= intensity
            elif blend_mode == 'replace':
                new_targets[emotion] = intensity
        elif emotion in self.complex_emotions:
            # Add a complex emotion (blend of basic emotions)
            for basic_emotion, ratio in self.complex_emotions[emotion].items():
                # Process based on blend mode
                if blend_mode == 'additive':
                    current = new_targets.get(basic_emotion, 0.0)
                    new_targets[basic_emotion] = min(1.0, current + (ratio * intensity))
                elif blend_mode == 'multiply':
                    if basic_emotion in new_targets:
                        new_targets[basic_emotion] *= ratio * intensity
                elif blend_mode == 'replace':
                    new_targets[basic_emotion] = ratio * intensity
        
        # Normalize weights if they sum to more than 1.0
        total_weight = sum(new_targets.values())
        if total_weight > 1.0:
            for e in new_targets:
                new_targets[e] /= total_weight
        
        # Update target emotions
        self.target_emotions = new_targets
    
    def adjust_intensity(self, scale_factor: float) -> None:
        """
        Scale the intensity of all current emotions.
        
        Args:
            scale_factor: Factor to scale emotions by (0-inf)
        """
        scale_factor = max(0.0, scale_factor)
        
        # Scale all target emotions
        for emotion in self.target_emotions:
            self.target_emotions[emotion] *= scale_factor
            
            # Clamp to valid range
            self.target_emotions[emotion] = min(max(self.target_emotions[emotion], 0.0), 1.0)
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """
        Get the current dominant emotion.
        
        Returns:
            Tuple of (emotion_name, intensity)
        """
        if not self.current_emotions:
            return ('neutral', 0.0)
        
        # Find emotion with highest intensity
        dominant_emotion = max(self.current_emotions.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def get_emotion_weights(self) -> Dict[str, float]:
        """
        Get current emotion weights.
        
        Returns:
            Dictionary of current emotion weights
        """
        return self.current_emotions.copy()
    
    def add_complex_emotion(self, 
                           name: str, 
                           components: Dict[str, float]) -> None:
        """
        Add a new complex emotion definition.
        
        Args:
            name: Name of the complex emotion
            components: Dictionary of basic emotions and their weights
        """
        # Normalize weights
        total = sum(components.values())
        if total > 0:
            normalized = {e: w/total for e, w in components.items()}
            self.complex_emotions[name] = normalized
        else:
            print(f"Warning: Cannot add complex emotion '{name}' with zero weights")
    
    def set_personality_bias(self, biases: Dict[str, float]) -> None:
        """
        Set personality biases for emotions.
        
        Args:
            biases: Dictionary of emotions and bias values (-1 to 1)
        """
        # Validate and update biases
        for emotion, bias in biases.items():
            if emotion in self.emotion_blendshape_weights or emotion in self.complex_emotions:
                self.personality_bias[emotion] = max(-0.5, min(0.5, bias))
    
    def reset(self) -> None:
        """
        Reset to neutral expression.
        """
        self.current_emotions = {'neutral': 1.0}
        self.target_emotions = {'neutral': 1.0}
        self.emotion_history = []
        
        print("Emotion controller reset to neutral") 