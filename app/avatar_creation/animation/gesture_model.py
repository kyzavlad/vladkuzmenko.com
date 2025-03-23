import os
import numpy as np
import time
import random
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class GestureModelConfig:
    """Configuration settings for the gesture learning model."""
    use_gpu: bool = True
    observation_window: int = 300  # frames to observe before starting to learn
    personality_influence: float = 0.5  # how much personality affects gestures
    learning_rate: float = 0.05  # rate of learning new patterns
    gesture_variety: float = 0.7  # variety of generated gestures (0-1)
    model_path: str = ''  # path to pre-trained or saved model
    min_pattern_length: int = 30  # minimum frames for pattern recognition
    max_pattern_length: int = 300  # maximum frames for pattern recognition
    context_awareness: bool = True  # whether to be aware of speech/emotion context

class GestureMannerismLearner:
    """
    Learning model for person-specific gestures and mannerisms.
    Analyzes video to extract characteristic movement patterns and can generate
    similar movements for avatar animation.
    """
    
    def __init__(self, config: GestureModelConfig = None):
        """
        Initialize the gesture and mannerism learning model.
        
        Args:
            config: Configuration for the gesture model
        """
        self.config = config or GestureModelConfig()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")
        
        # Motion patterns observed and learned
        self.motion_patterns = {
            'head': [],  # head movement patterns
            'hands': [],  # hand gesture patterns
            'posture': [],  # posture shift patterns
            'emphasis': [],  # emphasis gestures
            'idle': []  # idle state patterns
        }
        
        # Personality traits that influence gesture patterns
        self.personality_traits = {
            'expressiveness': 0.5,  # how much a person gestures (0-1)
            'energy': 0.5,  # energy level in movements (0-1)
            'formality': 0.5,  # formality in posture and gestures (0-1)
            'confidence': 0.5,  # confidence expressed in movements (0-1)
            'rhythm': 0.5,  # rhythmic quality of movements (0-1)
            'space_usage': 0.5,  # how much physical space is used (0-1)
        }
        
        # Context detection
        self.context = {
            'speaking': False,
            'listening': False,
            'emphasizing': False,
            'thinking': False,
            'emotional_state': 'neutral',
            'emotional_intensity': 0.0
        }
        
        # Observation state and history
        self.observation_frames = 0
        self.observation_history = []
        self.current_sequence = []
        self.max_history_length = 1000  # frames
        self.observed_sequences = []
        self.has_learned_patterns = False
        
        # Load pre-trained model if path provided
        if self.config.model_path and os.path.exists(self.config.model_path):
            self.load_model(self.config.model_path)
        
        print(f"Gesture & Mannerism Learning Model initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Observation window: {self.config.observation_window} frames")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Context awareness: {self.config.context_awareness}")
    
    def observe_motion(self, 
                      pose_data: Dict[str, Any], 
                      is_speaking: bool = False,
                      emotional_state: str = "neutral",
                      emotional_intensity: float = 0.0) -> None:
        """
        Observe and process motion data for learning.
        
        Args:
            pose_data: Dictionary containing pose keypoints and metadata
            is_speaking: Whether the person is currently speaking
            emotional_state: Current emotional state
            emotional_intensity: Intensity of current emotion (0-1)
        """
        # Update observation counter
        self.observation_frames += 1
        
        # Update context
        self.context['speaking'] = is_speaking
        self.context['emotional_state'] = emotional_state
        self.context['emotional_intensity'] = emotional_intensity
        
        # Extract features from pose data
        features = self._extract_motion_features(pose_data)
        
        # Add timestamp and context
        features.update({
            'timestamp': time.time(),
            'is_speaking': is_speaking,
            'emotional_state': emotional_state,
            'emotional_intensity': emotional_intensity
        })
        
        # Add to current sequence and history
        self.current_sequence.append(features)
        self.observation_history.append(features)
        
        # Trim history if needed
        if len(self.observation_history) > self.max_history_length:
            self.observation_history.pop(0)
        
        # Check if we've observed enough data
        if self.observation_frames >= self.config.observation_window:
            # Process sequences periodically
            if len(self.current_sequence) >= self.config.min_pattern_length:
                if random.random() < 0.1:  # Avoid processing every frame
                    self._process_current_sequence()
        
        # Detect when a natural sequence may have ended
        if len(self.current_sequence) > 0:
            if self._is_sequence_end(features):
                self._process_current_sequence()
    
    def generate_gesture(self, 
                        context: Dict[str, Any] = None,
                        duration_sec: float = 2.0,
                        gesture_type: str = None) -> Dict[str, Any]:
        """
        Generate a gesture based on learned patterns.
        
        Args:
            context: Current context information
            duration_sec: Duration of gesture in seconds
            gesture_type: Type of gesture to generate (or None for automatic)
            
        Returns:
            Dictionary with gesture animation data
        """
        # Default context if none provided
        if context is None:
            context = self.context.copy()
        
        # If we haven't learned enough patterns, generate a default gesture
        if not self.has_learned_patterns:
            return self._generate_default_gesture(duration_sec, gesture_type)
        
        # Select gesture type based on context if not specified
        if gesture_type is None:
            gesture_type = self._select_gesture_type(context)
        
        # Get patterns for the selected gesture type
        patterns = self.motion_patterns.get(gesture_type, [])
        
        if not patterns:
            return self._generate_default_gesture(duration_sec, gesture_type)
        
        # Select a pattern based on context matching and randomness
        selected_pattern = self._select_pattern_for_context(patterns, context)
        
        # Add variation to make it less repetitive
        variation_level = self.config.gesture_variety
        modified_pattern = self._add_variation_to_pattern(selected_pattern, variation_level)
        
        # Adjust to requested duration
        scaled_pattern = self._scale_pattern_to_duration(modified_pattern, duration_sec)
        
        # Format for animation system
        animation_data = self._format_for_animation(scaled_pattern, context)
        
        return animation_data
    
    def _extract_motion_features(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract meaningful features from raw pose data.
        
        Args:
            pose_data: Raw pose data with keypoints
            
        Returns:
            Dictionary of extracted motion features
        """
        features = {}
        
        # Placeholder for feature extraction from pose data
        # In a real implementation, this would extract meaningful motion features
        # such as hand positions, head angles, upper body pose, etc.
        
        # Simple placeholder implementation
        if 'keypoints' in pose_data:
            # Extract head position/rotation if available
            if 'head_pose' in pose_data:
                features['head_pitch'] = pose_data['head_pose'][0]
                features['head_yaw'] = pose_data['head_pose'][1]
                features['head_roll'] = pose_data['head_pose'][2]
            
            # Example: extract hand positions
            keypoints = pose_data['keypoints']
            if 'left_hand' in keypoints and 'right_hand' in keypoints:
                features['left_hand_pos'] = keypoints['left_hand']
                features['right_hand_pos'] = keypoints['right_hand']
            
            # Example: extract overall body posture
            if 'spine' in keypoints and 'shoulders' in keypoints:
                features['spine_angle'] = keypoints['spine']
                features['shoulder_width'] = keypoints['shoulders']
        
        # Calculate velocities if we have previous frames
        if self.current_sequence:
            prev = self.current_sequence[-1]
            for key in features:
                if key in prev and isinstance(features[key], (list, tuple, np.ndarray)):
                    # For position data, calculate velocity
                    prev_val = np.array(prev[key])
                    curr_val = np.array(features[key])
                    features[f"{key}_velocity"] = curr_val - prev_val
        
        return features
    
    def _is_sequence_end(self, features: Dict[str, Any]) -> bool:
        """
        Detect if the current sequence has naturally ended.
        
        Args:
            features: Current frame features
            
        Returns:
            True if sequence appears to have ended
        """
        # If sequence too short, it hasn't ended
        if len(self.current_sequence) < self.config.min_pattern_length:
            return False
        
        # If sequence too long, end it anyway
        if len(self.current_sequence) >= self.config.max_pattern_length:
            return True
        
        # Check for context changes that might indicate sequence boundaries
        if self.config.context_awareness:
            if len(self.current_sequence) > 0:
                prev = self.current_sequence[-1]
                
                # Speaking/not speaking transition
                if prev['is_speaking'] != features['is_speaking']:
                    return True
                
                # Emotional state change
                if prev['emotional_state'] != features['emotional_state']:
                    return True
                
                # Large change in emotional intensity
                if abs(prev['emotional_intensity'] - features['emotional_intensity']) > 0.3:
                    return True
        
        # Check for motion pauses or significant changes
        if len(self.current_sequence) >= 3:
            recent_frames = self.current_sequence[-3:]
            
            # Extract motion data from recent frames to check for pauses
            motion_values = []
            for frame in recent_frames:
                # Collect velocities
                velocities = [v for k, v in frame.items() if k.endswith('_velocity') 
                              and isinstance(v, (list, tuple, np.ndarray))]
                if velocities:
                    # Calculate total motion magnitude
                    try:
                        motion = sum(np.linalg.norm(np.array(v)) for v in velocities)
                        motion_values.append(motion)
                    except:
                        pass
            
            # If we have motion values and they're all very small, it's a pause
            if motion_values and all(m < 0.1 for m in motion_values):
                return True
        
        return False
    
    def _process_current_sequence(self) -> None:
        """
        Process and potentially learn from the current motion sequence.
        """
        if not self.current_sequence:
            return
            
        # Only process if sequence is long enough
        if len(self.current_sequence) >= self.config.min_pattern_length:
            # Determine the type of sequence
            seq_type = self._classify_sequence_type(self.current_sequence)
            
            # Extract a pattern from the sequence
            pattern = self._extract_pattern(self.current_sequence)
            
            # Store the observed sequence
            self.observed_sequences.append({
                'type': seq_type,
                'pattern': pattern,
                'context': {
                    'speaking': self.current_sequence[0]['is_speaking'],
                    'emotional_state': self.current_sequence[0]['emotional_state'],
                    'emotional_intensity': self.current_sequence[0]['emotional_intensity']
                },
                'length': len(self.current_sequence),
                'timestamp': time.time()
            })
            
            # Learn from this pattern (update our model)
            self._learn_pattern(pattern, seq_type)
            
            self.has_learned_patterns = True
        
        # Reset current sequence
        self.current_sequence = []
    
    def _classify_sequence_type(self, sequence: List[Dict[str, Any]]) -> str:
        """
        Classify what type of motion pattern this sequence represents.
        
        Args:
            sequence: List of motion feature frames
            
        Returns:
            Classification of sequence type
        """
        # Simple heuristic classification based on motion characteristics
        # In a real implementation, this would use more sophisticated analysis
        
        # Check if person is speaking
        is_speaking = sequence[0]['is_speaking']
        
        # Extract motion features
        head_motion = sum(np.linalg.norm(np.array(frame.get('head_yaw', 0))) for frame in sequence)
        hand_motion = 0
        for frame in sequence:
            if 'left_hand_pos_velocity' in frame and 'right_hand_pos_velocity' in frame:
                try:
                    left_vel = np.linalg.norm(np.array(frame['left_hand_pos_velocity']))
                    right_vel = np.linalg.norm(np.array(frame['right_hand_pos_velocity']))
                    hand_motion += left_vel + right_vel
                except:
                    pass
        
        # Classify based on motion characteristics
        if hand_motion > head_motion * 3:
            if is_speaking:
                return 'emphasis' if hand_motion > 10 else 'hands'
            else:
                return 'hands'
        elif head_motion > hand_motion * 2:
            return 'head'
        elif hand_motion < 3 and head_motion < 3:
            return 'idle'
        else:
            return 'posture'
    
    def _extract_pattern(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract a representative pattern from a sequence.
        
        Args:
            sequence: List of motion feature frames
            
        Returns:
            Pattern representation
        """
        # In a real implementation, this would use more sophisticated techniques
        # like dimensionality reduction, clustering, or template extraction
        
        # Simple implementation: store key statistics and sample frames
        pattern = {
            'length': len(sequence),
            'sample_frames': [],
            'statistics': {},
            'context': {
                'speaking': sequence[0]['is_speaking'],
                'emotional_state': sequence[0]['emotional_state'],
                'emotional_intensity': sequence[0]['emotional_intensity']
            }
        }
        
        # Sample frames (beginning, middle, end)
        indices = [0, len(sequence)//2, len(sequence)-1]
        pattern['sample_frames'] = [sequence[i] for i in indices]
        
        # Calculate statistics for numeric features
        feature_stats = {}
        numeric_features = set()
        
        for frame in sequence:
            for key, value in frame.items():
                if isinstance(value, (int, float)) and not key.startswith('timestamp'):
                    numeric_features.add(key)
        
        for feature in numeric_features:
            values = [frame.get(feature, 0) for frame in sequence if feature in frame]
            if values:
                feature_stats[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        pattern['statistics'] = feature_stats
        
        return pattern
    
    def _learn_pattern(self, pattern: Dict[str, Any], pattern_type: str) -> None:
        """
        Learn from an observed pattern.
        
        Args:
            pattern: The extracted pattern
            pattern_type: Type of pattern
        """
        # Add the pattern to our collection
        if pattern_type in self.motion_patterns:
            self.motion_patterns[pattern_type].append(pattern)
            
            # Limit the number of patterns we keep
            max_patterns = 20  # Keep the most recent patterns
            if len(self.motion_patterns[pattern_type]) > max_patterns:
                self.motion_patterns[pattern_type] = self.motion_patterns[pattern_type][-max_patterns:]
        
        # Update personality traits based on observed patterns
        self._update_personality_traits(pattern, pattern_type)
    
    def _update_personality_traits(self, pattern: Dict[str, Any], pattern_type: str) -> None:
        """
        Update personality trait model based on observed pattern.
        
        Args:
            pattern: The observed pattern
            pattern_type: Type of the pattern
        """
        # Learning rate for personality updates
        lr = self.config.learning_rate * 0.5  # Slower changes to personality
        
        # Update expressiveness based on gesture frequency and variety
        if pattern_type in ['hands', 'emphasis']:
            self.personality_traits['expressiveness'] = (1 - lr) * self.personality_traits['expressiveness'] + lr * 0.7
        elif pattern_type == 'idle':
            self.personality_traits['expressiveness'] = (1 - lr) * self.personality_traits['expressiveness'] + lr * 0.3
            
        # Update energy based on motion magnitude
        energy_estimate = 0.5  # Default
        if 'statistics' in pattern:
            stats = pattern['statistics']
            velocity_keys = [k for k in stats if k.endswith('_velocity')]
            if velocity_keys:
                # Calculate average velocity across all motion components
                avg_velocities = [stats[k]['mean'] for k in velocity_keys]
                if avg_velocities:
                    energy_estimate = min(1.0, sum(avg_velocities) / len(avg_velocities) / 5.0)
        
        self.personality_traits['energy'] = (1 - lr) * self.personality_traits['energy'] + lr * energy_estimate
        
        # Other personality traits would be updated similarly in a full implementation
    
    def _select_gesture_type(self, context: Dict[str, Any]) -> str:
        """
        Select appropriate gesture type based on context.
        
        Args:
            context: Current context information
            
        Returns:
            Selected gesture type
        """
        # Probability distribution for gesture types based on context
        probabilities = {
            'head': 0.2,
            'hands': 0.2,
            'posture': 0.1,
            'emphasis': 0.1,
            'idle': 0.4
        }
        
        # Adjust based on context
        if context.get('speaking', False):
            # When speaking, more emphasis and hand gestures
            probabilities['emphasis'] = 0.3
            probabilities['hands'] = 0.3
            probabilities['head'] = 0.2
            probabilities['idle'] = 0.1
            probabilities['posture'] = 0.1
        else:
            # When not speaking, more idle and reactive head movements
            probabilities['idle'] = 0.4
            probabilities['head'] = 0.3
            probabilities['hands'] = 0.1
            probabilities['posture'] = 0.15
            probabilities['emphasis'] = 0.05
        
        # Adjust based on emotional state
        emotional_state = context.get('emotional_state', 'neutral')
        intensity = context.get('emotional_intensity', 0.0)
        
        if emotional_state in ['happy', 'excited'] and intensity > 0.5:
            # More energetic when happy/excited
            probabilities['emphasis'] += 0.1
            probabilities['hands'] += 0.1
            probabilities['idle'] -= 0.2
        elif emotional_state in ['sad', 'tired'] and intensity > 0.5:
            # More subdued when sad/tired
            probabilities['idle'] += 0.2
            probabilities['emphasis'] -= 0.1
            probabilities['hands'] -= 0.1
        
        # Adjust based on personality
        expressiveness = self.personality_traits['expressiveness']
        energy = self.personality_traits['energy']
        
        # More expressive people use more gestures
        expr_factor = (expressiveness - 0.5) * 0.4
        probabilities['hands'] += expr_factor
        probabilities['emphasis'] += expr_factor
        probabilities['idle'] -= expr_factor * 2
        
        # More energetic people have fewer idle moments
        energy_factor = (energy - 0.5) * 0.4
        probabilities['idle'] -= energy_factor
        probabilities['posture'] += energy_factor
        
        # Ensure valid probabilities
        for key in probabilities:
            probabilities[key] = max(0.01, min(0.99, probabilities[key]))
        
        # Normalize
        total = sum(probabilities.values())
        probabilities = {k: v/total for k, v in probabilities.items()}
        
        # Select based on probability
        gesture_types = list(probabilities.keys())
        weights = [probabilities[g] for g in gesture_types]
        
        return random.choices(gesture_types, weights=weights, k=1)[0]
    
    def _select_pattern_for_context(self, patterns: List[Dict[str, Any]], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the most appropriate pattern for the current context.
        
        Args:
            patterns: List of available patterns
            context: Current context
            
        Returns:
            Selected pattern
        """
        if not patterns:
            # If no patterns available, return empty pattern
            return {'sample_frames': [], 'statistics': {}}
        
        # Score each pattern based on context match
        scored_patterns = []
        
        for pattern in patterns:
            score = 0.0
            
            # Context matching
            pattern_context = pattern.get('context', {})
            
            # Speaking state match
            if pattern_context.get('speaking') == context.get('speaking'):
                score += 0.3
            
            # Emotional state match
            if pattern_context.get('emotional_state') == context.get('emotional_state'):
                score += 0.3
            
            # Emotional intensity match
            intensity_diff = abs(pattern_context.get('emotional_intensity', 0) - 
                                context.get('emotional_intensity', 0))
            score += 0.2 * (1.0 - intensity_diff)
            
            # Add some randomness for variety
            score += random.random() * 0.3
            
            scored_patterns.append((score, pattern))
        
        # Sort by score
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        
        # Return the highest scoring pattern
        return scored_patterns[0][1]
    
    def _add_variation_to_pattern(self, pattern: Dict[str, Any], 
                                 variation_level: float) -> Dict[str, Any]:
        """
        Add natural variation to a pattern to avoid repetition.
        
        Args:
            pattern: Original pattern
            variation_level: Level of variation to add (0-1)
            
        Returns:
            Modified pattern
        """
        # Copy pattern to avoid modifying original
        modified = pattern.copy()
        
        # Add variation to statistics
        if 'statistics' in modified:
            stats = modified['statistics']
            for feature, values in stats.items():
                if 'std' in values and values['std'] > 0:
                    # Add random variation within standard deviation
                    variation = random.normalvariate(0, values['std'] * variation_level)
                    values['mean'] += variation
                    
                    # Stay within observed min/max
                    values['mean'] = max(values['min'], min(values['max'], values['mean']))
        
        # Adjust sample frames slightly
        if 'sample_frames' in modified:
            for frame in modified['sample_frames']:
                for key, value in frame.items():
                    if isinstance(value, (int, float)) and not key.startswith('timestamp'):
                        # Add small random variation
                        frame[key] = value * (1.0 + random.uniform(-0.1, 0.1) * variation_level)
        
        return modified
    
    def _scale_pattern_to_duration(self, pattern: Dict[str, Any], 
                                  duration_sec: float) -> Dict[str, Any]:
        """
        Scale a pattern to the requested duration.
        
        Args:
            pattern: Motion pattern
            duration_sec: Target duration in seconds
            
        Returns:
            Scaled pattern
        """
        # Copy to avoid modifying original
        scaled = pattern.copy()
        
        # Calculate scaling factor
        original_length = pattern.get('length', 60)  # Frames, assume 30fps
        original_duration = original_length / 30.0  # seconds
        
        scaling_factor = duration_sec / original_duration
        
        # Scale length
        scaled['length'] = int(original_length * scaling_factor)
        
        # Scale timing-related statistics if present
        if 'statistics' in scaled:
            for feature, values in scaled['statistics'].items():
                if feature.endswith('_velocity'):
                    # Adjust velocities based on time scaling
                    values['mean'] = values['mean'] * (1.0 / scaling_factor)
                    values['min'] = values['min'] * (1.0 / scaling_factor)
                    values['max'] = values['max'] * (1.0 / scaling_factor)
        
        return scaled
    
    def _format_for_animation(self, pattern: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format pattern data for the animation system.
        
        Args:
            pattern: Processed motion pattern
            context: Current context
            
        Returns:
            Animation-ready data
        """
        # Create animation data structure
        animation_data = {
            'type': 'gesture',
            'duration': pattern.get('length', 60) / 30.0,  # seconds
            'keyframes': [],
            'context': context
        }
        
        # If we have sample frames, use them as keyframes
        if 'sample_frames' in pattern and pattern['sample_frames']:
            frames = pattern['sample_frames']
            
            # Create keyframes at different points in the animation
            num_frames = len(frames)
            for i, frame in enumerate(frames):
                # Position in animation (0-1)
                position = i / (num_frames - 1) if num_frames > 1 else 0
                
                # Extract animation values from frame
                keyframe = {
                    'position': position,
                    'values': {}
                }
                
                # Copy relevant values
                for key, value in frame.items():
                    if not key.startswith('timestamp') and not key.endswith('_velocity'):
                        if isinstance(value, (int, float, list, tuple, np.ndarray)):
                            keyframe['values'][key] = value
                
                animation_data['keyframes'].append(keyframe)
        
        # If we don't have sample frames but have statistics, create procedural keyframes
        elif 'statistics' in pattern:
            stats = pattern['statistics']
            
            # Create evenly spaced keyframes
            num_keyframes = 5
            for i in range(num_keyframes):
                position = i / (num_keyframes - 1) if num_keyframes > 1 else 0
                
                keyframe = {
                    'position': position,
                    'values': {}
                }
                
                # Generate values based on statistics
                for feature, values in stats.items():
                    if not feature.endswith('_velocity'):
                        # Generate smooth curve values for feature
                        mean = values.get('mean', 0)
                        amplitude = (values.get('max', mean) - values.get('min', mean)) / 2.0
                        
                        if amplitude > 0:
                            # Create a smooth curve through feature range
                            value = mean + amplitude * math.sin(position * 2 * math.pi)
                            keyframe['values'][feature] = value
                
                animation_data['keyframes'].append(keyframe)
        
        return animation_data
    
    def _generate_default_gesture(self, duration_sec: float, 
                                 gesture_type: str = None) -> Dict[str, Any]:
        """
        Generate a simple default gesture when no learned patterns are available.
        
        Args:
            duration_sec: Duration in seconds
            gesture_type: Type of gesture
            
        Returns:
            Default animation data
        """
        if gesture_type is None:
            gesture_type = random.choice(['idle', 'head', 'hands'])
            
        animation_data = {
            'type': 'gesture',
            'duration': duration_sec,
            'keyframes': []
        }
        
        # Number of keyframes for the animation
        num_keyframes = 5
        
        # Generate keyframes based on gesture type
        for i in range(num_keyframes):
            position = i / (num_keyframes - 1) if num_keyframes > 1 else 0
            
            keyframe = {
                'position': position,
                'values': {}
            }
            
            if gesture_type == 'head':
                # Simple head movement
                amplitude = 0.3
                keyframe['values']['head_yaw'] = amplitude * math.sin(position * 2 * math.pi)
                keyframe['values']['head_pitch'] = amplitude * 0.5 * math.sin(position * 3 * math.pi)
                
            elif gesture_type == 'hands':
                # Simple hand gesture
                amplitude = 0.4
                keyframe['values']['left_hand_pos'] = [
                    0.2 + amplitude * math.sin(position * 2 * math.pi),
                    0.1 * math.sin(position * 3 * math.pi),
                    0
                ]
                keyframe['values']['right_hand_pos'] = [
                    -0.2 - amplitude * math.sin(position * 2 * math.pi + 0.5),
                    0.1 * math.sin(position * 3 * math.pi + 1.0),
                    0
                ]
                
            elif gesture_type == 'emphasis':
                # Emphasis gesture
                progress = abs(position - 0.5) * 2  # 0->1->0
                keyframe['values']['right_hand_pos'] = [
                    0.3 * (1 - progress),
                    0.2 * (1 - progress),
                    0
                ]
                keyframe['values']['head_yaw'] = 0.2 * (1 - progress)
                
            elif gesture_type == 'posture':
                # Posture shift
                progress = min(1.0, position * 2)  # 0->1
                keyframe['values']['spine_angle'] = 0.1 * progress
                keyframe['values']['shoulder_width'] = 1.0 + 0.05 * progress
                
            else:  # idle
                # Subtle idle motion
                amplitude = 0.1
                phase = position * 2 * math.pi
                keyframe['values']['head_yaw'] = amplitude * 0.3 * math.sin(phase)
                keyframe['values']['head_pitch'] = amplitude * 0.2 * math.sin(phase * 0.7)
                keyframe['values']['spine_angle'] = amplitude * 0.1 * math.sin(phase * 0.5)
            
            animation_data['keyframes'].append(keyframe)
            
        return animation_data
    
    def save_model(self, path: str) -> bool:
        """
        Save learned patterns and personality model to file.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare data to save
            data = {
                'personality_traits': self.personality_traits,
                'motion_patterns': {},
                'observation_frames': self.observation_frames,
                'has_learned_patterns': self.has_learned_patterns
            }
            
            # Convert motion patterns to JSON-compatible format
            for gesture_type, patterns in self.motion_patterns.items():
                # Convert each pattern
                json_patterns = []
                for pattern in patterns:
                    json_pattern = pattern.copy()
                    
                    # Convert numpy arrays to lists
                    for frame in json_pattern.get('sample_frames', []):
                        for key, value in list(frame.items()):
                            if isinstance(value, np.ndarray):
                                frame[key] = value.tolist()
                    
                    json_patterns.append(json_pattern)
                
                data['motion_patterns'][gesture_type] = json_patterns
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Model saved to {path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load learned patterns and personality model from file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful
        """
        try:
            if not os.path.exists(path):
                print(f"Model file does not exist: {path}")
                return False
                
            # Load from file
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Load personality traits
            if 'personality_traits' in data:
                self.personality_traits = data['personality_traits']
            
            # Load motion patterns
            if 'motion_patterns' in data:
                self.motion_patterns = data['motion_patterns']
            
            # Load observation count
            if 'observation_frames' in data:
                self.observation_frames = data['observation_frames']
            
            # Load learning status
            if 'has_learned_patterns' in data:
                self.has_learned_patterns = data['has_learned_patterns']
                
            print(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def reset(self) -> None:
        """
        Reset the model, clearing all learned patterns.
        """
        self.motion_patterns = {k: [] for k in self.motion_patterns.keys()}
        self.personality_traits = {k: 0.5 for k in self.personality_traits.keys()}
        self.observation_frames = 0
        self.observation_history = []
        self.current_sequence = []
        self.observed_sequences = []
        self.has_learned_patterns = False
        
        print("Gesture & Mannerism Learning Model reset to initial state") 