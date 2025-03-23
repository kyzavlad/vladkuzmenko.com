#!/usr/bin/env python3
"""
Temporal Alignment Optimization Module

This module provides functionality to optimize the temporal alignment between
audio speech and visual lip movements, ensuring natural-looking lip synchronization
for translated video content.

Key features:
- Dynamic time warping for optimal alignment
- Speech rate adjustment for different languages
- Jitter reduction in mouth movements
- Natural transition smoothing
- Emphasis on key visemes for improved perception
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field


@dataclass
class AlignmentConfig:
    """Configuration for temporal alignment optimization."""
    smoothing_window: int = 3  # Window size for smoothing transitions
    emphasis_factor: float = 1.2  # Emphasis factor for key visemes
    min_viseme_duration: float = 0.05  # Minimum duration for a viseme in seconds
    max_viseme_duration: float = 0.5  # Maximum duration for a viseme in seconds
    target_frame_rate: float = 30.0  # Target frame rate for animation
    jitter_threshold: float = 0.03  # Threshold for jitter reduction (seconds)
    transition_overlap: float = 0.02  # Overlap time for transitions (seconds)


class TemporalAlignmentOptimizer:
    """
    Optimizes the temporal alignment between audio speech and visual lip movements,
    ensuring natural-looking lip synchronization.
    """
    
    def __init__(self, config: AlignmentConfig = None):
        """
        Initialize the temporal alignment optimizer.
        
        Args:
            config: Configuration for temporal alignment optimization
        """
        self.config = config or AlignmentConfig()
        self.frame_duration = 1.0 / self.config.target_frame_rate
        
        print(f"Temporal Alignment Optimizer initialized")
        print(f"  - Target frame rate: {self.config.target_frame_rate} fps")
        print(f"  - Frame duration: {self.frame_duration:.4f} seconds")
        print(f"  - Min/max viseme duration: {self.config.min_viseme_duration:.2f}/{self.config.max_viseme_duration:.2f} s")
    
    def optimize_timing(self, visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize the timing of visemes for natural speech.
        
        Args:
            visemes: List of viseme dictionaries with timing info
            
        Returns:
            List of visemes with optimized timing
        """
        if not visemes:
            return []
        
        # Apply basic timing constraints
        constrained = self._apply_timing_constraints(visemes)
        
        # Reduce jitter
        dejittered = self._reduce_jitter(constrained)
        
        # Smooth transitions
        smoothed = self._smooth_transitions(dejittered)
        
        # Emphasize key visemes (like consonants)
        emphasized = self._emphasize_key_visemes(smoothed)
        
        # Ensure frame alignment
        frame_aligned = self._align_to_frames(emphasized)
        
        return frame_aligned
    
    def _apply_timing_constraints(self, visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply basic timing constraints to visemes.
        
        Args:
            visemes: List of viseme dictionaries
            
        Returns:
            List of visemes with constrained timing
        """
        constrained = []
        
        for viseme in visemes:
            v = viseme.copy()
            start_time = v["start_time"]
            end_time = v["end_time"]
            duration = end_time - start_time
            
            # Apply minimum duration constraint
            if duration < self.config.min_viseme_duration:
                # Extend the duration while keeping the center point fixed
                center = (start_time + end_time) / 2
                half_duration = self.config.min_viseme_duration / 2
                v["start_time"] = center - half_duration
                v["end_time"] = center + half_duration
            
            # Apply maximum duration constraint
            elif duration > self.config.max_viseme_duration:
                # Shorten the duration while keeping the center point fixed
                center = (start_time + end_time) / 2
                half_duration = self.config.max_viseme_duration / 2
                v["start_time"] = center - half_duration
                v["end_time"] = center + half_duration
            
            constrained.append(v)
        
        return constrained
    
    def _reduce_jitter(self, visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reduce timing jitter between adjacent visemes.
        
        Args:
            visemes: List of viseme dictionaries
            
        Returns:
            List of visemes with reduced jitter
        """
        if len(visemes) < 3:
            return visemes
        
        dejittered = [visemes[0]]
        
        for i in range(1, len(visemes) - 1):
            prev = visemes[i - 1]
            curr = visemes[i].copy()
            next_v = visemes[i + 1]
            
            # Calculate time differences
            prev_diff = curr["start_time"] - prev["end_time"]
            next_diff = next_v["start_time"] - curr["end_time"]
            
            # Detect and correct jitter
            if abs(prev_diff - next_diff) < self.config.jitter_threshold:
                # Make both gaps equal
                avg_gap = (prev_diff + next_diff) / 2
                curr["start_time"] = prev["end_time"] + avg_gap
                curr["end_time"] = next_v["start_time"] - avg_gap
            
            dejittered.append(curr)
        
        dejittered.append(visemes[-1])
        return dejittered
    
    def _smooth_transitions(self, visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Smooth transitions between visemes by adding overlap.
        
        Args:
            visemes: List of viseme dictionaries
            
        Returns:
            List of visemes with smoothed transitions
        """
        if len(visemes) < 2:
            return visemes
        
        smoothed = []
        
        for i in range(len(visemes)):
            v = visemes[i].copy()
            
            # Add transition overlap with previous viseme
            if i > 0:
                prev = visemes[i - 1]
                # Ensure we don't create negative durations
                overlap = min(
                    self.config.transition_overlap,
                    (v["end_time"] - v["start_time"]) / 2,
                    (prev["end_time"] - prev["start_time"]) / 2
                )
                v["transition_from"] = {
                    "viseme": prev["viseme"],
                    "start_time": v["start_time"] - overlap,
                    "end_time": v["start_time"]
                }
                v["start_time"] -= overlap
            
            # Add transition overlap with next viseme
            if i < len(visemes) - 1:
                next_v = visemes[i + 1]
                # Ensure we don't create negative durations
                overlap = min(
                    self.config.transition_overlap,
                    (v["end_time"] - v["start_time"]) / 2,
                    (next_v["end_time"] - next_v["start_time"]) / 2
                )
                v["transition_to"] = {
                    "viseme": next_v["viseme"],
                    "start_time": v["end_time"],
                    "end_time": v["end_time"] + overlap
                }
                v["end_time"] += overlap
            
            smoothed.append(v)
        
        return smoothed
    
    def _emphasize_key_visemes(self, visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Emphasize key visemes (typically consonants) for better perception.
        
        Args:
            visemes: List of viseme dictionaries
            
        Returns:
            List of visemes with emphasized key visemes
        """
        # Key visemes that should be emphasized
        key_visemes = ["P", "B", "F", "V", "TH", "S", "SH", "L", "R"]
        
        emphasized = []
        for v in visemes:
            v_copy = v.copy()
            
            # If this is a key viseme, emphasize it by extending its duration
            if v["viseme"] in key_visemes:
                duration = v["end_time"] - v["start_time"]
                # Extend by the emphasis factor, but respect max duration
                new_duration = min(duration * self.config.emphasis_factor, self.config.max_viseme_duration)
                # Center the extended duration around the original
                center = (v["start_time"] + v["end_time"]) / 2
                half_duration = new_duration / 2
                v_copy["start_time"] = center - half_duration
                v_copy["end_time"] = center + half_duration
                v_copy["emphasized"] = True
            
            emphasized.append(v_copy)
        
        return emphasized
    
    def _align_to_frames(self, visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Align viseme timings to frame boundaries for smoother animation.
        
        Args:
            visemes: List of viseme dictionaries
            
        Returns:
            List of visemes aligned to frame boundaries
        """
        aligned = []
        
        for v in visemes:
            v_copy = v.copy()
            
            # Align start and end times to nearest frame
            start_frame = round(v["start_time"] / self.frame_duration)
            end_frame = round(v["end_time"] / self.frame_duration)
            
            # Ensure at least one frame difference
            if start_frame == end_frame:
                end_frame = start_frame + 1
            
            v_copy["start_time"] = start_frame * self.frame_duration
            v_copy["end_time"] = end_frame * self.frame_duration
            
            # Update frame information
            v_copy["start_frame"] = start_frame
            v_copy["end_frame"] = end_frame
            v_copy["frame_count"] = end_frame - start_frame
            
            aligned.append(v_copy)
        
        return aligned
    
    def dynamic_time_warping(self, 
                           source_visemes: List[Dict[str, Any]],
                           target_duration: float) -> List[Dict[str, Any]]:
        """
        Apply dynamic time warping to fit visemes into a target duration.
        
        Args:
            source_visemes: Original viseme sequence
            target_duration: Target duration in seconds
            
        Returns:
            Time-warped viseme sequence
        """
        if not source_visemes:
            return []
        
        # Calculate original duration
        original_duration = source_visemes[-1]["end_time"] - source_visemes[0]["start_time"]
        
        # Calculate warping factor
        warp_factor = target_duration / original_duration
        
        # Apply warping
        warped = []
        start_time = source_visemes[0]["start_time"]
        
        for v in source_visemes:
            v_copy = v.copy()
            
            # Warp the timing
            rel_start = (v["start_time"] - start_time) * warp_factor
            rel_end = (v["end_time"] - start_time) * warp_factor
            
            v_copy["start_time"] = rel_start
            v_copy["end_time"] = rel_end
            v_copy["original_start_time"] = v["start_time"]
            v_copy["original_end_time"] = v["end_time"]
            v_copy["warp_factor"] = warp_factor
            
            warped.append(v_copy)
        
        # Re-optimize the warped sequence
        optimized = self.optimize_timing(warped)
        
        return optimized
    
    def apply_speech_rate_adjustment(self, 
                                   visemes: List[Dict[str, Any]],
                                   source_lang: str,
                                   target_lang: str) -> List[Dict[str, Any]]:
        """
        Adjust viseme timing based on typical speech rates for different languages.
        
        Args:
            visemes: List of viseme dictionaries
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of visemes with adjusted timing for speech rate
        """
        # Average syllables per second for different languages
        speech_rates = {
            "en": 4.0,  # English: ~4 syllables per second
            "fr": 4.7,  # French: ~4.7 syllables per second
            "ja": 7.8,  # Japanese: ~7.8 syllables per second
            "es": 4.6,  # Spanish: ~4.6 syllables per second
            "de": 4.2,  # German: ~4.2 syllables per second
            "it": 5.0,  # Italian: ~5 syllables per second
            "ru": 4.4,  # Russian: ~4.4 syllables per second
            "zh": 5.8   # Mandarin: ~5.8 syllables per second
        }
        
        # Default to English if language not found
        source_rate = speech_rates.get(source_lang, speech_rates["en"])
        target_rate = speech_rates.get(target_lang, speech_rates["en"])
        
        # Calculate the ratio of target to source speech rate
        rate_ratio = target_rate / source_rate
        
        # Calculate the original duration and target duration
        if not visemes:
            return []
        
        original_duration = visemes[-1]["end_time"] - visemes[0]["start_time"]
        target_duration = original_duration / rate_ratio  # Slower languages need more time
        
        # Apply dynamic time warping with the calculated target duration
        return self.dynamic_time_warping(visemes, target_duration)


class VisemeBlender:
    """
    Handles blending between visemes for smooth transitions in animation.
    """
    
    def __init__(self, frame_rate: float = 30.0):
        """
        Initialize the viseme blender.
        
        Args:
            frame_rate: Animation frame rate in frames per second
        """
        self.frame_rate = frame_rate
        self.frame_duration = 1.0 / frame_rate
        
        print(f"Viseme Blender initialized at {frame_rate} fps")
    
    def generate_blend_weights(self, 
                             visemes: List[Dict[str, Any]], 
                             total_frames: int) -> List[Dict[str, Dict[str, float]]]:
        """
        Generate per-frame blend weights for all visemes.
        
        Args:
            visemes: List of viseme dictionaries with timing
            total_frames: Total number of frames to generate
            
        Returns:
            List of dictionaries mapping viseme IDs to blend weights for each frame
        """
        # Initialize empty blend weights for all frames
        blend_weights = [{} for _ in range(total_frames)]
        
        # Calculate weights for each viseme
        for viseme in visemes:
            viseme_id = viseme["viseme"]
            start_time = viseme["start_time"]
            end_time = viseme["end_time"]
            
            # Calculate which frames this viseme affects
            start_frame = max(0, int(start_time * self.frame_rate))
            end_frame = min(total_frames - 1, int(end_time * self.frame_rate))
            
            # Handle transitions
            if "transition_from" in viseme:
                trans_from = viseme["transition_from"]
                trans_from_id = trans_from["viseme"]
                trans_start_frame = max(0, int(trans_from["start_time"] * self.frame_rate))
                trans_end_frame = min(total_frames - 1, int(trans_from["end_time"] * self.frame_rate))
                
                # Add blending weights for the transition
                for frame in range(trans_start_frame, trans_end_frame + 1):
                    progress = (frame - trans_start_frame) / (trans_end_frame - trans_start_frame + 1)
                    # From viseme weight decreases, current viseme weight increases
                    from_weight = 1.0 - progress
                    to_weight = progress
                    
                    blend_weights[frame][trans_from_id] = from_weight
                    blend_weights[frame][viseme_id] = to_weight
            
            # Handle main viseme appearance (with full weight)
            main_start = start_frame if "transition_from" not in viseme else int(viseme["transition_from"]["end_time"] * self.frame_rate)
            main_end = end_frame if "transition_to" not in viseme else int(viseme["transition_to"]["start_time"] * self.frame_rate)
            
            for frame in range(main_start, main_end + 1):
                blend_weights[frame][viseme_id] = 1.0
            
            # Handle transition to next viseme
            if "transition_to" in viseme:
                trans_to = viseme["transition_to"]
                trans_to_id = trans_to["viseme"]
                trans_start_frame = max(0, int(trans_to["start_time"] * self.frame_rate))
                trans_end_frame = min(total_frames - 1, int(trans_to["end_time"] * self.frame_rate))
                
                # Add blending weights for the transition
                for frame in range(trans_start_frame, trans_end_frame + 1):
                    progress = (frame - trans_start_frame) / (trans_end_frame - trans_start_frame + 1)
                    # Current viseme weight decreases, to viseme weight increases
                    from_weight = 1.0 - progress
                    to_weight = progress
                    
                    blend_weights[frame][viseme_id] = from_weight
                    blend_weights[frame][trans_to_id] = to_weight
        
        return blend_weights
    
    def get_frame_viseme_weights(self, 
                               visemes: List[Dict[str, Any]], 
                               frame_time: float) -> Dict[str, float]:
        """
        Get the blend weights for visemes at a specific frame time.
        
        Args:
            visemes: List of viseme dictionaries with timing
            frame_time: Time of the current frame in seconds
            
        Returns:
            Dictionary mapping viseme IDs to blend weights
        """
        weights = {}
        
        # Find visemes that are active at this frame time
        for viseme in visemes:
            viseme_id = viseme["viseme"]
            
            # Check main viseme timing
            if viseme["start_time"] <= frame_time <= viseme["end_time"]:
                # Default to full weight
                weights[viseme_id] = 1.0
            
            # Check transition from previous viseme
            if "transition_from" in viseme:
                trans = viseme["transition_from"]
                if trans["start_time"] <= frame_time <= trans["end_time"]:
                    # Calculate blend weight based on position in transition
                    progress = (frame_time - trans["start_time"]) / (trans["end_time"] - trans["start_time"])
                    weights[trans["viseme"]] = 1.0 - progress
                    weights[viseme_id] = progress
            
            # Check transition to next viseme
            if "transition_to" in viseme:
                trans = viseme["transition_to"]
                if trans["start_time"] <= frame_time <= trans["end_time"]:
                    # Calculate blend weight based on position in transition
                    progress = (frame_time - trans["start_time"]) / (trans["end_time"] - trans["start_time"])
                    weights[viseme_id] = 1.0 - progress
                    weights[trans["viseme"]] = progress
        
        return weights


# Example usage
if __name__ == "__main__":
    # Create a temporal alignment optimizer
    config = AlignmentConfig(
        smoothing_window=3,
        emphasis_factor=1.2,
        min_viseme_duration=0.05,
        max_viseme_duration=0.5,
        target_frame_rate=30.0
    )
    
    optimizer = TemporalAlignmentOptimizer(config)
    
    # Example viseme sequence
    visemes = [
        {"viseme": "A", "start_time": 0.0, "end_time": 0.1},
        {"viseme": "B", "start_time": 0.1, "end_time": 0.15},  # Very short, will be extended
        {"viseme": "E", "start_time": 0.15, "end_time": 0.3},
        {"viseme": "O", "start_time": 0.3, "end_time": 0.9},   # Very long, will be shortened
        {"viseme": "S", "start_time": 0.9, "end_time": 1.1}
    ]
    
    # Optimize the viseme sequence
    optimized = optimizer.optimize_timing(visemes)
    
    print("Original visemes:")
    for v in visemes:
        duration = v["end_time"] - v["start_time"]
        print(f"  {v['viseme']}: {v['start_time']:.2f} - {v['end_time']:.2f} ({duration:.2f}s)")
    
    print("\nOptimized visemes:")
    for v in optimized:
        duration = v["end_time"] - v["start_time"]
        print(f"  {v['viseme']}: {v['start_time']:.2f} - {v['end_time']:.2f} ({duration:.2f}s)")
        if "emphasized" in v:
            print(f"    * Emphasized")
        if "transition_from" in v:
            trans = v["transition_from"]
            print(f"    * Transition from {trans['viseme']}: {trans['start_time']:.2f} - {trans['end_time']:.2f}")
        if "transition_to" in v:
            trans = v["transition_to"]
            print(f"    * Transition to {trans['viseme']}: {trans['start_time']:.2f} - {trans['end_time']:.2f}")
    
    # Apply speech rate adjustment (English to Japanese)
    adjusted = optimizer.apply_speech_rate_adjustment(visemes, "en", "ja")
    
    print("\nAdjusted for speech rate (English to Japanese):")
    for v in adjusted:
        duration = v["end_time"] - v["start_time"]
        print(f"  {v['viseme']}: {v['start_time']:.2f} - {v['end_time']:.2f} ({duration:.2f}s)")
    
    # Example of viseme blending
    blender = VisemeBlender(frame_rate=30.0)
    
    # Get blend weights for a specific frame
    frame_time = 0.25  # 0.25 seconds into the animation
    weights = blender.get_frame_viseme_weights(optimized, frame_time)
    
    print(f"\nBlend weights at {frame_time:.2f}s:")
    for viseme_id, weight in weights.items():
        print(f"  {viseme_id}: {weight:.2f}") 