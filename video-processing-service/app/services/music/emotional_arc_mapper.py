"""
Emotional Arc Mapper Module

This module provides functionality for mapping emotional arcs to video content,
analyzing how emotions evolve over time and generating appropriate music cues.
"""

import os
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from app.services.music.mood_analyzer import MoodAnalyzer

logger = logging.getLogger(__name__)

class EmotionalArcMapper:
    """
    Maps the emotional arc of video content over time.
    
    This class analyzes how emotions evolve throughout a video, creating a timeline
    of emotional shifts and key moments that can be used for synchronized music selection.
    """
    
    # Common emotional arc patterns in storytelling
    ARC_PATTERNS = {
        "rising": "Continuously increasing emotional intensity",
        "falling": "Continuously decreasing emotional intensity",
        "arc": "Rise followed by fall (classic narrative arc)",
        "inverse_arc": "Fall followed by rise (redemption arc)",
        "plateau": "Rise to sustained level",
        "valley": "Fall to sustained low level",
        "oscillating": "Alternating highs and lows (complex narrative)",
        "steady": "Minimal emotional change (informational content)"
    }
    
    # Emotional transition types
    TRANSITION_TYPES = {
        "gradual": "Smooth transition between emotional states",
        "sudden": "Abrupt change in emotional state",
        "contrast": "Shift to opposite emotional state",
        "intensification": "Strengthening of the same emotion",
        "diminution": "Weakening of the same emotion"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the emotional arc mapper.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Set default parameters
        self.segment_duration = self.config.get('segment_duration', 5)  # seconds
        self.min_segment_duration = self.config.get('min_segment_duration', 2)  # seconds
        self.smoothing_window = self.config.get('smoothing_window', 3)  # segments
        self.key_moment_threshold = self.config.get('key_moment_threshold', 0.3)  # minimum change to consider key moment
        
        # Initialize the mood analyzer for initial mood analysis
        self.mood_analyzer = MoodAnalyzer(self.config)
    
    def map_emotional_arc(
        self,
        video_path: str,
        transcript: Optional[List[Dict[str, Any]]] = None,
        mood_analysis: Optional[Dict[str, Any]] = None,
        segment_duration: Optional[int] = None,
        detect_key_moments: bool = True,
        smooth_arc: bool = True
    ) -> Dict[str, Any]:
        """
        Map the emotional arc of video content.
        
        Args:
            video_path: Path to the video file
            transcript: Optional transcript data (list of segments)
            mood_analysis: Optional pre-computed mood analysis
            segment_duration: Duration of segments in seconds (overrides config)
            detect_key_moments: Whether to detect key emotional moments
            smooth_arc: Whether to apply smoothing to the emotional arc
            
        Returns:
            Dictionary with emotional arc mapping results
        """
        results = {
            "status": "success",
            "input_path": video_path,
            "emotional_arc": []
        }
        
        # Use provided segment duration or default
        seg_duration = segment_duration or self.segment_duration
        
        # Get mood analysis if not provided
        if not mood_analysis:
            try:
                mood_analysis = self.mood_analyzer.analyze_mood(
                    video_path=video_path,
                    transcript=transcript
                )
            except Exception as e:
                logger.error(f"Error in mood analysis: {str(e)}")
                results["status"] = "error"
                results["error"] = f"Mood analysis failed: {str(e)}"
                return results
        
        # Extract overall mood
        if "overall_mood" in mood_analysis:
            results["overall_mood"] = mood_analysis["overall_mood"]
        
        # Extract or generate timeline
        timeline = []
        if "mood_timeline" in mood_analysis and mood_analysis["mood_timeline"]:
            timeline = mood_analysis["mood_timeline"]
        elif "timeline" in mood_analysis and mood_analysis["timeline"]:
            timeline = mood_analysis["timeline"]
        
        # If no timeline, try to extract from components
        if not timeline:
            for component in ["transcript_mood", "audio_mood", "visual_mood"]:
                if component in mood_analysis and "segment_moods" in mood_analysis[component]:
                    timeline = mood_analysis[component]["segment_moods"]
                    break
        
        # If still no timeline, return error
        if not timeline:
            results["status"] = "error"
            results["error"] = "No mood timeline available in the analysis"
            return results
        
        # Sort timeline by start time
        timeline = sorted(timeline, key=lambda x: x.get("start", 0))
        
        # Process the timeline to create emotional arc
        arc, segment_properties = self._process_timeline(timeline, smooth_arc)
        results["emotional_arc"] = arc
        
        # Calculate arc pattern and characteristics
        pattern_results = self._identify_arc_pattern(arc)
        results.update(pattern_results)
        
        # Detect key emotional moments if requested
        if detect_key_moments:
            key_moments = self._detect_key_moments(arc, self.key_moment_threshold)
            results["key_moments"] = key_moments
        
        # Generate music cue points
        music_cues = self._generate_music_cues(arc, key_moments if detect_key_moments else None)
        results["music_cues"] = music_cues
        
        # Overall story characteristics
        results["emotional_dynamics"] = self._calculate_emotional_dynamics(arc)
        
        return results
    
    def _process_timeline(
        self, 
        timeline: List[Dict[str, Any]],
        smooth_arc: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process the mood timeline to create an emotional arc.
        
        Args:
            timeline: List of mood segments
            smooth_arc: Whether to apply smoothing
            
        Returns:
            Processed emotional arc and segment properties
        """
        emotional_arc = []
        
        # Track key emotional properties across segments
        valence_values = []
        arousal_values = []
        segment_properties = {
            "mean_valence": 0,
            "mean_arousal": 0,
            "valence_range": (0, 0),
            "arousal_range": (0, 0),
            "valence_variance": 0,
            "arousal_variance": 0
        }
        
        # Process each segment
        for i, segment in enumerate(timeline):
            # Extract key values from segment
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            # If segment has valence and arousal, use them directly
            if "valence" in segment and "arousal" in segment:
                valence = segment["valence"]
                arousal = segment["arousal"]
            # If using different format
            elif "mood" in segment and isinstance(segment["mood"], dict):
                valence = segment["mood"].get("valence", 0)
                arousal = segment["mood"].get("arousal", 0)
            else:
                # Default to neutral values
                valence = 0
                arousal = 0.5
            
            # Get mood labels if available
            if "mood_labels" in segment:
                mood_labels = segment["mood_labels"]
            else:
                # Generate mood labels from valence-arousal
                mood_labels = self.mood_analyzer._map_values_to_mood_labels(valence, arousal)
            
            # Calculate emotional intensity (combination of absolute valence and arousal)
            emotional_intensity = (abs(valence) + arousal) / 2
            
            # Determine the primary emotion
            primary_mood = mood_labels[0] if mood_labels else "neutral"
            
            # Calculate changes from previous segment if not the first segment
            changes = {}
            if i > 0:
                prev = emotional_arc[i-1]
                changes = {
                    "valence_change": valence - prev["valence"],
                    "arousal_change": arousal - prev["arousal"],
                    "intensity_change": emotional_intensity - prev["emotional_intensity"]
                }
            
            # Create segment data
            segment_data = {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "valence": valence,
                "arousal": arousal,
                "emotional_intensity": emotional_intensity,
                "mood_labels": mood_labels,
                "primary_mood": primary_mood
            }
            
            # Add changes if available
            if changes:
                segment_data.update(changes)
            
            emotional_arc.append(segment_data)
            
            # Track values for statistics
            valence_values.append(valence)
            arousal_values.append(arousal)
        
        # Apply smoothing to the emotional arc if requested
        if smooth_arc and len(emotional_arc) > self.smoothing_window:
            emotional_arc = self._smooth_emotional_arc(emotional_arc, self.smoothing_window)
        
        # Calculate segment properties
        if valence_values and arousal_values:
            segment_properties.update({
                "mean_valence": np.mean(valence_values),
                "mean_arousal": np.mean(arousal_values),
                "valence_range": (min(valence_values), max(valence_values)),
                "arousal_range": (min(arousal_values), max(arousal_values)),
                "valence_variance": np.var(valence_values),
                "arousal_variance": np.var(arousal_values)
            })
        
        return emotional_arc, segment_properties
    
    def _smooth_emotional_arc(
        self, 
        arc: List[Dict[str, Any]], 
        window_size: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Apply smoothing to the emotional arc to reduce noise.
        
        Args:
            arc: Emotional arc data
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed emotional arc
        """
        smoothed_arc = []
        
        # Ensure window size is valid
        window_size = min(window_size, len(arc))
        half_window = window_size // 2
        
        for i in range(len(arc)):
            # Create a copy of the segment
            smoothed_segment = arc[i].copy()
            
            # Calculate window boundaries
            window_start = max(0, i - half_window)
            window_end = min(len(arc), i + half_window + 1)
            
            # Extract values from window
            window_valence = [arc[j]["valence"] for j in range(window_start, window_end)]
            window_arousal = [arc[j]["arousal"] for j in range(window_start, window_end)]
            window_intensity = [arc[j]["emotional_intensity"] for j in range(window_start, window_end)]
            
            # Calculate smoothed values
            smoothed_segment["valence"] = np.mean(window_valence)
            smoothed_segment["arousal"] = np.mean(window_arousal)
            smoothed_segment["emotional_intensity"] = np.mean(window_intensity)
            
            # Recalculate mood labels based on smoothed values
            smoothed_segment["mood_labels"] = self.mood_analyzer._map_values_to_mood_labels(
                smoothed_segment["valence"], 
                smoothed_segment["arousal"]
            )
            smoothed_segment["primary_mood"] = smoothed_segment["mood_labels"][0] if smoothed_segment["mood_labels"] else "neutral"
            
            # Calculate changes from previous segment
            if i > 0:
                prev = smoothed_arc[i-1]
                smoothed_segment.update({
                    "valence_change": smoothed_segment["valence"] - prev["valence"],
                    "arousal_change": smoothed_segment["arousal"] - prev["arousal"],
                    "intensity_change": smoothed_segment["emotional_intensity"] - prev["emotional_intensity"]
                })
            
            smoothed_arc.append(smoothed_segment)
        
        return smoothed_arc
    
    def _identify_arc_pattern(self, arc: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify the emotional arc pattern of the content.
        
        Args:
            arc: Emotional arc data
            
        Returns:
            Dictionary with arc pattern identification results
        """
        if not arc or len(arc) < 3:
            return {
                "arc_pattern": "unknown",
                "arc_confidence": 0,
                "pattern_description": "Insufficient data to determine arc pattern"
            }
        
        # Extract intensity values
        intensities = [segment["emotional_intensity"] for segment in arc]
        
        # Calculate trend using linear regression
        x = np.arange(len(intensities))
        slope, intercept = np.polyfit(x, intensities, 1)
        
        # Calculate key points
        start_intensity = intensities[0]
        end_intensity = intensities[-1]
        max_intensity = max(intensities)
        min_intensity = min(intensities)
        max_index = intensities.index(max_intensity)
        min_index = intensities.index(min_intensity)
        
        # Calculate normalized position of extremes
        max_position = max_index / len(intensities)
        min_position = min_index / len(intensities)
        
        # Intensity range
        intensity_range = max_intensity - min_intensity
        
        # Calculate smoothness of transitions
        changes = [abs(intensities[i] - intensities[i-1]) for i in range(1, len(intensities))]
        mean_change = np.mean(changes) if changes else 0
        max_change = max(changes) if changes else 0
        
        # Pattern confidence scores
        pattern_scores = {}
        
        # Rising pattern
        if slope > 0.01 and end_intensity > start_intensity:
            rising_score = 0.5 + 0.5 * min(1, slope * 10)
            pattern_scores["rising"] = rising_score
        
        # Falling pattern
        if slope < -0.01 and end_intensity < start_intensity:
            falling_score = 0.5 + 0.5 * min(1, abs(slope) * 10)
            pattern_scores["falling"] = falling_score
        
        # Arc pattern (rise then fall)
        if max_position > 0.2 and max_position < 0.8 and max_intensity > start_intensity and max_intensity > end_intensity:
            arc_score = 0.7 * (1 - abs(max_position - 0.5)) + 0.3 * min(1, intensity_range)
            pattern_scores["arc"] = arc_score
        
        # Inverse arc pattern (fall then rise)
        if min_position > 0.2 and min_position < 0.8 and min_intensity < start_intensity and min_intensity < end_intensity:
            inverse_arc_score = 0.7 * (1 - abs(min_position - 0.5)) + 0.3 * min(1, intensity_range)
            pattern_scores["inverse_arc"] = inverse_arc_score
        
        # Plateau pattern
        plateau_sections = self._find_plateaus(intensities)
        if plateau_sections and max(plateau_sections, key=len):
            longest_plateau = len(max(plateau_sections, key=len)) / len(intensities)
            if longest_plateau > 0.3:
                plateau_score = 0.6 + 0.4 * min(1, longest_plateau)
                pattern_scores["plateau"] = plateau_score
        
        # Valley pattern
        valley_sections = self._find_valleys(intensities)
        if valley_sections and max(valley_sections, key=len):
            longest_valley = len(max(valley_sections, key=len)) / len(intensities)
            if longest_valley > 0.3:
                valley_score = 0.6 + 0.4 * min(1, longest_valley)
                pattern_scores["valley"] = valley_score
        
        # Oscillating pattern
        turning_points = self._count_turning_points(intensities)
        if turning_points > 2:
            oscillating_score = 0.5 + 0.5 * min(1, turning_points / 8)
            pattern_scores["oscillating"] = oscillating_score
        
        # Steady pattern
        if intensity_range < 0.3 and mean_change < 0.1:
            steady_score = 0.7 + 0.3 * (1 - min(1, intensity_range * 3))
            pattern_scores["steady"] = steady_score
        
        # Determine the primary pattern
        primary_pattern = max(pattern_scores.items(), key=lambda x: x[1]) if pattern_scores else ("unknown", 0)
        
        # Format the results
        pattern_result = {
            "arc_pattern": primary_pattern[0],
            "arc_confidence": primary_pattern[1],
            "pattern_description": self.ARC_PATTERNS.get(primary_pattern[0], "Unknown pattern"),
            "pattern_scores": pattern_scores,
            "trend_analysis": {
                "slope": float(slope),
                "overall_direction": "rising" if slope > 0.01 else "falling" if slope < -0.01 else "stable",
                "intensity_range": intensity_range,
                "mean_change_rate": float(mean_change),
                "max_change_rate": float(max_change),
                "turning_points": turning_points
            }
        }
        
        return pattern_result
    
    def _detect_key_moments(
        self, 
        arc: List[Dict[str, Any]], 
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detect key emotional moments in the arc.
        
        Args:
            arc: Emotional arc data
            threshold: Minimum change to consider as key moment
            
        Returns:
            List of key emotional moments
        """
        key_moments = []
        
        if not arc or len(arc) < 3:
            return key_moments
        
        for i in range(1, len(arc) - 1):
            segment = arc[i]
            prev_segment = arc[i-1]
            next_segment = arc[i+1]
            
            # Look for intensity peaks and valleys
            is_peak = (segment["emotional_intensity"] > prev_segment["emotional_intensity"] and 
                      segment["emotional_intensity"] > next_segment["emotional_intensity"])
            is_valley = (segment["emotional_intensity"] < prev_segment["emotional_intensity"] and 
                         segment["emotional_intensity"] < next_segment["emotional_intensity"])
            
            # Look for significant emotional shifts
            intensity_change = abs(segment["emotional_intensity"] - prev_segment["emotional_intensity"])
            valence_change = abs(segment["valence"] - prev_segment["valence"])
            arousal_change = abs(segment["arousal"] - prev_segment["arousal"])
            
            # Calculate overall change
            overall_change = (intensity_change + valence_change + arousal_change) / 3
            
            # Detect mood shifts
            mood_shift = segment["primary_mood"] != prev_segment["primary_mood"]
            
            # Determine transition type
            transition_type = self._determine_transition_type(
                overall_change, segment, prev_segment, next_segment)
            
            # Check if this is a key moment
            if ((is_peak or is_valley) and overall_change > threshold) or (mood_shift and overall_change > threshold * 0.7):
                moment_type = "peak" if is_peak else "valley" if is_valley else "transition"
                
                key_moment = {
                    "time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "type": moment_type,
                    "moment_mood": segment["primary_mood"],
                    "transition_type": transition_type,
                    "emotional_intensity": segment["emotional_intensity"],
                    "valence": segment["valence"],
                    "arousal": segment["arousal"],
                    "change_magnitude": overall_change,
                    "description": f"{transition_type} transition to {segment['primary_mood']}"
                }
                
                key_moments.append(key_moment)
        
        # Also consider very beginning and end as potential key moments
        if arc:
            # Start
            start_segment = arc[0]
            key_moments.insert(0, {
                "time": start_segment["start_time"],
                "end_time": start_segment["end_time"],
                "type": "beginning",
                "moment_mood": start_segment["primary_mood"],
                "transition_type": "beginning",
                "emotional_intensity": start_segment["emotional_intensity"],
                "valence": start_segment["valence"],
                "arousal": start_segment["arousal"],
                "change_magnitude": 1.0,  # High by default as it's the start
                "description": f"Beginning with {start_segment['primary_mood']} mood"
            })
            
            # End
            end_segment = arc[-1]
            key_moments.append({
                "time": end_segment["start_time"],
                "end_time": end_segment["end_time"],
                "type": "ending",
                "moment_mood": end_segment["primary_mood"],
                "transition_type": "ending",
                "emotional_intensity": end_segment["emotional_intensity"],
                "valence": end_segment["valence"],
                "arousal": end_segment["arousal"],
                "change_magnitude": 1.0,  # High by default as it's the end
                "description": f"Ending with {end_segment['primary_mood']} mood"
            })
        
        return key_moments
    
    def _generate_music_cues(
        self, 
        arc: List[Dict[str, Any]],
        key_moments: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate music cue points based on the emotional arc.
        
        Args:
            arc: Emotional arc data
            key_moments: Detected key moments (optional)
            
        Returns:
            List of music cue points
        """
        music_cues = []
        
        if not arc:
            return music_cues
        
        # Calculate video duration
        if arc[-1]["end_time"] > 0:
            duration = arc[-1]["end_time"]
        else:
            duration = sum(segment["duration"] for segment in arc)
        
        # Start with beginning cue
        start_segment = arc[0]
        music_cues.append({
            "time": 0,
            "cue_type": "intro",
            "mood": start_segment["primary_mood"],
            "intensity": start_segment["emotional_intensity"],
            "valence": start_segment["valence"],
            "arousal": start_segment["arousal"],
            "description": f"Start with {start_segment['primary_mood']} music"
        })
        
        # Add cues from key moments if available
        if key_moments:
            for moment in key_moments:
                # Skip beginning and ending (already handled)
                if moment["type"] in ["beginning", "ending"]:
                    continue
                
                music_cues.append({
                    "time": moment["time"],
                    "cue_type": "transition",
                    "mood": moment["moment_mood"],
                    "intensity": moment["emotional_intensity"],
                    "valence": moment["valence"],
                    "arousal": moment["arousal"],
                    "description": f"Transition to {moment['moment_mood']} at key moment"
                })
        else:
            # If no key moments, add cues at points of significant change
            significant_changes = []
            
            for i in range(1, len(arc)):
                if "intensity_change" in arc[i] and abs(arc[i]["intensity_change"]) > 0.2:
                    significant_changes.append(i)
            
            # Also add evenly spaced cues if not enough significant changes
            if len(significant_changes) < 3:
                # Aim for approximately 3-5 cues
                target_cues = min(5, max(3, len(arc) // 5))
                cue_indices = [i * len(arc) // target_cues for i in range(1, target_cues)]
                
                # Combine with significant changes
                all_indices = sorted(set(significant_changes + cue_indices))
                
                for idx in all_indices:
                    if idx < len(arc):
                        segment = arc[idx]
                        music_cues.append({
                            "time": segment["start_time"],
                            "cue_type": "section",
                            "mood": segment["primary_mood"],
                            "intensity": segment["emotional_intensity"],
                            "valence": segment["valence"],
                            "arousal": segment["arousal"],
                            "description": f"Section change to {segment['primary_mood']}"
                        })
        
        # Add ending cue
        end_segment = arc[-1]
        music_cues.append({
            "time": duration - min(30, duration * 0.1),  # Start ending 10% before end or 30 seconds, whichever is smaller
            "cue_type": "outro",
            "mood": end_segment["primary_mood"],
            "intensity": end_segment["emotional_intensity"],
            "valence": end_segment["valence"],
            "arousal": end_segment["arousal"],
            "description": f"End with {end_segment['primary_mood']} music"
        })
        
        # Sort cues by time
        music_cues = sorted(music_cues, key=lambda x: x["time"])
        
        return music_cues
    
    def _calculate_emotional_dynamics(self, arc: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall emotional dynamics of the content.
        
        Args:
            arc: Emotional arc data
            
        Returns:
            Dictionary with emotional dynamics metrics
        """
        if not arc:
            return {
                "emotional_range": 0,
                "emotional_variability": 0,
                "emotional_complexity": "unknown"
            }
        
        # Extract intensity values
        intensities = [segment["emotional_intensity"] for segment in arc]
        valences = [segment["valence"] for segment in arc]
        arousals = [segment["arousal"] for segment in arc]
        
        # Calculate ranges
        intensity_range = max(intensities) - min(intensities)
        valence_range = max(valences) - min(valences)
        arousal_range = max(arousals) - min(arousals)
        
        # Calculate variability
        intensity_var = np.var(intensities) if len(intensities) > 1 else 0
        valence_var = np.var(valences) if len(valences) > 1 else 0
        arousal_var = np.var(arousals) if len(arousals) > 1 else 0
        
        # Combined emotional variability score
        emotional_variability = (intensity_var + valence_var + arousal_var) / 3
        
        # Count distinct moods
        mood_counts = {}
        for segment in arc:
            mood = segment["primary_mood"]
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        # Mood diversity
        mood_diversity = len(mood_counts)
        
        # Determine emotional complexity
        complexity_score = emotional_variability * 3 + mood_diversity * 0.2
        
        if complexity_score < 0.1:
            complexity = "uniform"
        elif complexity_score < 0.3:
            complexity = "simple"
        elif complexity_score < 0.6:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        return {
            "emotional_range": {
                "intensity_range": float(intensity_range),
                "valence_range": float(valence_range),
                "arousal_range": float(arousal_range)
            },
            "emotional_variability": float(emotional_variability),
            "mood_diversity": mood_diversity,
            "distinct_moods": list(mood_counts.keys()),
            "emotional_complexity": complexity
        }
    
    def _find_plateaus(self, values: List[float], threshold: float = 0.1) -> List[List[int]]:
        """Find plateaus (relatively stable high values) in the data."""
        if not values:
            return []
        
        plateaus = []
        current_plateau = []
        max_value = max(values)
        plateau_threshold = max_value - (max_value - min(values)) * threshold
        
        for i, value in enumerate(values):
            if value >= plateau_threshold:
                current_plateau.append(i)
            else:
                if len(current_plateau) > 2:  # Require at least 3 points for a plateau
                    plateaus.append(current_plateau)
                current_plateau = []
        
        # Check for plateau at the end
        if len(current_plateau) > 2:
            plateaus.append(current_plateau)
        
        return plateaus
    
    def _find_valleys(self, values: List[float], threshold: float = 0.1) -> List[List[int]]:
        """Find valleys (relatively stable low values) in the data."""
        if not values:
            return []
        
        valleys = []
        current_valley = []
        min_value = min(values)
        valley_threshold = min_value + (max(values) - min_value) * threshold
        
        for i, value in enumerate(values):
            if value <= valley_threshold:
                current_valley.append(i)
            else:
                if len(current_valley) > 2:  # Require at least 3 points for a valley
                    valleys.append(current_valley)
                current_valley = []
        
        # Check for valley at the end
        if len(current_valley) > 2:
            valleys.append(current_valley)
        
        return valleys
    
    def _count_turning_points(self, values: List[float]) -> int:
        """Count the number of turning points (peaks and valleys) in the data."""
        if len(values) < 3:
            return 0
        
        turning_points = 0
        for i in range(1, len(values) - 1):
            # Check for peak
            if values[i] > values[i-1] and values[i] > values[i+1]:
                turning_points += 1
            # Check for valley
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                turning_points += 1
        
        return turning_points
    
    def _determine_transition_type(
        self, 
        change_magnitude: float,
        current: Dict[str, Any],
        previous: Dict[str, Any],
        next_segment: Dict[str, Any]
    ) -> str:
        """Determine the type of emotional transition."""
        # Check for sudden changes
        if change_magnitude > 0.4:
            transition = "sudden"
        else:
            transition = "gradual"
        
        # Check for contrast (valence sign change)
        if (current["valence"] * previous["valence"] < 0 and 
            abs(current["valence"]) > 0.3 and abs(previous["valence"]) > 0.3):
            transition = "contrast"
        
        # Check for intensification (same direction, stronger)
        elif (abs(current["valence"]) > abs(previous["valence"]) and 
              np.sign(current["valence"]) == np.sign(previous["valence"]) and
              current["arousal"] > previous["arousal"]):
            transition = "intensification"
        
        # Check for diminution (same direction, weaker)
        elif (abs(current["valence"]) < abs(previous["valence"]) and 
              np.sign(current["valence"]) == np.sign(previous["valence"]) and
              current["arousal"] < previous["arousal"]):
            transition = "diminution"
        
        return transition 