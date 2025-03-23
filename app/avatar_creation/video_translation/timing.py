#!/usr/bin/env python3
"""
Script Timing Preservation Module

This module provides functionality to preserve timing information when translating scripts,
particularly for video dubbing and subtitling. It ensures that translated content
matches the timing of the original content as closely as possible, taking into account
the differences in language structure, word length, and speech rate across different languages.

Key features:
- Optimization of segment durations based on language characteristics
- Syllable counting and speech rate estimation
- Detection of potential timing issues
- Support for multiple languages with different timing profiles
"""

import re
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any


@dataclass
class TimedSegment:
    """Represents a segment of text with timing information."""
    text: str
    start_time: float
    end_time: float
    segment_id: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Calculate the duration of the segment in seconds."""
        return self.end_time - self.start_time
    
    @property
    def characters_per_second(self) -> float:
        """Calculate the characters per second for this segment."""
        if self.duration <= 0:
            return 0
        return len(self.text) / self.duration
    
    @property
    def word_count(self) -> int:
        """Count the number of words in the segment."""
        return len(re.findall(r'\b\w+\b', self.text))
    
    @property 
    def words_per_minute(self) -> float:
        """Calculate the words per minute for this segment."""
        if self.duration <= 0:
            return 0
        return (self.word_count / self.duration) * 60
    
    def adjust_end_time(self, new_duration: float) -> None:
        """Adjust the end time to match a new duration."""
        self.end_time = self.start_time + new_duration
    
    def adjust_start_time(self, new_start_time: float) -> None:
        """Adjust the start time while maintaining duration."""
        duration = self.duration
        self.start_time = new_start_time
        self.end_time = new_start_time + duration
    
    def split(self, at_time: float) -> Tuple['TimedSegment', 'TimedSegment']:
        """Split the segment at the specified time point."""
        if at_time <= self.start_time or at_time >= self.end_time:
            raise ValueError(f"Split time {at_time} must be between start {self.start_time} and end {self.end_time}")
        
        # Determine the character position to split at (proportional to time)
        time_ratio = (at_time - self.start_time) / self.duration
        char_position = int(len(self.text) * time_ratio)
        
        # Create the two new segments
        first_text = self.text[:char_position].strip()
        second_text = self.text[char_position:].strip()
        
        first_segment = TimedSegment(
            text=first_text,
            start_time=self.start_time,
            end_time=at_time,
            segment_id=f"{self.segment_id}_a" if self.segment_id else None
        )
        
        second_segment = TimedSegment(
            text=second_text,
            start_time=at_time,
            end_time=self.end_time,
            segment_id=f"{self.segment_id}_b" if self.segment_id else None
        )
        
        return first_segment, second_segment


@dataclass
class LanguageTimingProfile:
    """Holds timing profiles for different languages."""
    language_code: str
    # Average spoken characters per second
    avg_chars_per_second: float
    # Average spoken words per minute
    avg_words_per_minute: float
    # Syllable detection pattern (regex pattern)
    syllable_pattern: str
    # Average syllables per word
    avg_syllables_per_word: float
    # The optimal timing adjustment factor based on source to target
    timing_adjustment_factors: Dict[str, float] = field(default_factory=dict)
    # Maximum characters per second that are comfortably speakable
    max_comfortable_chars_per_second: float = 15.0
    # Additional language-specific properties
    properties: Dict[str, Any] = field(default_factory=dict)


class ScriptTimingPreserver:
    """
    Preserves timing information when translating scripts.
    
    This class provides methods for adjusting timing information based on
    language profiles, ensuring that translated content fits within the
    original timing constraints as closely as possible.
    """
    
    def __init__(self):
        """Initialize the script timing preserver."""
        self.language_profiles = self._load_language_profiles()
    
    def _load_language_profiles(self) -> Dict[str, LanguageTimingProfile]:
        """
        Load language timing profiles from a data source.
        
        In a real implementation, this would load from a database or file.
        Here we provide a simplified set of profiles for demonstration.
        """
        profiles = {}
        
        # English profile
        profiles["en"] = LanguageTimingProfile(
            language_code="en",
            avg_chars_per_second=13.0,
            avg_words_per_minute=150,
            syllable_pattern=r'[aeiouy]+(?:[^aeiouy](?!$))?',
            avg_syllables_per_word=1.5,
            timing_adjustment_factors={
                "en": 1.0,    # English to English (no change)
                "es": 1.1,    # English to Spanish (Spanish typically 10% longer)
                "fr": 1.15,   # English to French
                "de": 1.2,    # English to German (German typically 20% longer)
                "ja": 0.8,    # English to Japanese 
                "zh": 0.7,    # English to Chinese
                "ru": 0.9,    # English to Russian
                "ar": 0.85,   # English to Arabic
            },
            max_comfortable_chars_per_second=15.0,
            properties={
                "contraction_frequent": True,
                "abbreviation_frequent": True
            }
        )
        
        # Spanish profile
        profiles["es"] = LanguageTimingProfile(
            language_code="es",
            avg_chars_per_second=12.0,
            avg_words_per_minute=160,
            syllable_pattern=r'[aáeéiíoóuú]+(?:[^aáeéiíoóuú](?!$))?',
            avg_syllables_per_word=1.9,
            timing_adjustment_factors={
                "en": 0.9,    # Spanish to English
                "es": 1.0,    # Spanish to Spanish (no change)
                "fr": 1.05,   # Spanish to French
                "de": 1.1,    # Spanish to German
                "ja": 0.75,   # Spanish to Japanese
                "zh": 0.65,   # Spanish to Chinese
                "ru": 0.8,    # Spanish to Russian
                "ar": 0.75,   # Spanish to Arabic
            },
            max_comfortable_chars_per_second=14.0,
            properties={
                "contraction_frequent": False,
                "abbreviation_frequent": False
            }
        )
        
        # Add more language profiles as needed
        
        return profiles
    
    def adjust_timing(self, segment: TimedSegment, translation: str, 
                     source_lang: str, target_lang: str) -> TimedSegment:
        """
        Adjust timing for a translated segment based on language profiles.
        
        Args:
            segment: The original timed segment
            translation: The translated text
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            A new TimedSegment with adjusted timing
        """
        # Get language profiles
        source_profile = self.language_profiles.get(source_lang)
        target_profile = self.language_profiles.get(target_lang)
        
        if not source_profile or not target_profile:
            # Fallback if profiles not available - maintain original timing
            return TimedSegment(
                text=translation,
                start_time=segment.start_time,
                end_time=segment.end_time,
                segment_id=segment.segment_id
            )
        
        # Get the adjustment factor from source to target
        adjustment_factor = source_profile.timing_adjustment_factors.get(target_lang, 1.0)
        
        # Calculate syllable ratio if available
        source_syllables = self.count_syllables(segment.text, source_lang)
        target_syllables = self.count_syllables(translation, target_lang)
        
        if source_syllables > 0 and target_syllables > 0:
            syllable_ratio = target_syllables / source_syllables
            # Blend the adjustment factor with syllable ratio (giving more weight to syllable ratio)
            adjusted_factor = (adjustment_factor * 0.3) + (syllable_ratio * 0.7)
        else:
            adjusted_factor = adjustment_factor
        
        # Calculate new duration based on the adjusted factor
        new_duration = segment.duration * adjusted_factor
        
        # Create new segment with adjusted timing
        return TimedSegment(
            text=translation,
            start_time=segment.start_time,
            end_time=segment.start_time + new_duration,
            segment_id=segment.segment_id
        )
    
    def count_syllables(self, text: str, language_code: str) -> int:
        """
        Count the number of syllables in a text for a specific language.
        
        Args:
            text: The text to count syllables in
            language_code: The language code
            
        Returns:
            The number of syllables
        """
        profile = self.language_profiles.get(language_code)
        if not profile:
            # Fallback syllable counting for unknown languages
            # Count vowel sequences as a rough approximation
            return len(re.findall(r'[aeiouy]+', text.lower()))
        
        # Use language-specific syllable pattern
        return len(re.findall(profile.syllable_pattern, text.lower()))
    
    def optimize_segments(self, segments: List[TimedSegment], 
                         translations: List[str],
                         source_lang: str, target_lang: str) -> List[TimedSegment]:
        """
        Optimize a sequence of segments with their translations.
        
        This method ensures that:
        1. Each translated segment fits within reasonable timing constraints
        2. The overall timing pattern is preserved
        3. Segments don't overlap
        
        Args:
            segments: The original timed segments
            translations: The translated texts (in the same order as segments)
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            A list of optimized TimedSegments
        """
        if len(segments) != len(translations):
            raise ValueError("Number of segments and translations must match")
        
        # First pass: adjust each segment independently
        adjusted_segments = [
            self.adjust_timing(segment, translation, source_lang, target_lang)
            for segment, translation in zip(segments, translations)
        ]
        
        # Second pass: resolve overlaps and timing issues
        optimized_segments = self._resolve_timing_conflicts(adjusted_segments)
        
        # Final pass: check for speaking rate issues
        for i, segment in enumerate(optimized_segments):
            issues = self.check_timing_issues(segment, target_lang)
            if issues["has_issues"] and issues["too_fast"]:
                # Try to extend duration if the speaking rate is too fast
                if i < len(optimized_segments) - 1:
                    # Check if there's a gap to the next segment we can use
                    next_segment = optimized_segments[i + 1]
                    available_gap = next_segment.start_time - segment.end_time
                    if available_gap > 0:
                        needed_extra = issues["recommended_duration"] - segment.duration
                        extension = min(needed_extra, available_gap * 0.8)  # Use up to 80% of gap
                        segment.end_time += extension
        
        return optimized_segments
    
    def _resolve_timing_conflicts(self, segments: List[TimedSegment]) -> List[TimedSegment]:
        """
        Resolve timing conflicts between adjacent segments.
        
        Args:
            segments: The segments to process
            
        Returns:
            A list of segments with conflicts resolved
        """
        if not segments:
            return []
        
        result = [segments[0]]
        
        for i in range(1, len(segments)):
            current = segments[i]
            previous = result[-1]
            
            # Check for overlap
            if current.start_time < previous.end_time:
                # Find a compromise point
                overlap = previous.end_time - current.start_time
                
                # If overlap is small, just push current segment forward
                if overlap < 0.3:  # Less than 300ms overlap
                    current.adjust_start_time(previous.end_time)
                else:
                    # For larger overlaps, find a midpoint and adjust both segments
                    midpoint = (previous.end_time + current.start_time) / 2
                    previous.end_time = midpoint
                    current.adjust_start_time(midpoint)
            
            result.append(current)
        
        return result
    
    def check_timing_issues(self, segment: TimedSegment, language_code: str) -> Dict[str, Any]:
        """
        Check for potential timing issues in a segment.
        
        Args:
            segment: The segment to check
            language_code: The language code
            
        Returns:
            A dictionary with timing issue flags and data
        """
        profile = self.language_profiles.get(language_code)
        if not profile:
            # Can't check issues without a profile
            return {"has_issues": False}
        
        result = {
            "has_issues": False,
            "too_fast": False,
            "too_slow": False,
            "chars_per_second": segment.characters_per_second,
            "recommended_duration": segment.duration
        }
        
        # Check if speaking rate is too fast
        if segment.characters_per_second > profile.max_comfortable_chars_per_second:
            result["has_issues"] = True
            result["too_fast"] = True
            # Calculate recommended duration
            result["recommended_duration"] = len(segment.text) / profile.max_comfortable_chars_per_second
        
        # Check if speaking rate is too slow (optional, as slow speech is less problematic)
        if segment.characters_per_second < profile.avg_chars_per_second * 0.6:
            result["has_issues"] = True
            result["too_slow"] = True
        
        return result
    
    def process_script(self, script: Dict[str, Any], translations: Dict[str, Dict[str, str]], 
                      source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Process an entire script with timing information.
        
        Args:
            script: The original script with timing info
            translations: Dictionary mapping segment IDs to translations
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            A processed script with adjusted timing
        """
        # Create timed segments from script
        segments = []
        for segment_data in script.get("segments", []):
            segment = TimedSegment(
                text=segment_data.get("text", ""),
                start_time=segment_data.get("start_time", 0),
                end_time=segment_data.get("end_time", 0),
                segment_id=segment_data.get("id")
            )
            segments.append(segment)
        
        # Gather translations
        translation_list = []
        for segment in segments:
            if segment.segment_id and segment.segment_id in translations.get(target_lang, {}):
                translation_list.append(translations[target_lang][segment.segment_id])
            else:
                # Use original text if translation not available
                translation_list.append(segment.text)
        
        # Optimize segments
        optimized_segments = self.optimize_segments(
            segments, translation_list, source_lang, target_lang
        )
        
        # Create new script with optimized timing
        result = script.copy()
        result["segments"] = [
            {
                "id": segment.segment_id,
                "text": segment.text,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration
            }
            for segment in optimized_segments
        ]
        
        return result


# Example usage
if __name__ == "__main__":
    # Create sample segment and translation
    segment = TimedSegment(
        text="Welcome to our demonstration of the avatar creation system.",
        start_time=0.0,
        end_time=3.0,
        segment_id="intro_1"
    )
    
    # Spanish translation (typically longer than English)
    translation = "Bienvenido a nuestra demostración del sistema de creación de avatares."
    
    # Initialize timing preserver
    preserver = ScriptTimingPreserver()
    
    # Adjust timing for translated segment
    adjusted = preserver.adjust_timing(segment, translation, "en", "es")
    
    print(f"Original: {segment.text}")
    print(f"Duration: {segment.duration:.2f}s, CPS: {segment.characters_per_second:.2f}")
    
    print(f"\nTranslated: {adjusted.text}")
    print(f"Duration: {adjusted.duration:.2f}s, CPS: {adjusted.characters_per_second:.2f}")
    
    # Check for timing issues
    issues = preserver.check_timing_issues(adjusted, "es")
    print(f"\nTiming issues: {issues['has_issues']}")
    if issues["has_issues"]:
        if issues["too_fast"]:
            print(f"Speaking rate too fast: {issues['chars_per_second']:.2f} chars/sec")
            print(f"Recommended duration: {issues['recommended_duration']:.2f}s")
        if issues["too_slow"]:
            print(f"Speaking rate too slow")
    
    # Example of optimizing multiple segments
    segments = [
        TimedSegment(text="First segment", start_time=0.0, end_time=2.0, segment_id="1"),
        TimedSegment(text="Second segment", start_time=2.5, end_time=4.0, segment_id="2"),
        TimedSegment(text="Third segment", start_time=4.5, end_time=6.0, segment_id="3")
    ]
    
    translations = [
        "Primer segmento, un poco más largo que el original",  # Intentionally longer
        "Segundo segmento",
        "Tercer segmento, también más largo que el original"   # Also longer
    ]
    
    optimized = preserver.optimize_segments(segments, translations, "en", "es")
    
    print("\nOptimized segment timing:")
    for seg in optimized:
        print(f"{seg.start_time:.2f}s - {seg.end_time:.2f}s: {seg.text}")
        print(f"  Duration: {seg.duration:.2f}s, CPS: {seg.characters_per_second:.2f}") 