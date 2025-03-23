#!/usr/bin/env python3
"""
Cross-Language Lip Synchronization Module

This module provides specialized functionality for cross-language lip synchronization,
allowing accurate lip movement synthesis across different languages with varying
phonetic structures and speech patterns.

Key features:
- Advanced phoneme mapping between language pairs
- Viseme-based synchronization (visual phoneme units)
- Handling of language-specific mouth movements
- Support for languages with different syllable structures
- Adaptation for speech rate differences
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field


@dataclass
class VisemeMapping:
    """Mapping between phonemes and visemes (visual mouth shapes)."""
    language: str  # Language code
    phoneme_to_viseme: Dict[str, str]  # Maps phonemes to viseme IDs
    viseme_descriptions: Dict[str, str]  # Descriptions of each viseme
    similar_visemes: Dict[str, List[str]]  # Groups of similar visemes
    
    @classmethod
    def from_file(cls, file_path: str) -> 'VisemeMapping':
        """Load viseme mapping from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Viseme mapping file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            language=data.get("language", ""),
            phoneme_to_viseme=data.get("phoneme_to_viseme", {}),
            viseme_descriptions=data.get("viseme_descriptions", {}),
            similar_visemes=data.get("similar_visemes", {})
        )


@dataclass
class CrossLanguageMap:
    """Mapping for cross-language synchronization."""
    source_language: str  # Source language code
    target_language: str  # Target language code
    phoneme_mapping: Dict[str, str]  # Direct phoneme mappings
    viseme_mapping: Dict[str, str]  # Viseme mappings for visual consistency
    timing_adjustments: Dict[str, float]  # Timing adjustments for specific phonemes
    
    @classmethod
    def from_file(cls, file_path: str) -> 'CrossLanguageMap':
        """Load cross-language mapping from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cross-language mapping file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            source_language=data.get("source_language", ""),
            target_language=data.get("target_language", ""),
            phoneme_mapping=data.get("phoneme_mapping", {}),
            viseme_mapping=data.get("viseme_mapping", {}),
            timing_adjustments=data.get("timing_adjustments", {})
        )


class CrossLanguageSynchronizer:
    """
    Handles synchronization of lip movements across different languages,
    enabling realistic dubbing for translated content.
    """
    
    def __init__(self, mapping_dir: str = "data/lip_sync/mappings"):
        """
        Initialize the cross-language synchronizer.
        
        Args:
            mapping_dir: Directory containing mapping files
        """
        self.mapping_dir = mapping_dir
        self.viseme_mappings: Dict[str, VisemeMapping] = {}
        self.cross_language_maps: Dict[str, CrossLanguageMap] = {}
        
        # Load available mappings
        self._load_mappings()
        
        print(f"Cross-Language Synchronizer initialized")
        print(f"  - Loaded {len(self.viseme_mappings)} viseme mappings")
        print(f"  - Loaded {len(self.cross_language_maps)} cross-language maps")
    
    def _load_mappings(self) -> None:
        """Load available viseme and cross-language mappings."""
        # In a real implementation, this would scan the mapping directory
        # For now, we'll create some placeholder mappings
        
        # Create viseme mappings for English
        english_visemes = {
            "P": ["p", "b", "m"],
            "F": ["f", "v"],
            "TH": ["th", "dh"],
            "S": ["s", "z"],
            "SH": ["sh", "zh", "ch", "jh"],
            "W": ["w"],
            "R": ["r"],
            "L": ["l"],
            "Y": ["y"],
            "A": ["aa", "ae", "ah"],
            "E": ["eh", "ey"],
            "I": ["ih", "iy"],
            "O": ["ow", "ao", "oy"],
            "U": ["uw", "uh"]
        }
        
        # Create mapping English -> Spanish
        en_phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", 
                     "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", 
                     "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", 
                     "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
        
        es_phonemes = ["a", "e", "i", "o", "u", "p", "b", "t", "d", "k", "g", 
                      "f", "s", "z", "x", "j", "y", "r", "rr", "l", "m", "n", "ñ"]
        
        # Create a basic phoneme mapping (simplified)
        en_es_phoneme_map = {
            "AA": "a", "AE": "a", "AH": "a",
            "AO": "o", "AW": "au", "AY": "ai",
            "B": "b", "CH": "ch", "D": "d",
            "DH": "d", "EH": "e", "ER": "er",
            "EY": "ei", "F": "f", "G": "g",
            "HH": "j", "IH": "i", "IY": "i",
            "JH": "y", "K": "k", "L": "l",
            "M": "m", "N": "n", "NG": "n",
            "OW": "o", "OY": "oi", "P": "p",
            "R": "r", "S": "s", "SH": "s",
            "T": "t", "TH": "z", "UH": "u",
            "UW": "u", "V": "b", "W": "u",
            "Y": "i", "Z": "s", "ZH": "s"
        }
        
        # Create viseme mappings (english visemes)
        en_viseme_map = {}
        for viseme, phoneme_list in english_visemes.items():
            for phoneme in phoneme_list:
                phoneme = phoneme.upper()
                if phoneme in en_phonemes:
                    en_viseme_map[phoneme] = viseme
        
        # Create viseme descriptions
        viseme_descriptions = {
            "P": "Closed lips (as in 'p', 'b', 'm')",
            "F": "Lower lip touching upper teeth (as in 'f', 'v')",
            "TH": "Tongue between teeth (as in 'th')",
            "S": "Teeth nearly closed, lips open (as in 's', 'z')",
            "SH": "Rounded lips, slight protrusion (as in 'sh')",
            "W": "Rounded, protruded lips (as in 'w')",
            "R": "Open mouth, slightly rounded (as in 'r')",
            "L": "Mouth open, tongue tip up (as in 'l')",
            "Y": "Slight smile shape (as in 'y')",
            "A": "Open mouth (as in 'ah')",
            "E": "Slightly spread lips (as in 'eh')",
            "I": "Spread lips, teeth showing (as in 'ee')",
            "O": "Rounded open lips (as in 'oh')",
            "U": "Protruded lips, small opening (as in 'oo')"
        }
        
        # Create similar visemes grouping
        similar_visemes = {
            "P_group": ["P", "B", "M"],
            "F_group": ["F", "V"],
            "S_group": ["S", "Z"],
            "SH_group": ["SH", "ZH", "CH", "JH"],
            "Vowel_open": ["A", "E"],
            "Vowel_closed": ["I", "Y"],
            "Vowel_rounded": ["O", "U", "W"]
        }
        
        # Store English viseme mapping
        self.viseme_mappings["en"] = VisemeMapping(
            language="en",
            phoneme_to_viseme=en_viseme_map,
            viseme_descriptions=viseme_descriptions,
            similar_visemes=similar_visemes
        )
        
        # Create and store English -> Spanish cross-language map
        self.cross_language_maps["en-es"] = CrossLanguageMap(
            source_language="en",
            target_language="es",
            phoneme_mapping=en_es_phoneme_map,
            viseme_mapping={},  # For simplicity, we'll use phoneme mapping only for now
            timing_adjustments={
                # Spanish vowels are generally shorter than English vowels
                "a": 0.9,
                "e": 0.9,
                "i": 0.9,
                "o": 0.9,
                "u": 0.9,
                # Spanish consonants like 'rr' need more time
                "rr": 1.3
            }
        )
    
    def get_viseme_mapping(self, language: str) -> Optional[VisemeMapping]:
        """
        Get viseme mapping for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            Viseme mapping or None if not found
        """
        return self.viseme_mappings.get(language)
    
    def get_cross_language_map(self, source_lang: str, target_lang: str) -> Optional[CrossLanguageMap]:
        """
        Get cross-language mapping between source and target languages.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Cross-language mapping or None if not found
        """
        key = f"{source_lang}-{target_lang}"
        return self.cross_language_maps.get(key)
    
    def phoneme_to_viseme(self, phoneme: str, language: str) -> str:
        """
        Convert a phoneme to its corresponding viseme.
        
        Args:
            phoneme: Phoneme to convert
            language: Language code
            
        Returns:
            Viseme ID or the phoneme itself if no mapping exists
        """
        mapping = self.get_viseme_mapping(language)
        if not mapping:
            return phoneme
        
        return mapping.phoneme_to_viseme.get(phoneme.upper(), phoneme)
    
    def map_phonemes(self, 
                   phonemes: List[Dict[str, Any]],
                   source_lang: str,
                   target_lang: str) -> List[Dict[str, Any]]:
        """
        Map phonemes from source language to target language with timing adjustments.
        
        Args:
            phonemes: List of phoneme dictionaries with timing info
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of mapped phonemes with adjusted timing
        """
        cross_map = self.get_cross_language_map(source_lang, target_lang)
        if not cross_map:
            # No mapping available, return the original phonemes
            return phonemes
        
        # Map the phonemes with timing adjustments
        mapped_phonemes = []
        for phoneme_data in phonemes:
            source_phoneme = phoneme_data["phoneme"]
            
            # Get target phoneme
            target_phoneme = cross_map.phoneme_mapping.get(source_phoneme.upper(), source_phoneme)
            
            # Create new data with the mapped phoneme
            mapped_data = phoneme_data.copy()
            mapped_data["phoneme"] = target_phoneme
            mapped_data["original_phoneme"] = source_phoneme
            
            # Apply timing adjustment if available
            timing_factor = cross_map.timing_adjustments.get(target_phoneme, 1.0)
            if timing_factor != 1.0:
                duration = phoneme_data["end_time"] - phoneme_data["start_time"]
                adjusted_duration = duration * timing_factor
                mapped_data["end_time"] = mapped_data["start_time"] + adjusted_duration
                mapped_data["timing_adjusted"] = True
                mapped_data["timing_factor"] = timing_factor
            
            mapped_phonemes.append(mapped_data)
        
        return mapped_phonemes
    
    def map_to_visemes(self, 
                     phonemes: List[Dict[str, Any]], 
                     language: str) -> List[Dict[str, Any]]:
        """
        Convert phonemes to visemes for a specific language.
        
        Args:
            phonemes: List of phoneme dictionaries
            language: Language code
            
        Returns:
            List of viseme dictionaries
        """
        viseme_mapping = self.get_viseme_mapping(language)
        if not viseme_mapping:
            # No mapping available, return the phonemes as visemes
            return [{"viseme": p["phoneme"], **p} for p in phonemes]
        
        # Convert phonemes to visemes
        visemes = []
        for phoneme_data in phonemes:
            phoneme = phoneme_data["phoneme"].upper()
            viseme = viseme_mapping.phoneme_to_viseme.get(phoneme, phoneme)
            
            viseme_data = phoneme_data.copy()
            viseme_data["viseme"] = viseme
            viseme_data["phoneme"] = phoneme_data["phoneme"]
            
            visemes.append(viseme_data)
        
        return visemes
    
    def optimize_viseme_sequence(self, 
                              visemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize a sequence of visemes for more natural transitions.
        
        Args:
            visemes: List of viseme dictionaries
            
        Returns:
            List of optimized visemes
        """
        # In a real implementation, this would apply more sophisticated optimizations
        # For now, we'll just merge adjacent identical visemes
        
        if not visemes:
            return []
        
        optimized = [visemes[0]]
        
        for i in range(1, len(visemes)):
            current = visemes[i]
            previous = optimized[-1]
            
            # If same viseme, merge them
            if current["viseme"] == previous["viseme"]:
                merged = previous.copy()
                merged["end_time"] = current["end_time"]
                merged["duration"] = merged["end_time"] - merged["start_time"]
                merged["phonemes"] = previous.get("phonemes", [previous["phoneme"]]) + [current["phoneme"]]
                optimized[-1] = merged
            else:
                optimized.append(current)
        
        return optimized
    
    def process_cross_language(self, 
                             phonemes: List[Dict[str, Any]],
                             source_lang: str,
                             target_lang: str) -> List[Dict[str, Any]]:
        """
        Process phonemes for cross-language lip synchronization.
        
        This performs the full pipeline:
        1. Map phonemes from source to target language
        2. Convert to visemes
        3. Optimize the viseme sequence
        
        Args:
            phonemes: List of phoneme dictionaries with timing info
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of processed visemes ready for lip synchronization
        """
        # Map phonemes from source to target language
        mapped_phonemes = self.map_phonemes(phonemes, source_lang, target_lang)
        
        # Convert mapped phonemes to visemes
        visemes = self.map_to_visemes(mapped_phonemes, target_lang)
        
        # Optimize the viseme sequence
        optimized_visemes = self.optimize_viseme_sequence(visemes)
        
        return optimized_visemes


class LanguageSpecificMouthShapes:
    """
    Provides language-specific mouth shapes for more accurate lip synchronization.
    
    Different languages have characteristic mouth movements and articulation patterns.
    This class helps adjust visual speech synthesis to account for these differences.
    """
    
    def __init__(self):
        """Initialize with default language-specific mouth shape data."""
        # Language codes mapped to their characteristic mouth shape parameters
        self.language_parameters = {
            "en": {
                "lip_rounding": 1.0,      # Standard English lip rounding
                "jaw_openness": 1.0,      # Standard English jaw openness
                "lip_protrusion": 1.0,    # Standard English lip protrusion
                "tongue_visibility": 0.5  # Standard English tongue visibility
            },
            "fr": {
                "lip_rounding": 1.3,      # French has more rounded vowels
                "jaw_openness": 0.9,      # Slightly less jaw movement
                "lip_protrusion": 1.4,    # More lip protrusion (e.g., for 'u')
                "tongue_visibility": 0.3  # Less visible tongue
            },
            "ja": {
                "lip_rounding": 0.8,      # Less lip rounding
                "jaw_openness": 0.7,      # Less jaw movement
                "lip_protrusion": 0.6,    # Less lip protrusion
                "tongue_visibility": 0.2  # Less visible tongue
            },
            "es": {
                "lip_rounding": 1.1,      # More rounded in some vowels
                "jaw_openness": 1.2,      # More jaw movement
                "lip_protrusion": 0.9,    # Less protrusion
                "tongue_visibility": 0.7  # More visible tongue (e.g., for 'r')
            },
            "de": {
                "lip_rounding": 1.2,      # More rounded vowels
                "jaw_openness": 1.0,      # Standard jaw openness
                "lip_protrusion": 1.1,    # Slightly more protrusion
                "tongue_visibility": 0.4  # Standard tongue visibility
            }
            # Additional languages would be defined here
        }
        
        # Specific mouth shapes for unique phonemes
        self.special_phonemes = {
            "es": {
                "rr": {  # Spanish rolled 'r'
                    "tongue_visibility": 0.9,
                    "jaw_openness": 0.8,
                    "vibration": 1.0
                }
            },
            "fr": {
                "r": {  # French 'r'
                    "throat_constriction": 0.8,
                    "jaw_openness": 0.7
                }
            },
            "de": {
                "ch": {  # German 'ch'
                    "tongue_position": "back",
                    "jaw_openness": 0.6
                }
            }
        }
    
    def get_language_parameters(self, language: str) -> Dict[str, float]:
        """
        Get mouth shape parameters for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary of mouth shape parameters
        """
        return self.language_parameters.get(language, self.language_parameters["en"])
    
    def get_special_phoneme_shape(self, phoneme: str, language: str) -> Optional[Dict[str, Any]]:
        """
        Get special mouth shape for a specific phoneme in a language.
        
        Args:
            phoneme: Phoneme ID
            language: Language code
            
        Returns:
            Dictionary of mouth shape parameters or None if not found
        """
        language_specials = self.special_phonemes.get(language, {})
        return language_specials.get(phoneme.lower())
    
    def adjust_mouth_shape(self, 
                         base_shape: Dict[str, float], 
                         language: str,
                         phoneme: Optional[str] = None) -> Dict[str, float]:
        """
        Adjust a base mouth shape according to language-specific parameters.
        
        Args:
            base_shape: Base mouth shape parameters
            language: Language code
            phoneme: Optional specific phoneme
            
        Returns:
            Adjusted mouth shape parameters
        """
        # Get general language parameters
        lang_params = self.get_language_parameters(language)
        
        # Apply language-specific adjustments
        adjusted_shape = base_shape.copy()
        for param, value in lang_params.items():
            if param in adjusted_shape:
                adjusted_shape[param] *= value
        
        # Apply special phoneme-specific adjustments if applicable
        if phoneme:
            special_shape = self.get_special_phoneme_shape(phoneme, language)
            if special_shape:
                for param, value in special_shape.items():
                    adjusted_shape[param] = value
        
        return adjusted_shape


# Example usage
if __name__ == "__main__":
    # Initialize the cross-language synchronizer
    synchronizer = CrossLanguageSynchronizer()
    
    # Sample phoneme sequence (simplified)
    phonemes = [
        {"phoneme": "HH", "start_time": 0.0, "end_time": 0.1},
        {"phoneme": "EH", "start_time": 0.1, "end_time": 0.2},
        {"phoneme": "L", "start_time": 0.2, "end_time": 0.3},
        {"phoneme": "OW", "start_time": 0.3, "end_time": 0.5}
    ]
    
    # Process for cross-language lip sync (English to Spanish)
    result = synchronizer.process_cross_language(phonemes, "en", "es")
    
    print("Original phonemes:")
    for p in phonemes:
        print(f"  {p['phoneme']}: {p['start_time']:.1f} - {p['end_time']:.1f}")
    
    print("\nProcessed for lip sync (English to Spanish):")
    for v in result:
        print(f"  Viseme: {v['viseme']}, Phoneme: {v['phoneme']}, Time: {v['start_time']:.1f} - {v['end_time']:.1f}")
    
    # Example of language-specific mouth shapes
    mouth_shapes = LanguageSpecificMouthShapes()
    
    base_shape = {
        "lip_rounding": 0.5,
        "jaw_openness": 0.7,
        "lip_protrusion": 0.4
    }
    
    # Adjust for different languages
    en_shape = mouth_shapes.adjust_mouth_shape(base_shape, "en")
    es_shape = mouth_shapes.adjust_mouth_shape(base_shape, "es")
    fr_shape = mouth_shapes.adjust_mouth_shape(base_shape, "fr")
    
    print("\nMouth shape adjustments for different languages:")
    print(f"  English: {en_shape}")
    print(f"  Spanish: {es_shape}")
    print(f"  French: {fr_shape}") 