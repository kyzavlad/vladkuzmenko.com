#!/usr/bin/env python3
"""
Prosody Modeling Module for Voice Translation

This module provides functionality for analyzing, modeling, and preserving prosody
characteristics during voice translation. It handles speaker-specific prosody patterns
and natural speech rhythm, ensuring that translated speech maintains natural intonation.

Key features:
- Speaker-specific prosody modeling
- Natural pause insertion for target language
- Speech rate adjustment for comprehension
- Intonation pattern preservation
- Stress placement adaptation across languages
- Rhythm and emphasis transfer
"""

import numpy as np
import librosa
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field


@dataclass
class ProsodyFeatures:
    """Features describing speech prosody."""
    pitch_contour: np.ndarray  # F0 contour
    pitch_statistics: Dict[str, float]  # Statistical features of pitch
    energy_contour: np.ndarray  # Energy/volume contour
    energy_statistics: Dict[str, float]  # Statistical features of energy
    speech_rate: float  # Overall speech rate
    pauses: List[Tuple[float, float]]  # List of pause positions and durations
    syllable_durations: List[float]  # Duration of each syllable
    stress_positions: List[int]  # Positions of stressed syllables
    phrase_boundaries: List[int]  # Positions of phrase boundaries
    
    @classmethod
    def create_empty(cls) -> 'ProsodyFeatures':
        """Create an empty ProsodyFeatures object with default values."""
        return cls(
            pitch_contour=np.array([]),
            pitch_statistics={"mean": 0.0, "std": 0.0, "range": 0.0, "slope": 0.0},
            energy_contour=np.array([]),
            energy_statistics={"mean": 0.0, "std": 0.0, "range": 0.0},
            speech_rate=1.0,
            pauses=[],
            syllable_durations=[],
            stress_positions=[],
            phrase_boundaries=[]
        )


@dataclass
class SpeechUnit:
    """Represents a unit of speech (word, phrase, or sentence)."""
    text: str  # Text content
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    syllable_count: int  # Number of syllables
    stressed_syllables: List[int]  # Indices of stressed syllables
    has_final_pause: bool = False  # Whether the unit ends with a pause
    pause_duration: float = 0.0  # Duration of the final pause
    pitch_mean: float = 0.0  # Mean pitch for this unit
    energy_mean: float = 0.0  # Mean energy for this unit
    speech_rate: float = 1.0  # Relative speech rate for this unit
    
    @property
    def duration(self) -> float:
        """Get the duration of the speech unit without the final pause."""
        return self.end_time - self.start_time - (self.pause_duration if self.has_final_pause else 0.0)


@dataclass
class ProsodyMapping:
    """Mapping of prosody features between languages."""
    source_language: str  # Source language code
    target_language: str  # Target language code
    speech_rate_factor: float = 1.0  # Speech rate adjustment factor
    pitch_shift: float = 0.0  # Global pitch shift
    energy_scale: float = 1.0  # Global energy scale
    pause_scale: float = 1.0  # Scale factor for pause durations
    pause_insertion_threshold: float = 0.7  # Threshold for inserting pauses in target
    stress_pattern_mapping: Dict[str, str] = field(default_factory=dict)  # Maps stress patterns


class ProsodyModeler:
    """
    Models and preserves speech prosody characteristics during voice translation.
    """
    
    def __init__(self, model_path: str = "models/prosody/model_weights"):
        """
        Initialize the prosody modeler.
        
        Args:
            model_path: Path to pre-trained prosody models
        """
        self.model_path = model_path
        self.loaded_models = False
        
        # Will be initialized when models are loaded
        self.pitch_tracker = None
        self.segmenter = None
        self.syllable_detector = None
        self.stress_detector = None
        
        # Language-specific settings
        self.language_settings = self._load_language_settings()
        
        print(f"Prosody Modeler initialized")
        print(f"  - Model path: {model_path}")
        print(f"  - Supported languages: {', '.join(self.language_settings.keys())}")
    
    def _load_language_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        Load settings for different languages.
        
        Returns:
            Dictionary with language-specific settings
        """
        # In a real implementation, this would load from configuration files
        # For demonstration, we'll use hardcoded values
        
        return {
            "en": {
                "syllable_rate": 4.0,  # Syllables per second (average)
                "avg_pause_duration": 0.3,  # Average pause duration in seconds
                "pause_frequency": 0.25,  # Frequency of pauses (pauses per syllable)
                "sentence_final_lengthening": 1.3,  # Scale factor for final syllable duration
                "stress_pattern": "variable",  # English has variable stress patterns
                "avg_syllable_duration": 0.25,  # Average syllable duration in seconds
            },
            "es": {
                "syllable_rate": 4.6,  # Syllables per second (average)
                "avg_pause_duration": 0.25,  # Average pause duration in seconds
                "pause_frequency": 0.2,  # Frequency of pauses (pauses per syllable)
                "sentence_final_lengthening": 1.5,  # Scale factor for final syllable duration
                "stress_pattern": "penultimate",  # Spanish typically stresses penultimate syllable
                "avg_syllable_duration": 0.22,  # Average syllable duration in seconds
            },
            "fr": {
                "syllable_rate": 4.7,  # Syllables per second (average)
                "avg_pause_duration": 0.22,  # Average pause duration in seconds
                "pause_frequency": 0.18,  # Frequency of pauses (pauses per syllable)
                "sentence_final_lengthening": 1.7,  # Scale factor for final syllable duration
                "stress_pattern": "final",  # French typically stresses final syllable
                "avg_syllable_duration": 0.21,  # Average syllable duration in seconds
            },
            "de": {
                "syllable_rate": 4.2,  # Syllables per second (average)
                "avg_pause_duration": 0.35,  # Average pause duration in seconds
                "pause_frequency": 0.28,  # Frequency of pauses (pauses per syllable)
                "sentence_final_lengthening": 1.4,  # Scale factor for final syllable duration
                "stress_pattern": "first",  # German often stresses first syllable
                "avg_syllable_duration": 0.24,  # Average syllable duration in seconds
            },
            "ja": {
                "syllable_rate": 7.8,  # Mora per second (Japanese uses mora, not syllables)
                "avg_pause_duration": 0.2,  # Average pause duration in seconds
                "pause_frequency": 0.15,  # Frequency of pauses (pauses per syllable)
                "sentence_final_lengthening": 1.2,  # Scale factor for final syllable duration
                "stress_pattern": "pitch-accent",  # Japanese uses pitch accent
                "avg_syllable_duration": 0.13,  # Average mora duration in seconds
            },
            "zh": {
                "syllable_rate": 5.8,  # Syllables per second (average)
                "avg_pause_duration": 0.25,  # Average pause duration in seconds
                "pause_frequency": 0.3,  # Frequency of pauses (pauses per syllable)
                "sentence_final_lengthening": 1.3,  # Scale factor for final syllable duration
                "stress_pattern": "tonal",  # Chinese uses tones instead of stress
                "avg_syllable_duration": 0.17,  # Average syllable duration in seconds
            }
        }
    
    def load_models(self) -> bool:
        """
        Load necessary models for prosody analysis and modeling.
        
        Returns:
            True if successful, False otherwise
        """
        if self.loaded_models:
            return True
        
        try:
            # In a real implementation, this would load actual models
            # For demonstration, we'll just simulate successful loading
            self.pitch_tracker = object()  # Placeholder for actual model
            self.segmenter = object()  # Placeholder for actual model
            self.syllable_detector = object()  # Placeholder for actual model
            self.stress_detector = object()  # Placeholder for actual model
            
            self.loaded_models = True
            print("Prosody models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading prosody models: {e}")
            return False
    
    def extract_prosody_features(self, audio: np.ndarray, sr: int, language: str = "en") -> ProsodyFeatures:
        """
        Extract prosody features from audio.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            language: Language code of the speech
            
        Returns:
            ProsodyFeatures object containing extracted features
        """
        if not self.loaded_models:
            self.load_models()
        
        # Extract pitch (F0) contour
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=sr)
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            pitch_mean = np.mean(f0_voiced)
            pitch_std = np.std(f0_voiced)
            pitch_range = np.max(f0_voiced) - np.min(f0_voiced)
            
            # Calculate pitch slope (linear regression)
            if len(f0_voiced) > 1:
                x = np.arange(len(f0_voiced))
                pitch_slope = np.polyfit(x, f0_voiced, 1)[0]
            else:
                pitch_slope = 0.0
        else:
            pitch_mean = pitch_std = pitch_range = pitch_slope = 0.0
        
        # Extract energy/volume contour
        energy = librosa.feature.rms(y=audio)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        energy_range = np.max(energy) - np.min(energy)
        
        # Detect pauses (segments with low energy)
        pause_threshold = np.mean(energy) * 0.2
        is_pause = energy < pause_threshold
        
        # Find contiguous pause regions
        pause_regions = []
        in_pause = False
        pause_start = 0
        
        for i, pause in enumerate(is_pause):
            if pause and not in_pause:
                # Start of a pause
                in_pause = True
                pause_start = i / len(energy) * (len(audio) / sr)
            elif not pause and in_pause:
                # End of a pause
                in_pause = False
                pause_end = i / len(energy) * (len(audio) / sr)
                if pause_end - pause_start > 0.1:  # Only count pauses longer than 100ms
                    pause_regions.append((pause_start, pause_end - pause_start))
        
        # If still in a pause at the end
        if in_pause:
            pause_end = len(energy) / len(energy) * (len(audio) / sr)
            if pause_end - pause_start > 0.1:
                pause_regions.append((pause_start, pause_end - pause_start))
        
        # Estimate speech rate
        # In a real implementation, this would be more sophisticated
        syllable_count = len(librosa.onset.onset_detect(
            y=audio, sr=sr, units='time', pre_max=0.03, post_max=0.03,
            pre_avg=0.1, post_avg=0.1, delta=0.1, wait=0.03
        ))
        
        total_speech_time = (len(audio) / sr) - sum(duration for _, duration in pause_regions)
        speech_rate = syllable_count / total_speech_time if total_speech_time > 0 else 0
        
        # In a real implementation, we would also:
        # - Detect syllables and their durations
        # - Identify stressed syllables
        # - Detect phrase boundaries
        
        # For demonstration, we'll use placeholder values
        syllable_durations = [0.2] * syllable_count
        stress_positions = [i for i in range(0, syllable_count, 3)]  # Every 3rd syllable
        phrase_boundaries = [i for i in range(0, syllable_count, 8)]  # Every 8th syllable
        
        return ProsodyFeatures(
            pitch_contour=f0,
            pitch_statistics={
                "mean": float(pitch_mean),
                "std": float(pitch_std),
                "range": float(pitch_range),
                "slope": float(pitch_slope)
            },
            energy_contour=energy,
            energy_statistics={
                "mean": float(energy_mean),
                "std": float(energy_std),
                "range": float(energy_range)
            },
            speech_rate=float(speech_rate),
            pauses=pause_regions,
            syllable_durations=syllable_durations,
            stress_positions=stress_positions,
            phrase_boundaries=phrase_boundaries
        )
    
    def segment_speech(self, 
                     audio: np.ndarray, 
                     sr: int, 
                     transcript: str,
                     language: str = "en") -> List[SpeechUnit]:
        """
        Segment speech into words, phrases, or sentences.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            transcript: Transcript of the speech
            language: Language code of the speech
            
        Returns:
            List of SpeechUnit objects
        """
        if not self.loaded_models:
            self.load_models()
        
        # In a real implementation, this would use sophisticated forced alignment
        # For demonstration, we'll create simplified speech units
        
        # Split transcript into sentences (very basic split)
        sentences = []
        current = ""
        for char in transcript:
            current += char
            if char in ".!?":
                sentences.append(current.strip())
                current = ""
        
        if current:
            sentences.append(current.strip())
        
        # Extract prosody features to help with segmentation
        prosody = self.extract_prosody_features(audio, sr, language)
        
        # Approximate duration per sentence
        total_duration = len(audio) / sr
        duration_per_sentence = total_duration / len(sentences) if sentences else 0
        
        # Create speech units
        speech_units = []
        
        for i, sentence in enumerate(sentences):
            # Very basic approximation of timing
            start_time = i * duration_per_sentence
            end_time = (i + 1) * duration_per_sentence
            
            # Approximate syllable count based on language
            if language == "ja":
                # For Japanese, count mora (roughly number of characters in hiragana)
                syllable_count = len(sentence)
            elif language == "zh":
                # For Chinese, count characters
                syllable_count = len(sentence)
            else:
                # For other languages, use a simple heuristic
                syllable_count = sum(1 for c in sentence if c.lower() in "aeiouy")
                syllable_count = max(syllable_count, len(sentence.split()))
            
            # Determine if there's a pause after this sentence
            has_final_pause = i < len(sentences) - 1
            pause_duration = 0.3 if has_final_pause else 0.0  # Default pause
            
            # Find the nearest pause in detected pauses
            if has_final_pause and prosody.pauses:
                for pause_start, pause_dur in prosody.pauses:
                    if abs(pause_start - end_time) < 0.5:  # If close to expected end
                        pause_duration = pause_dur
                        break
            
            # Determine stressed syllables (simplified)
            if language == "en":
                # For English, stress approximately every other syllable
                stressed_syllables = [i for i in range(0, syllable_count, 2)]
            elif language == "fr":
                # For French, stress the last syllable
                stressed_syllables = [syllable_count - 1] if syllable_count > 0 else []
            elif language == "es":
                # For Spanish, typically stress the penultimate syllable
                stressed_syllables = [syllable_count - 2] if syllable_count > 1 else []
            else:
                # Default pattern
                stressed_syllables = [i for i in range(0, syllable_count, 3)]
            
            # Create and add the speech unit
            speech_unit = SpeechUnit(
                text=sentence,
                start_time=start_time,
                end_time=end_time,
                syllable_count=syllable_count,
                stressed_syllables=stressed_syllables,
                has_final_pause=has_final_pause,
                pause_duration=pause_duration,
                pitch_mean=prosody.pitch_statistics["mean"],
                energy_mean=prosody.energy_statistics["mean"],
                speech_rate=prosody.speech_rate
            )
            
            speech_units.append(speech_unit)
        
        return speech_units
    
    def adjust_speech_rate(self, 
                         source_prosody: ProsodyFeatures,
                         source_lang: str,
                         target_lang: str) -> float:
        """
        Calculate speech rate adjustment factor between languages.
        
        Args:
            source_prosody: Prosody features of the source audio
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Speech rate adjustment factor
        """
        # Get language settings
        source_settings = self.language_settings.get(source_lang, self.language_settings["en"])
        target_settings = self.language_settings.get(target_lang, self.language_settings["en"])
        
        # Calculate base adjustment factor from language averages
        base_factor = source_settings["syllable_rate"] / target_settings["syllable_rate"]
        
        # Adjust based on the source speaker's actual rate compared to the language average
        if source_settings["syllable_rate"] > 0:
            speaker_factor = source_prosody.speech_rate / source_settings["syllable_rate"]
        else:
            speaker_factor = 1.0
        
        # Ensure the adjustment stays within reasonable bounds
        rate_factor = base_factor * speaker_factor
        
        # Clamp to reasonable bounds to prevent extreme adjustments
        return max(0.5, min(2.0, rate_factor))
    
    def generate_pause_pattern(self, 
                             source_units: List[SpeechUnit],
                             target_text: str,
                             source_lang: str,
                             target_lang: str) -> List[Tuple[int, float]]:
        """
        Generate pause positions and durations for the target text.
        
        Args:
            source_units: List of source speech units
            target_text: Target text (translated)
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of (position, duration) tuples for pauses
        """
        # Get language settings
        source_settings = self.language_settings.get(source_lang, self.language_settings["en"])
        target_settings = self.language_settings.get(target_lang, self.language_settings["en"])
        
        # Split target text into sentences (very basic)
        sentences = []
        current = ""
        for char in target_text:
            current += char
            if char in ".!?":
                sentences.append(current.strip())
                current = ""
        
        if current:
            sentences.append(current.strip())
        
        # If there are fewer source units than target sentences, pad the source units
        while len(source_units) < len(sentences):
            # Create a dummy unit with average values
            dummy_unit = SpeechUnit(
                text="",
                start_time=0.0,
                end_time=0.0,
                syllable_count=0,
                stressed_syllables=[],
                has_final_pause=True,
                pause_duration=source_settings["avg_pause_duration"]
            )
            source_units.append(dummy_unit)
        
        # If there are more source units than target sentences, use only the first N units
        source_units = source_units[:len(sentences)]
        
        # Calculate pause positions and durations
        pauses = []
        
        for i, (source_unit, target_sentence) in enumerate(zip(source_units, sentences)):
            if source_unit.has_final_pause:
                # Calculate pause duration adjusted for languages
                pause_scale = target_settings["avg_pause_duration"] / source_settings["avg_pause_duration"]
                pause_duration = source_unit.pause_duration * pause_scale
                
                # Ensure minimum pause duration
                pause_duration = max(pause_duration, 0.1)
                
                # Calculate position (character index) in target text
                position = len("".join(sentences[:i+1]))
                
                pauses.append((position, pause_duration))
        
        # Add additional pauses for very long sentences
        for i, sentence in enumerate(sentences):
            if len(sentence) > 100:  # Very long sentence
                # Add a mid-sentence pause
                commas = [j for j, char in enumerate(sentence) if char == ',']
                if commas:
                    # Find a comma roughly in the middle
                    middle_idx = len(sentence) // 2
                    comma_pos = min(commas, key=lambda x: abs(x - middle_idx))
                    
                    # Offset by the preceding text
                    position = len("".join(sentences[:i])) + comma_pos
                    
                    # Add a shorter pause
                    pause_duration = target_settings["avg_pause_duration"] * 0.7
                    
                    pauses.append((position, pause_duration))
        
        return sorted(pauses)
    
    def create_prosody_mapping(self, 
                             source_prosody: ProsodyFeatures,
                             source_lang: str,
                             target_lang: str) -> ProsodyMapping:
        """
        Create a mapping for transferring prosody from source to target language.
        
        Args:
            source_prosody: Prosody features of the source audio
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            ProsodyMapping object for transferring prosody
        """
        # Get language settings
        source_settings = self.language_settings.get(source_lang, self.language_settings["en"])
        target_settings = self.language_settings.get(target_lang, self.language_settings["en"])
        
        # Calculate speech rate adjustment
        speech_rate_factor = self.adjust_speech_rate(source_prosody, source_lang, target_lang)
        
        # Calculate pause scale factor
        pause_scale = target_settings["avg_pause_duration"] / source_settings["avg_pause_duration"]
        
        # For pitch and energy, maintain the speaker's characteristics
        # No global pitch shift or energy scale by default
        pitch_shift = 0.0
        energy_scale = 1.0
        
        # Create stress pattern mapping
        stress_mapping = {}
        
        # Map between different stress patterns
        if source_settings["stress_pattern"] != target_settings["stress_pattern"]:
            if source_settings["stress_pattern"] == "variable" and target_settings["stress_pattern"] == "penultimate":
                stress_mapping = {"variable": "penultimate"}
            elif source_settings["stress_pattern"] == "variable" and target_settings["stress_pattern"] == "final":
                stress_mapping = {"variable": "final"}
            # Add more mappings as needed
        
        return ProsodyMapping(
            source_language=source_lang,
            target_language=target_lang,
            speech_rate_factor=speech_rate_factor,
            pitch_shift=pitch_shift,
            energy_scale=energy_scale,
            pause_scale=pause_scale,
            pause_insertion_threshold=0.7,
            stress_pattern_mapping=stress_mapping
        )
    
    def apply_prosody_mapping(self, 
                            mapping: ProsodyMapping, 
                            source_units: List[SpeechUnit],
                            target_text: str) -> Dict[str, Any]:
        """
        Apply prosody mapping to generate prosody parameters for target speech.
        
        Args:
            mapping: ProsodyMapping for the language pair
            source_units: List of source speech units
            target_text: Target text (translated)
            
        Returns:
            Dictionary with prosody parameters for speech synthesis
        """
        # Split target text into sentences (very basic)
        sentences = []
        current = ""
        for char in target_text:
            current += char
            if char in ".!?":
                sentences.append(current.strip())
                current = ""
        
        if current:
            sentences.append(current.strip())
        
        # Generate pause pattern
        pauses = self.generate_pause_pattern(
            source_units, target_text, mapping.source_language, mapping.target_language
        )
        
        # Calculate syllable durations for the target text
        target_durations = []
        
        source_lang = mapping.source_language
        target_lang = mapping.target_language
        source_settings = self.language_settings.get(source_lang, self.language_settings["en"])
        target_settings = self.language_settings.get(target_lang, self.language_settings["en"])
        
        # Basic syllable count for target text
        if target_lang == "ja":
            # For Japanese, count mora (roughly number of characters in hiragana)
            syllable_count = len(target_text)
        elif target_lang == "zh":
            # For Chinese, count characters
            syllable_count = len(target_text)
        else:
            # For other languages, use a simple heuristic
            syllable_count = sum(1 for c in target_text if c.lower() in "aeiouy")
            syllable_count = max(syllable_count, len(target_text.split()))
        
        # Basic duration calculation
        avg_syllable_duration = target_settings["avg_syllable_duration"]
        target_durations = [avg_syllable_duration] * syllable_count
        
        # Adjust speech rate
        for i in range(len(target_durations)):
            target_durations[i] /= mapping.speech_rate_factor
        
        # Compile prosody parameters for speech synthesis
        prosody_params = {
            "pitch_shift": mapping.pitch_shift,
            "energy_scale": mapping.energy_scale,
            "speech_rate_factor": mapping.speech_rate_factor,
            "pauses": pauses,
            "syllable_durations": target_durations,
            "pitch_contour": None,  # Would be generated by a real system
            "energy_contour": None,  # Would be generated by a real system
        }
        
        return prosody_params
    
    def process_audio(self,
                    audio: np.ndarray, 
                    sr: int,
                    transcript: str,
                    translated_text: str,
                    source_lang: str,
                    target_lang: str) -> Dict[str, Any]:
        """
        Process audio to extract and map prosody for voice translation.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            transcript: Transcript of the source speech
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dictionary with prosody parameters for speech synthesis
        """
        if not self.loaded_models:
            self.load_models()
        
        # Extract prosody features
        prosody_features = self.extract_prosody_features(audio, sr, source_lang)
        
        # Segment speech into units
        speech_units = self.segment_speech(audio, sr, transcript, source_lang)
        
        # Create prosody mapping
        mapping = self.create_prosody_mapping(prosody_features, source_lang, target_lang)
        
        # Apply mapping to generate prosody parameters
        prosody_params = self.apply_prosody_mapping(mapping, speech_units, translated_text)
        
        return prosody_params


# Example usage
if __name__ == "__main__":
    import librosa
    
    # Initialize the prosody modeler
    prosody_modeler = ProsodyModeler()
    
    # Example audio path (this would be a real file in actual usage)
    example_audio_path = "example/speech/english_speech.wav"
    example_transcript = "This is an example sentence. It demonstrates prosody modeling for voice translation."
    example_translation = "Esta es una frase de ejemplo. Demuestra el modelado de prosodia para la traducción de voz."
    
    try:
        # Try to load example audio
        audio, sr = librosa.load(example_audio_path, sr=None)
        
        # Process the audio
        prosody_params = prosody_modeler.process_audio(
            audio, sr, example_transcript, example_translation, "en", "es"
        )
        
        # Print results
        print("Prosody mapping parameters:")
        for key, value in prosody_params.items():
            if key == "pauses":
                print(f"  pauses: {len(value)} pauses generated")
                for i, (pos, dur) in enumerate(value[:3]):
                    print(f"    {i+1}: position {pos}, duration {dur:.2f}s")
                if len(value) > 3:
                    print(f"    ... and {len(value) - 3} more")
            elif key == "syllable_durations":
                print(f"  syllable_durations: {len(value)} durations generated")
                avg_dur = sum(value) / len(value) if value else 0
                print(f"    average duration: {avg_dur:.4f}s")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Example could not be run: {e}")
        print("This is just a demonstration. Please use with actual audio files.") 