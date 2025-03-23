#!/usr/bin/env python3
"""
Emotion Transfer Module for Voice Translation

This module provides specialized functionality for transferring emotional characteristics
from source to target speech during voice translation. It ensures that the emotional
content of the original speech is preserved in the translated version.

Key capabilities:
- Emotion detection from speech audio
- Emotional feature extraction
- Emotion-to-acoustic parameter mapping
- Cross-language emotion adaptation
- Generation of emotion-aware voice synthesis parameters
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field


@dataclass
class EmotionFeatures:
    """Extracted acoustic features related to emotion in speech."""
    pitch_stats: Dict[str, float]  # Statistical features of pitch (mean, std, range, etc.)
    energy_stats: Dict[str, float]  # Statistical features of energy
    speaking_rate: float  # Overall speaking rate
    voice_quality: Dict[str, float]  # Voice quality parameters
    spectral_features: Dict[str, np.ndarray]  # Spectral features
    temporal_dynamics: Dict[str, np.ndarray]  # Temporal dynamics of features
    
    @classmethod
    def create_empty(cls) -> 'EmotionFeatures':
        """Create an empty EmotionFeatures object with default values."""
        return cls(
            pitch_stats={"mean": 0.0, "std": 0.0, "range": 0.0, "slope": 0.0},
            energy_stats={"mean": 0.0, "std": 0.0, "range": 0.0},
            speaking_rate=1.0,
            voice_quality={"breathiness": 0.5, "tenseness": 0.5, "jitter": 0.0, "shimmer": 0.0},
            spectral_features={},
            temporal_dynamics={}
        )


@dataclass
class EmotionTransferParameters:
    """Parameters for emotion transfer in voice synthesis."""
    pitch_shift: float = 0.0  # Global pitch shift in semitones
    pitch_range_scale: float = 1.0  # Scale factor for pitch range
    pitch_variability: float = 1.0  # How much pitch varies over time
    energy_scale: float = 1.0  # Overall energy/volume scaling
    energy_variability: float = 1.0  # How much energy varies over time
    speaking_rate_scale: float = 1.0  # Speaking rate scaling factor
    articulation_strength: float = 1.0  # How clearly words are articulated
    voice_quality_adjustments: Dict[str, float] = field(default_factory=dict)  # Voice quality parameter adjustments
    attack_scale: float = 1.0  # Scale factor for attack portions of phonemes
    decay_scale: float = 1.0  # Scale factor for decay portions of phonemes


class EmotionTransferSystem:
    """
    System for transferring emotional characteristics from source to target speech
    across different languages.
    """
    
    def __init__(self, model_path: str = "models/emotion/transfer_model"):
        """
        Initialize the emotion transfer system.
        
        Args:
            model_path: Path to pre-trained emotion models
        """
        self.model_path = model_path
        self.emotion_detector = None  # Will be loaded when needed
        self.feature_extractor = None  # Will be loaded when needed
        self.parameter_generator = None  # Will be loaded when needed
        
        # Emotion-to-acoustic mappings for various languages
        self.emotion_acoustic_mappings = {
            "en": self._load_language_mapping("en"),
            "es": self._load_language_mapping("es"),
            "fr": self._load_language_mapping("fr"),
            "de": self._load_language_mapping("de"),
            "ja": self._load_language_mapping("ja"),
            "zh": self._load_language_mapping("zh"),
        }
        
        print(f"Emotion Transfer System initialized")
        print(f"  - Model path: {model_path}")
        print(f"  - Supported languages: {', '.join(self.emotion_acoustic_mappings.keys())}")
    
    def _load_language_mapping(self, language: str) -> Dict[str, Dict[str, float]]:
        """
        Load emotion-to-acoustic parameter mappings for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary mapping emotions to acoustic parameter adjustments
        """
        # In a real implementation, this would load language-specific mappings from files
        # For demonstration, we'll use hardcoded values for common emotions
        
        # Default mappings (for English)
        default_mappings = {
            "neutral": {
                "pitch_shift": 0.0,
                "pitch_range_scale": 1.0,
                "energy_scale": 1.0,
                "speaking_rate_scale": 1.0,
                "articulation_strength": 1.0,
            },
            "happy": {
                "pitch_shift": 2.0,  # Higher pitch
                "pitch_range_scale": 1.5,  # More pitch variation
                "energy_scale": 1.2,  # Louder
                "speaking_rate_scale": 1.2,  # Faster
                "articulation_strength": 1.1,  # More articulated
            },
            "sad": {
                "pitch_shift": -1.5,  # Lower pitch
                "pitch_range_scale": 0.8,  # Less pitch variation
                "energy_scale": 0.8,  # Softer
                "speaking_rate_scale": 0.8,  # Slower
                "articulation_strength": 0.9,  # Less articulated
            },
            "angry": {
                "pitch_shift": 1.0,  # Slightly higher pitch
                "pitch_range_scale": 1.3,  # More pitch variation
                "energy_scale": 1.4,  # Much louder
                "speaking_rate_scale": 1.1,  # Slightly faster
                "articulation_strength": 1.3,  # More articulated
            },
            "fearful": {
                "pitch_shift": 3.0,  # Much higher pitch
                "pitch_range_scale": 1.4,  # More pitch variation
                "energy_scale": 0.9,  # Slightly softer
                "speaking_rate_scale": 1.3,  # Faster
                "articulation_strength": 0.8,  # Less articulated
            },
            "disgusted": {
                "pitch_shift": -0.5,  # Slightly lower pitch
                "pitch_range_scale": 0.9,  # Less pitch variation
                "energy_scale": 1.1,  # Slightly louder
                "speaking_rate_scale": 0.9,  # Slightly slower
                "articulation_strength": 1.2,  # More articulated
            },
            "surprised": {
                "pitch_shift": 4.0,  # Much higher pitch
                "pitch_range_scale": 1.6,  # Much more pitch variation
                "energy_scale": 1.3,  # Louder
                "speaking_rate_scale": 1.0,  # Normal speed
                "articulation_strength": 1.1,  # More articulated
            }
        }
        
        # Adjust for language-specific emotional expression
        # In a real implementation, these would be based on research and data
        if language == "ja":  # Japanese
            # Japanese emotional expression tends to be more constrained
            for emotion in default_mappings:
                if emotion != "neutral":
                    default_mappings[emotion]["pitch_range_scale"] *= 0.8
                    default_mappings[emotion]["energy_scale"] *= 0.9
        
        elif language == "es" or language == "it":  # Spanish/Italian
            # Spanish/Italian emotional expression tends to be more exaggerated
            for emotion in default_mappings:
                if emotion != "neutral":
                    default_mappings[emotion]["pitch_range_scale"] *= 1.2
                    default_mappings[emotion]["energy_scale"] *= 1.1
        
        return default_mappings
    
    def extract_emotion_features(self, audio: np.ndarray, sr: int) -> EmotionFeatures:
        """
        Extract emotion-related acoustic features from audio.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            
        Returns:
            EmotionFeatures object containing extracted features
        """
        # In a real implementation, this would use sophisticated feature extraction
        # For demonstration, we'll create basic features using librosa
        
        # Extract pitch (F0) using PYIN algorithm
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
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_range = 0.0
            pitch_slope = 0.0
        
        # Extract energy/volume
        energy = librosa.feature.rms(y=audio)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        energy_range = np.max(energy) - np.min(energy)
        
        # Estimate speaking rate
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        speaking_rate = tempo / 120.0  # Normalize around a standard tempo of 120 BPM
        
        # Extract voice quality metrics (simplified)
        # In a real implementation, these would be more sophisticated
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Use some MFCC statistics as proxies for voice quality
        breathiness = np.mean(mfccs[1])  # Using MFCC coefficient as proxy
        tenseness = np.mean(spectral_centroid) / 1000.0  # Using spectral centroid as proxy
        jitter = 0.01  # Placeholder
        shimmer = 0.05  # Placeholder
        
        # Create and return the emotion features
        return EmotionFeatures(
            pitch_stats={
                "mean": float(pitch_mean),
                "std": float(pitch_std),
                "range": float(pitch_range),
                "slope": float(pitch_slope)
            },
            energy_stats={
                "mean": float(energy_mean),
                "std": float(energy_std),
                "range": float(energy_range)
            },
            speaking_rate=float(speaking_rate),
            voice_quality={
                "breathiness": float(breathiness),
                "tenseness": float(tenseness),
                "jitter": float(jitter),
                "shimmer": float(shimmer)
            },
            spectral_features={
                "mfccs": mfccs,
                "centroid": spectral_centroid
            },
            temporal_dynamics={
                "energy_contour": energy,
                "pitch_contour": f0
            }
        )
    
    def detect_emotion_from_features(self, features: EmotionFeatures) -> Dict[str, float]:
        """
        Detect emotion from acoustic features.
        
        Args:
            features: Extracted emotion features
            
        Returns:
            Dictionary mapping emotion labels to confidence scores
        """
        # In a real implementation, this would use a trained classifier
        # For demonstration, we'll use a simplified rule-based system
        
        # Initialize all emotions with low confidence
        emotions = {
            "neutral": 0.1,
            "happy": 0.1,
            "sad": 0.1,
            "angry": 0.1,
            "fearful": 0.1,
            "disgusted": 0.1,
            "surprised": 0.1
        }
        
        # Apply simple rules based on acoustic features
        pitch_mean = features.pitch_stats["mean"]
        pitch_range = features.pitch_stats["range"]
        energy_mean = features.energy_stats["mean"]
        speaking_rate = features.speaking_rate
        
        # Neutral: moderate values for all parameters
        emotions["neutral"] = 0.5
        
        # Happy: higher pitch, wider range, higher energy, faster
        if pitch_mean > 180 and pitch_range > 100 and energy_mean > 0.1 and speaking_rate > 1.1:
            emotions["happy"] = 0.7
        
        # Sad: lower pitch, narrower range, lower energy, slower
        if pitch_mean < 150 and pitch_range < 80 and energy_mean < 0.08 and speaking_rate < 0.9:
            emotions["sad"] = 0.7
        
        # Angry: higher energy, wider pitch range, moderate to fast rate
        if energy_mean > 0.12 and pitch_range > 120 and speaking_rate >= 1.0:
            emotions["angry"] = 0.7
        
        # Fearful: higher pitch, wider range, variable energy
        if pitch_mean > 200 and pitch_range > 150:
            emotions["fearful"] = 0.6
        
        # Surprised: very high pitch, wide range, higher energy
        if pitch_mean > 250 and pitch_range > 200 and energy_mean > 0.11:
            emotions["surprised"] = 0.8
        
        # Normalize to sum to 1.0
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] /= total
        
        return emotions
    
    def generate_transfer_parameters(self, 
                                  source_emotion: Dict[str, float],
                                  source_lang: str,
                                  target_lang: str) -> EmotionTransferParameters:
        """
        Generate parameters for transferring emotion from source to target language.
        
        Args:
            source_emotion: Detected emotion as confidence scores
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            EmotionTransferParameters for voice synthesis
        """
        # Get mappings for source and target languages
        source_mappings = self.emotion_acoustic_mappings.get(source_lang, self.emotion_acoustic_mappings["en"])
        target_mappings = self.emotion_acoustic_mappings.get(target_lang, self.emotion_acoustic_mappings["en"])
        
        # Weighted combination of parameters based on emotion confidences
        params = {
            "pitch_shift": 0.0,
            "pitch_range_scale": 1.0,
            "energy_scale": 1.0,
            "speaking_rate_scale": 1.0,
            "articulation_strength": 1.0,
            "voice_quality_adjustments": {},
            "pitch_variability": 1.0,
            "energy_variability": 1.0,
            "attack_scale": 1.0,
            "decay_scale": 1.0
        }
        
        # Calculate weighted parameters from source mappings
        source_params = {}
        for param in ["pitch_shift", "pitch_range_scale", "energy_scale", "speaking_rate_scale", "articulation_strength"]:
            source_params[param] = sum(
                emotion_score * source_mappings[emotion][param]
                for emotion, emotion_score in source_emotion.items()
                if emotion in source_mappings
            )
        
        # Calculate equivalent parameters for target language
        # This adjusts the parameters to account for language-specific emotion expression
        for param in ["pitch_shift", "pitch_range_scale", "energy_scale", "speaking_rate_scale", "articulation_strength"]:
            # Calculate normalized parameter (how far from neutral in the source language)
            if param in source_mappings["neutral"] and source_mappings["neutral"][param] != 0:
                normalized_param = source_params[param] / source_mappings["neutral"][param]
                
                # Apply the same relative change to the target language's neutral setting
                params[param] = normalized_param * target_mappings["neutral"][param]
            else:
                params[param] = source_params[param]
        
        # Add additional parameters based on dominant emotion
        dominant_emotion = max(source_emotion.items(), key=lambda x: x[1])[0]
        
        if dominant_emotion == "happy":
            params["pitch_variability"] = 1.3
            params["energy_variability"] = 1.2
            params["attack_scale"] = 1.1
        elif dominant_emotion == "sad":
            params["pitch_variability"] = 0.8
            params["energy_variability"] = 0.9
            params["decay_scale"] = 1.2
        elif dominant_emotion == "angry":
            params["pitch_variability"] = 1.2
            params["energy_variability"] = 1.4
            params["attack_scale"] = 1.3
        elif dominant_emotion == "fearful":
            params["pitch_variability"] = 1.4
            params["energy_variability"] = 1.3
            params["attack_scale"] = 0.9
        
        # Voice quality adjustments
        voice_quality = {}
        
        if dominant_emotion == "happy":
            voice_quality = {"breathiness": -0.1, "tenseness": 0.1}
        elif dominant_emotion == "sad":
            voice_quality = {"breathiness": 0.2, "tenseness": -0.2}
        elif dominant_emotion == "angry":
            voice_quality = {"breathiness": -0.3, "tenseness": 0.3, "jitter": 0.1}
        elif dominant_emotion == "fearful":
            voice_quality = {"breathiness": 0.2, "tenseness": 0.2, "jitter": 0.2}
        
        params["voice_quality_adjustments"] = voice_quality
        
        # Create and return the transfer parameters
        return EmotionTransferParameters(
            pitch_shift=params["pitch_shift"],
            pitch_range_scale=params["pitch_range_scale"],
            pitch_variability=params["pitch_variability"],
            energy_scale=params["energy_scale"],
            energy_variability=params["energy_variability"],
            speaking_rate_scale=params["speaking_rate_scale"],
            articulation_strength=params["articulation_strength"],
            voice_quality_adjustments=params["voice_quality_adjustments"],
            attack_scale=params["attack_scale"],
            decay_scale=params["decay_scale"]
        )
    
    def apply_emotion_transfer(self, 
                            audio: np.ndarray,
                            sr: int,
                            transfer_params: EmotionTransferParameters) -> np.ndarray:
        """
        Apply emotion transfer parameters to audio.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            transfer_params: Parameters for emotion transfer
            
        Returns:
            Modified audio with transformed emotional characteristics
        """
        # In a real implementation, this would apply sophisticated transformations
        # For demonstration, we'll apply basic transformations
        
        # Create a copy of the audio to modify
        modified_audio = audio.copy()
        
        # Pitch shifting (very simplified)
        if transfer_params.pitch_shift != 0:
            modified_audio = librosa.effects.pitch_shift(
                modified_audio, sr=sr, n_steps=transfer_params.pitch_shift
            )
        
        # Time stretching (for speaking rate)
        if transfer_params.speaking_rate_scale != 1.0:
            rate_factor = 1.0 / transfer_params.speaking_rate_scale  # Inverse because librosa speeds up when factor < 1
            modified_audio = librosa.effects.time_stretch(modified_audio, rate=rate_factor)
        
        # Energy scaling
        if transfer_params.energy_scale != 1.0:
            modified_audio = modified_audio * transfer_params.energy_scale
        
        # In a real implementation, we would also apply:
        # - Pitch contour transformation based on pitch_range_scale and pitch_variability
        # - Energy contour transformation based on energy_variability
        # - Voice quality adjustments
        # - Attack and decay scaling for phonemes
        
        return modified_audio
    
    def process_audio(self,
                    source_audio: np.ndarray,
                    sr: int,
                    source_lang: str,
                    target_lang: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process audio for emotion transfer from source to target language.
        
        Args:
            source_audio: Source audio signal as numpy array
            sr: Sample rate of the audio
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Tuple of (transfer parameters as JSON-serializable dict, detected emotion)
        """
        # Extract emotion features from source audio
        emotion_features = self.extract_emotion_features(source_audio, sr)
        
        # Detect emotion from features
        detected_emotion = self.detect_emotion_from_features(emotion_features)
        
        # Generate transfer parameters
        transfer_params = self.generate_transfer_parameters(
            detected_emotion, source_lang, target_lang
        )
        
        # Convert parameters to dict for serialization
        transfer_params_dict = {
            "pitch_shift": transfer_params.pitch_shift,
            "pitch_range_scale": transfer_params.pitch_range_scale,
            "pitch_variability": transfer_params.pitch_variability,
            "energy_scale": transfer_params.energy_scale,
            "energy_variability": transfer_params.energy_variability,
            "speaking_rate_scale": transfer_params.speaking_rate_scale,
            "articulation_strength": transfer_params.articulation_strength,
            "voice_quality_adjustments": transfer_params.voice_quality_adjustments,
            "attack_scale": transfer_params.attack_scale,
            "decay_scale": transfer_params.decay_scale
        }
        
        return transfer_params_dict, detected_emotion


# Example usage
if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt
    
    # Initialize the emotion transfer system
    emotion_transfer = EmotionTransferSystem()
    
    # Example audio path (this would be a real file in actual usage)
    example_audio_path = "example/emotions/happy_speech.wav"
    
    try:
        # Try to load example audio
        audio, sr = librosa.load(example_audio_path, sr=None)
        
        # Process the audio
        emotion_features = emotion_transfer.extract_emotion_features(audio, sr)
        detected_emotion = emotion_transfer.detect_emotion_from_features(emotion_features)
        
        # Print results
        print("Detected emotions:")
        for emotion, score in sorted(detected_emotion.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {score:.2f}")
        
        # Generate transfer parameters for English to Spanish
        transfer_params = emotion_transfer.generate_transfer_parameters(
            detected_emotion, "en", "es"
        )
        
        print("\nTransfer parameters:")
        for key, value in vars(transfer_params).items():
            print(f"  {key}: {value}")
        
        # Apply emotion transfer
        modified_audio = emotion_transfer.apply_emotion_transfer(audio, sr, transfer_params)
        
        print("\nAudio transformation applied")
        
        # Plot before/after waveforms
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title("Original Audio")
        plt.plot(audio)
        
        plt.subplot(2, 1, 2)
        plt.title("Emotion-Transferred Audio")
        plt.plot(modified_audio)
        
        plt.tight_layout()
        plt.savefig("emotion_transfer_example.png")
        print("Visualization saved to emotion_transfer_example.png")
        
    except Exception as e:
        print(f"Example could not be run: {e}")
        print("This is just a demonstration. Please use with actual audio files.") 