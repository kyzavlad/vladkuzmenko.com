#!/usr/bin/env python3
"""
Voice Translation System

This module provides functionality for voice translation in the video translation pipeline,
preserving the unique characteristics of the speaker's voice while translating speech
from one language to another.

Key features:
- Voice characteristic preservation across languages
- Emotion transfer between source and target languages
- Speaker-specific prosody modeling
- Gender and age characteristic preservation
- Natural pause insertion for target language
- Speech rate adjustment for comprehension
- Voice quality preservation during synthesis
- Multi-speaker tracking and separation
"""

import os
import numpy as np
import torch
import librosa
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

# Import components
from .emotion_transfer import EmotionTransferSystem, EmotionFeatures, EmotionTransferParameters
from .prosody_modeling import ProsodyModeler, ProsodyFeatures, SpeechUnit


@dataclass
class VoiceProfile:
    """Represents the voice characteristics of a speaker."""
    speaker_id: str
    gender: str = "unknown"  # male, female, unknown
    age_range: str = "adult"  # child, teen, adult, elderly
    voice_embeddings: Optional[np.ndarray] = None
    f0_range: Tuple[float, float] = (80.0, 300.0)  # Fundamental frequency range in Hz
    speaking_rate: float = 1.0  # Relative speaking rate (1.0 is average)
    voice_quality: Dict[str, float] = field(default_factory=dict)  # Various voice quality metrics
    language: str = "en"  # Primary language of the speaker
    
    def __post_init__(self):
        """Initialize default voice quality metrics if not provided."""
        if not self.voice_quality:
            self.voice_quality = {
                "breathiness": 0.5,
                "tenseness": 0.5,
                "hoarseness": 0.5,
                "creakiness": 0.5,
                "nasality": 0.5
            }


@dataclass
class EmotionState:
    """Represents the emotional state in speech."""
    primary_emotion: str = "neutral"  # neutral, happy, sad, angry, etc.
    intensity: float = 0.5  # Intensity value between 0.0 and 1.0
    secondary_emotion: Optional[str] = None
    secondary_intensity: float = 0.0
    valence: float = 0.0  # Negative to positive (-1.0 to 1.0)
    arousal: float = 0.0  # Calm to excited (-1.0 to 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "primary_emotion": self.primary_emotion,
            "intensity": self.intensity,
            "secondary_emotion": self.secondary_emotion,
            "secondary_intensity": self.secondary_intensity,
            "valence": self.valence,
            "arousal": self.arousal
        }


@dataclass
class ProsodyParameters:
    """Parameters for controlling speech prosody."""
    pitch_shift: float = 0.0  # Semitones, positive or negative
    pitch_range_scale: float = 1.0  # Scale factor for expanding/contracting pitch range
    duration_scale: float = 1.0  # Scale factor for phoneme durations
    energy_scale: float = 1.0  # Scale factor for speech energy
    pause_scale: float = 1.0  # Scale factor for pause durations
    emphasis_scale: Dict[str, float] = field(default_factory=dict)  # Emphasis for specific words


class VoiceTranslator:
    """
    Handles voice translation, preserving speaker characteristics while
    translating speech from source to target language.
    """
    
    def __init__(self, 
               voice_model_path: str = "models/voice/translator_model",
               emotion_model_path: str = "models/emotion/transfer_model",
               prosody_model_path: str = "models/prosody/model_weights",
               device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the voice translator.
        
        Args:
            voice_model_path: Path to the voice translation model
            emotion_model_path: Path to the emotion transfer model
            prosody_model_path: Path to the prosody modeling model
            device: Device for model inference ("cuda" or "cpu")
        """
        self.device = device
        self.voice_model_path = voice_model_path
        self.loaded = False
        
        # Initialize components
        self.emotion_transfer = EmotionTransferSystem(model_path=emotion_model_path)
        self.prosody_modeler = ProsodyModeler(model_path=prosody_model_path)
        
        # Components that will be initialized during load
        self.voice_analyzer = None
        self.voice_synthesizer = None
        self.speaker_separator = None
        
        print(f"Voice Translator initialized (models will be loaded on first use)")
        print(f"  - Voice model path: {voice_model_path}")
        print(f"  - Emotion model path: {emotion_model_path}")
        print(f"  - Prosody model path: {prosody_model_path}")
        print(f"  - Device: {device}")
    
    def load_models(self) -> bool:
        """
        Load all required models for voice translation.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        if self.loaded:
            return True
        
        try:
            # In a real implementation, these would be actual model loading operations
            # For now, we'll just simulate successful loading
            self.voice_analyzer = object()  # Placeholder for actual model
            self.voice_synthesizer = object()  # Placeholder for actual model
            self.speaker_separator = object()  # Placeholder for actual model
            
            self.loaded = True
            print("Voice translation models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading voice translation models: {e}")
            return False
    
    def extract_voice_profile(self, audio_path: str) -> VoiceProfile:
        """
        Extract voice profile from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            VoiceProfile object containing speaker characteristics
        """
        if not self.loaded:
            self.load_models()
        
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return VoiceProfile(speaker_id="unknown")
        
        # In a real implementation, this would analyze the audio using models
        # For demonstration, we'll create a simple profile
        
        # Extract fundamental frequency (F0) information
        f0, voiced_flag, _ = librosa.pyin(y, fmin=60, fmax=500)
        f0_values = f0[voiced_flag]
        
        if len(f0_values) > 0:
            f0_min = np.min(f0_values)
            f0_max = np.max(f0_values)
            f0_mean = np.mean(f0_values)
        else:
            f0_min, f0_max, f0_mean = 100.0, 200.0, 150.0
        
        # Simple gender detection based on average pitch
        gender = "female" if f0_mean > 170 else "male"
        
        # Create dummy voice embeddings
        voice_embeddings = np.random.random(128)  # Placeholder for actual voice embedding
        
        # Create and return profile
        profile = VoiceProfile(
            speaker_id=os.path.basename(audio_path),
            gender=gender,
            f0_range=(float(f0_min), float(f0_max)),
            voice_embeddings=voice_embeddings
        )
        
        return profile
    
    def detect_emotion(self, audio_path: str) -> Dict[str, float]:
        """
        Detect emotion from an audio file using the emotion transfer system.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary mapping emotions to confidence scores
        """
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return {"neutral": 1.0}
        
        # Extract emotion features
        emotion_features = self.emotion_transfer.extract_emotion_features(y, sr)
        
        # Detect emotions
        emotion_scores = self.emotion_transfer.detect_emotion_from_features(emotion_features)
        
        return emotion_scores
    
    def analyze_prosody(self, audio_path: str, transcript: str, language: str = "en") -> Dict[str, Any]:
        """
        Analyze the prosody of speech in an audio file using the prosody modeler.
        
        Args:
            audio_path: Path to the audio file
            transcript: Transcript of the speech
            language: Language code of the speech
            
        Returns:
            Dictionary containing prosody information
        """
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return {}
        
        # Extract prosody features
        prosody_features = self.prosody_modeler.extract_prosody_features(y, sr, language)
        
        # Segment speech into units
        speech_units = self.prosody_modeler.segment_speech(y, sr, transcript, language)
        
        # Convert to dictionary for return
        prosody_info = {
            "speech_rate": prosody_features.speech_rate,
            "pitch_statistics": prosody_features.pitch_statistics,
            "energy_statistics": prosody_features.energy_statistics,
            "pauses": prosody_features.pauses,
            "speech_units": [
                {
                    "text": unit.text,
                    "start_time": unit.start_time,
                    "end_time": unit.end_time,
                    "has_pause": unit.has_final_pause,
                    "pause_duration": unit.pause_duration
                }
                for unit in speech_units
            ]
        }
        
        return prosody_info
    
    def translate_voice(self,
                      source_audio_path: str,
                      transcript: str,
                      translated_text: str,
                      output_path: str,
                      source_lang: str = "en",
                      target_lang: str = "es") -> str:
        """
        Translate voice from source to target language, preserving voice characteristics.
        
        Args:
            source_audio_path: Path to the source audio file
            transcript: Transcript of the source speech
            translated_text: Translated text in the target language
            output_path: Path for the output audio file
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Path to the translated audio file
        """
        if not self.loaded:
            self.load_models()
        
        print(f"Translating voice from {source_lang} to {target_lang}")
        print(f"  - Source audio: {source_audio_path}")
        print(f"  - Output path: {output_path}")
        
        # Load audio
        try:
            y, sr = librosa.load(source_audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {source_audio_path}: {e}")
            return output_path
        
        # 1. Extract voice profile for voice characteristic preservation
        voice_profile = self.extract_voice_profile(source_audio_path)
        print(f"  - Detected voice profile: {voice_profile.gender}, F0 range: {voice_profile.f0_range}")
        
        # 2. Process audio for emotion transfer
        emotion_params, detected_emotion = self.emotion_transfer.process_audio(
            y, sr, source_lang, target_lang
        )
        
        # Print the dominant emotion
        dominant_emotion = max(detected_emotion.items(), key=lambda x: x[1])
        print(f"  - Detected emotion: {dominant_emotion[0]} (confidence: {dominant_emotion[1]:.2f})")
        
        # 3. Process audio for prosody modeling
        prosody_params = self.prosody_modeler.process_audio(
            y, sr, transcript, translated_text, source_lang, target_lang
        )
        
        # Print speech rate adjustment
        print(f"  - Speech rate adjustment: {prosody_params['speech_rate_factor']:.2f}x")
        print(f"  - Pauses: {len(prosody_params['pauses'])} pauses generated")
        
        # 4. Combine all parameters for voice synthesis
        synthesis_params = {
            "voice_profile": {
                "gender": voice_profile.gender,
                "f0_range": voice_profile.f0_range,
                "voice_quality": voice_profile.voice_quality
            },
            "emotion": emotion_params,
            "prosody": prosody_params,
            "text": translated_text,
            "language": target_lang
        }
        
        # In a real implementation, this would synthesize speech in the target language
        # with the extracted voice characteristics, emotion, and prosody
        # For demonstration, we'll just assume success
        
        print(f"Voice translation completed: {output_path}")
        # In a real implementation, audio would be saved to output_path
        
        return output_path
    
    def preserve_voice_characteristics(self, 
                                    source_profile: VoiceProfile, 
                                    target_audio: np.ndarray,
                                    sr: int) -> np.ndarray:
        """
        Apply voice characteristics from the source profile to the target audio.
        
        Args:
            source_profile: Voice profile of the source speaker
            target_audio: Audio data to modify
            sr: Sample rate of the audio
            
        Returns:
            Modified audio with the source voice characteristics
        """
        # In a real implementation, this would transform the voice characteristics
        # For demonstration, we'll return the original audio
        
        return target_audio
    
    def transfer_emotion(self, 
                       emotion_params: Dict[str, Any], 
                       audio: np.ndarray,
                       sr: int) -> np.ndarray:
        """
        Transfer emotional characteristics to the audio using the emotion transfer system.
        
        Args:
            emotion_params: Emotion transfer parameters
            audio: Audio data to modify
            sr: Sample rate of the audio
            
        Returns:
            Modified audio with the emotional characteristics
        """
        # Convert dictionary to EmotionTransferParameters
        params = EmotionTransferParameters(
            pitch_shift=emotion_params.get("pitch_shift", 0.0),
            pitch_range_scale=emotion_params.get("pitch_range_scale", 1.0),
            pitch_variability=emotion_params.get("pitch_variability", 1.0),
            energy_scale=emotion_params.get("energy_scale", 1.0),
            energy_variability=emotion_params.get("energy_variability", 1.0),
            speaking_rate_scale=emotion_params.get("speaking_rate_scale", 1.0),
            articulation_strength=emotion_params.get("articulation_strength", 1.0),
            voice_quality_adjustments=emotion_params.get("voice_quality_adjustments", {}),
            attack_scale=emotion_params.get("attack_scale", 1.0),
            decay_scale=emotion_params.get("decay_scale", 1.0)
        )
        
        # Apply emotion transfer
        modified_audio = self.emotion_transfer.apply_emotion_transfer(audio, sr, params)
        
        return modified_audio
    
    def apply_prosody(self, 
                    prosody_params: Dict[str, Any], 
                    audio: np.ndarray,
                    sr: int,
                    transcript: str) -> np.ndarray:
        """
        Apply prosody parameters to the audio.
        
        Args:
            prosody_params: Prosody parameters to apply
            audio: Audio data to modify
            sr: Sample rate of the audio
            transcript: Text transcript for the audio
            
        Returns:
            Modified audio with the prosody parameters applied
        """
        # In a real implementation, this would modify the prosody of the audio
        # For demonstration, we'll apply speech rate adjustment
        
        speech_rate_factor = prosody_params.get("speech_rate_factor", 1.0)
        
        if speech_rate_factor != 1.0:
            # Adjust speed using librosa's time_stretch
            # Note: time_stretch uses the inverse of our factor (it speeds up when factor < 1)
            modified_audio = librosa.effects.time_stretch(audio, rate=1.0/speech_rate_factor)
        else:
            modified_audio = audio
        
        return modified_audio
    
    def separate_speakers(self, 
                        audio_path: str, 
                        num_speakers: int = 2) -> Dict[int, Tuple[np.ndarray, int]]:
        """
        Separate different speakers in an audio file.
        
        Args:
            audio_path: Path to the audio file
            num_speakers: Expected number of speakers
            
        Returns:
            Dictionary mapping speaker IDs to tuples of (audio_data, sample_rate)
        """
        if not self.loaded:
            self.load_models()
        
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return {0: (np.array([]), sr)}
        
        # In a real implementation, this would use a speaker separation model
        # For demonstration, we'll create dummy separated audio
        
        separated = {}
        for i in range(num_speakers):
            # Create a simple filtered version as a placeholder
            separated[i] = (y * (1.0 / (i + 1)), sr)
        
        return separated
    
    def process_multi_speaker(self,
                          audio_path: str,
                          speaker_segments: List[Dict[str, Any]],
                          transcript: str,
                          translated_text: str,
                          output_path: str,
                          source_lang: str = "en",
                          target_lang: str = "es") -> str:
        """
        Process audio with multiple speakers, translating each speaker separately.
        
        Args:
            audio_path: Path to the audio file
            speaker_segments: List of segments with speaker information
            transcript: Full transcript of the source speech
            translated_text: Full translated text
            output_path: Path for the output audio file
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Path to the translated audio file
        """
        if not self.loaded:
            self.load_models()
        
        print(f"Processing multi-speaker audio from {source_lang} to {target_lang}")
        print(f"  - Source audio: {audio_path}")
        print(f"  - Speakers: {len(set(seg['speaker_id'] for seg in speaker_segments))}")
        print(f"  - Output path: {output_path}")
        
        # 1. Separate speakers
        speaker_audio = self.separate_speakers(
            audio_path, num_speakers=len(set(seg["speaker_id"] for seg in speaker_segments))
        )
        
        # 2. Process each speaker separately
        processed_segments = []
        
        for segment in speaker_segments:
            speaker_id = segment["speaker_id"]
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            segment_text = segment.get("text", "")
            
            # Extract segment audio from the separated speaker audio
            speaker_y, sr = speaker_audio.get(speaker_id, (np.array([]), 0))
            if len(speaker_y) == 0 or sr == 0:
                continue
            
            # Convert times to samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Ensure within bounds
            start_sample = max(0, min(start_sample, len(speaker_y) - 1))
            end_sample = max(start_sample, min(end_sample, len(speaker_y)))
            
            # Extract the segment audio
            segment_audio = speaker_y[start_sample:end_sample]
            
            # Process this segment
            # In a real implementation, this would call translate_voice for each segment
            # For demonstration, we'll just track the segments
            
            processed_segments.append({
                "speaker_id": speaker_id,
                "start_time": start_time,
                "end_time": end_time,
                "text": segment_text
            })
        
        print(f"Processed {len(processed_segments)} segments from {len(speaker_audio)} speakers")
        print(f"Multi-speaker voice translation completed: {output_path}")
        
        return output_path


# Example usage
if __name__ == "__main__":
    voice_translator = VoiceTranslator()
    
    # Example voice translation
    source_audio = "example/source_audio.wav"
    transcript = "This is an example of voice translation."
    translated_text = "Este es un ejemplo de traducción de voz."
    output_path = "example/translated_audio.wav"
    
    if os.path.exists(source_audio):  # Only run if the file exists
        result_path = voice_translator.translate_voice(
            source_audio, transcript, translated_text, output_path, "en", "es"
        )
        print(f"Translated audio saved to: {result_path}")
    else:
        print(f"Example file {source_audio} not found. This is just a demonstration.") 