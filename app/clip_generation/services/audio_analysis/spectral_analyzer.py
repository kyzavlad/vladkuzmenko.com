"""
Spectral Analysis Module

This module provides spectral analysis capabilities for audio processing,
including background noise profiling and non-speech sound classification.
"""

import os
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import tempfile
from enum import Enum
from pathlib import Path

from app.clip_generation.services.audio_analysis.audio_analyzer import AudioSegment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SoundType(str, Enum):
    """Sound type classification."""
    SPEECH = "speech"
    SILENCE = "silence"
    MUSIC = "music"
    AMBIENT = "ambient"
    NOISE = "noise"
    EFFECT = "effect"  # Sound effects
    UNKNOWN = "unknown"


@dataclass
class SpectralProfile:
    """Spectral profile for an audio segment."""
    # Basic properties
    energy_db: float  # RMS energy in dB
    spectral_centroid: float  # Brightness of sound
    spectral_bandwidth: float  # Width of frequency distribution
    spectral_rolloff: float  # Roll-off frequency
    zero_crossing_rate: float  # Number of zero-crossings
    
    # Advanced properties
    mfcc: List[float]  # Mel-frequency cepstral coefficients
    spectral_flatness: float  # Flatness (tonal vs. noise)
    spectral_contrast: List[float]  # Contrast between peaks and valleys
    
    # Classification
    sound_type: SoundType  # Classification of sound
    confidence: float  # Confidence in classification (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "energy_db": self.energy_db,
            "spectral_centroid": self.spectral_centroid,
            "spectral_bandwidth": self.spectral_bandwidth,
            "spectral_rolloff": self.spectral_rolloff,
            "zero_crossing_rate": self.zero_crossing_rate,
            "mfcc": self.mfcc,
            "spectral_flatness": self.spectral_flatness,
            "spectral_contrast": self.spectral_contrast,
            "sound_type": self.sound_type,
            "confidence": self.confidence
        }


class SpectralAnalyzer:
    """
    Analyzes spectral characteristics of audio.
    
    This class provides methods for:
    - Background noise profiling
    - Spectral feature extraction
    - Non-speech sound classification (music, ambient, effects)
    - Adaptive thresholding for silence detection
    """
    
    def __init__(
        self,
        n_mfcc: int = 13,  # Number of MFCC coefficients
        frame_length: int = 2048,  # Frame length for analysis
        hop_length: int = 512,  # Hop length for analysis
        n_fft: int = 2048,  # FFT size
        sample_rate: int = 44100,  # Default sample rate
        visualize: bool = False,  # Generate visualizations
        model_dir: Optional[str] = None  # Directory for models
    ):
        """
        Initialize the spectral analyzer.
        
        Args:
            n_mfcc: Number of MFCC coefficients to extract
            frame_length: Frame length for spectral analysis
            hop_length: Hop length for spectral analysis
            n_fft: FFT size
            sample_rate: Default sample rate
            visualize: Generate visualization plots
            model_dir: Directory for sound classification models
        """
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.visualize = visualize
        
        # Set up model directory
        self.model_dir = Path(model_dir) if model_dir else None
        
        # Sound classifier model (lazy loaded)
        self._classifier = None
        
        # Noise profile (computed from analysis)
        self.noise_profile = None
        
        logger.info(f"Initialized SpectralAnalyzer with frame_length={frame_length}, hop_length={hop_length}")
    
    def _load_classifier(self) -> bool:
        """
        Load sound classifier model.
        
        Returns:
            True if successful, False otherwise
        """
        if self._classifier is not None:
            return True
        
        try:
            import torch
            import torchaudio
            
            logger.info("Loading sound classifier model")
            
            # Load YAMNet-like model or PANNs for sound classification
            # This is a placeholder - in a real implementation we would
            # load an actual sound classification model
            
            # Example with torch hub (would use a real model in production)
            # self._classifier = torch.hub.load('harritaylor/torchvggish', 'vggish')
            
            logger.warning("Using mock sound classifier (real model not implemented)")
            
            # Create a mock classifier for now
            class MockClassifier:
                def __call__(self, audio, sample_rate):
                    # Return mock classification
                    return {
                        "type": SoundType.UNKNOWN,
                        "confidence": 0.5
                    }
            
            self._classifier = MockClassifier()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading sound classifier: {str(e)}")
            return False
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract spectral features from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary of spectral features
        """
        try:
            import librosa
            
            # Adjust sample rate if needed
            if sample_rate != self.sample_rate:
                logger.info(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz for analysis")
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
                sample_rate = self.sample_rate
            
            logger.info(f"Extracting spectral features from {len(audio_data)/sample_rate:.2f}s audio")
            
            # Basic features
            # RMS energy
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            energy_db = 20 * np.log10(np.mean(rms) + 1e-10)
            
            # Spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio_data,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            
            # Advanced features
            # MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(
                y=audio_data,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Package results
            features = {
                "energy_db": float(energy_db),
                "spectral_centroid": float(np.mean(centroid)),
                "spectral_bandwidth": float(np.mean(bandwidth)),
                "spectral_rolloff": float(np.mean(rolloff)),
                "zero_crossing_rate": float(np.mean(zcr)),
                "mfcc": [float(np.mean(coef)) for coef in mfcc],
                "spectral_flatness": float(np.mean(flatness)),
                "spectral_contrast": [float(np.mean(band)) for band in contrast],
                "energy_stats": {
                    "mean": float(np.mean(rms)),
                    "std": float(np.std(rms)),
                    "min": float(np.min(rms)),
                    "max": float(np.max(rms)),
                    "percentiles": {
                        "10": float(np.percentile(rms, 10)),
                        "25": float(np.percentile(rms, 25)),
                        "50": float(np.percentile(rms, 50)),
                        "75": float(np.percentile(rms, 75)),
                        "90": float(np.percentile(rms, 90)),
                    }
                }
            }
            
            # Visualize if requested
            if self.visualize:
                self._visualize_spectral(audio_data, sample_rate, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {str(e)}")
            return {}
    
    def _visualize_spectral(self, audio_data: np.ndarray, sample_rate: int, features: Dict[str, Any]) -> None:
        """
        Generate visualization of spectral features.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            features: Extracted features
        """
        try:
            import matplotlib.pyplot as plt
            import librosa.display
            
            # Create temporary directory for visualizations
            temp_dir = Path(tempfile.gettempdir()) / "clip_generation" / "spectral_visualization"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up figure
            plt.figure(figsize=(12, 8))
            
            # Plot waveform
            plt.subplot(3, 1, 1)
            librosa.display.waveshow(audio_data, sr=sample_rate)
            plt.title("Waveform")
            
            # Plot spectrogram
            plt.subplot(3, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
            librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Spectrogram")
            
            # Plot MFCCs
            plt.subplot(3, 1, 3)
            librosa.display.specshow(
                librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=self.n_mfcc),
                sr=sample_rate, x_axis='time'
            )
            plt.colorbar()
            plt.title("MFCCs")
            
            plt.tight_layout()
            
            # Save figure
            timestamp = int(time.time())
            output_path = temp_dir / f"spectral_analysis_{timestamp}.png"
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Spectral visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing spectral features: {str(e)}")
    
    def profile_background_noise(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Create a profile of background noise in the audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with noise profile
        """
        try:
            import librosa
            
            logger.info("Profiling background noise")
            
            # Extract RMS energy
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            
            # Convert to dB
            db = 20 * np.log10(rms + 1e-10)
            
            # Find quietest segments (10th percentile of energy)
            threshold_db = np.percentile(db, 10)
            quiet_mask = db < threshold_db
            
            # If no quiet segments found, use the lowest 10% of frames
            if not np.any(quiet_mask):
                n_frames = len(db)
                n_quiet_frames = max(1, int(n_frames * 0.1))
                quiet_indices = np.argsort(db)[:n_quiet_frames]
                quiet_mask = np.zeros_like(db, dtype=bool)
                quiet_mask[quiet_indices] = True
            
            # Extract quietest segments
            quiet_frames = []
            for i, is_quiet in enumerate(quiet_mask):
                if is_quiet:
                    start_sample = i * self.hop_length
                    end_sample = start_sample + self.frame_length
                    if end_sample <= len(audio_data):
                        quiet_frames.append(audio_data[start_sample:end_sample])
            
            # If no frames extracted, fallback to full audio
            if not quiet_frames:
                logger.warning("No quiet frames found for noise profiling")
                noise_audio = audio_data
            else:
                # Concatenate quiet frames
                noise_audio = np.concatenate(quiet_frames)
            
            # Extract spectral features from noise audio
            noise_features = self.extract_features(noise_audio, sample_rate)
            
            # Create noise profile
            noise_profile = {
                "noise_floor_db": float(threshold_db),
                "features": noise_features,
                "energy_stats": noise_features.get("energy_stats", {}),
                "spectral_stats": {
                    "flatness": noise_features.get("spectral_flatness", 0),
                    "centroid": noise_features.get("spectral_centroid", 0),
                    "bandwidth": noise_features.get("spectral_bandwidth", 0),
                    "zero_crossing_rate": noise_features.get("zero_crossing_rate", 0)
                }
            }
            
            # Store noise profile for later use
            self.noise_profile = noise_profile
            
            logger.info(f"Noise floor: {threshold_db:.2f} dB")
            
            return noise_profile
            
        except Exception as e:
            logger.error(f"Error profiling background noise: {str(e)}")
            return {"noise_floor_db": -60.0}  # Default fallback
    
    def suggest_silence_threshold(self, noise_profile: Optional[Dict[str, Any]] = None) -> float:
        """
        Suggest appropriate silence threshold based on noise profile.
        
        Args:
            noise_profile: Noise profile (if None, uses stored profile)
            
        Returns:
            Recommended silence threshold in dB
        """
        profile = noise_profile or self.noise_profile
        
        if not profile:
            logger.warning("No noise profile available, using default threshold")
            return -35.0  # Default threshold
        
        # Get noise floor
        noise_floor_db = profile.get("noise_floor_db", -60.0)
        
        # Set threshold above noise floor (typically 10-15 dB above)
        threshold_db = noise_floor_db + 10.0
        
        logger.info(f"Suggested silence threshold: {threshold_db:.2f} dB (noise floor: {noise_floor_db:.2f} dB)")
        
        return threshold_db
    
    def classify_sound(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[SoundType, float]:
        """
        Classify non-speech sounds in audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (sound_type, confidence)
        """
        try:
            # Load classifier if needed
            if not self._load_classifier():
                logger.warning("Sound classifier not available, returning unknown type")
                return SoundType.UNKNOWN, 0.0
            
            # Extract features
            features = self.extract_features(audio_data, sample_rate)
            
            # Simple heuristic classification based on features
            # A more robust approach would use a trained classifier
            energy_db = features.get("energy_db", -100)
            zcr = features.get("zero_crossing_rate", 0)
            flatness = features.get("spectral_flatness", 0)
            centroid = features.get("spectral_centroid", 0)
            
            # Very low energy is silence
            if energy_db < -50:
                return SoundType.SILENCE, 0.9
            
            # High ZCR with high centroid often indicates noise
            if zcr > 0.15 and centroid > 3000:
                return SoundType.NOISE, 0.7
            
            # Low flatness often indicates music (more tonal)
            if flatness < 0.2 and energy_db > -25:
                return SoundType.MUSIC, 0.6
            
            # Moderate energy with moderate flatness might be ambient
            if -40 < energy_db < -20 and 0.2 < flatness < 0.6:
                return SoundType.AMBIENT, 0.5
            
            # Use classifier for more accurate results
            # In a real implementation, this would use the loaded ML model
            result = self._classifier(audio_data, sample_rate)
            sound_type = result.get("type", SoundType.UNKNOWN)
            confidence = result.get("confidence", 0.5)
            
            return sound_type, confidence
            
        except Exception as e:
            logger.error(f"Error classifying sound: {str(e)}")
            return SoundType.UNKNOWN, 0.0
    
    def segment_and_classify(self, audio_data: np.ndarray, sample_rate: int, 
                            segment_length_ms: int = 1000) -> List[AudioSegment]:
        """
        Segment audio and classify each segment.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            segment_length_ms: Length of each segment in milliseconds
            
        Returns:
            List of AudioSegment objects with classifications
        """
        try:
            # First profile background noise
            noise_profile = self.profile_background_noise(audio_data, sample_rate)
            silence_threshold = self.suggest_silence_threshold(noise_profile)
            
            # Calculate segment size in samples
            segment_length_samples = int(sample_rate * segment_length_ms / 1000)
            
            # Calculate number of segments
            num_segments = len(audio_data) // segment_length_samples
            
            segments = []
            
            # Process each segment
            for i in range(num_segments):
                start_sample = i * segment_length_samples
                end_sample = start_sample + segment_length_samples
                
                # Extract segment audio
                segment_audio = audio_data[start_sample:end_sample]
                
                # Calculate segment times
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                
                # Extract features
                features = self.extract_features(segment_audio, sample_rate)
                
                # Classify sound
                sound_type, confidence = self.classify_sound(segment_audio, sample_rate)
                
                # Create profile
                profile = SpectralProfile(
                    energy_db=features.get("energy_db", -100),
                    spectral_centroid=features.get("spectral_centroid", 0),
                    spectral_bandwidth=features.get("spectral_bandwidth", 0),
                    spectral_rolloff=features.get("spectral_rolloff", 0),
                    zero_crossing_rate=features.get("zero_crossing_rate", 0),
                    mfcc=features.get("mfcc", [0] * self.n_mfcc),
                    spectral_flatness=features.get("spectral_flatness", 0),
                    spectral_contrast=features.get("spectral_contrast", [0]),
                    sound_type=sound_type,
                    confidence=confidence
                )
                
                # Create segment
                segment = AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    segment_type=sound_type,
                    confidence=confidence,
                    metadata={
                        "spectral_profile": profile.to_dict(),
                        "energy_db": features.get("energy_db", -100),
                        "detection_source": "spectral_analysis"
                    }
                )
                
                segments.append(segment)
            
            # Handle final segment if needed
            if len(audio_data) % segment_length_samples > 0:
                start_sample = num_segments * segment_length_samples
                
                # Extract segment audio
                segment_audio = audio_data[start_sample:]
                
                # Calculate segment times
                start_time = start_sample / sample_rate
                end_time = len(audio_data) / sample_rate
                
                # Extract features and classify
                features = self.extract_features(segment_audio, sample_rate)
                sound_type, confidence = self.classify_sound(segment_audio, sample_rate)
                
                # Create profile
                profile = SpectralProfile(
                    energy_db=features.get("energy_db", -100),
                    spectral_centroid=features.get("spectral_centroid", 0),
                    spectral_bandwidth=features.get("spectral_bandwidth", 0),
                    spectral_rolloff=features.get("spectral_rolloff", 0),
                    zero_crossing_rate=features.get("zero_crossing_rate", 0),
                    mfcc=features.get("mfcc", [0] * self.n_mfcc),
                    spectral_flatness=features.get("spectral_flatness", 0),
                    spectral_contrast=features.get("spectral_contrast", [0]),
                    sound_type=sound_type,
                    confidence=confidence
                )
                
                # Create segment
                segment = AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    segment_type=sound_type,
                    confidence=confidence,
                    metadata={
                        "spectral_profile": profile.to_dict(),
                        "energy_db": features.get("energy_db", -100),
                        "detection_source": "spectral_analysis"
                    }
                )
                
                segments.append(segment)
            
            # Post-process segments
            # Merge adjacent segments of same type
            merged_segments = []
            if segments:
                current_segment = segments[0]
                
                for segment in segments[1:]:
                    if (segment.segment_type == current_segment.segment_type and
                        segment.start_time == current_segment.end_time):
                        # Merge by extending end time
                        current_segment.end_time = segment.end_time
                    else:
                        # Add current and start new
                        merged_segments.append(current_segment)
                        current_segment = segment
                
                # Add final segment
                merged_segments.append(current_segment)
            
            logger.info(f"Segmented and classified audio into {len(merged_segments)} segments")
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Error segmenting and classifying audio: {str(e)}")
            return [] 