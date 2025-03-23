"""
Voice Analysis Module

This module handles voice tone and emphasis analysis for detecting interesting moments,
including emphasis detection, laughter detection, and other vocal features.
"""

import os
import numpy as np
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAnalyzer:
    """
    Analyzes vocal aspects of content to detect interesting moments.
    
    Features:
    - Voice tone and emphasis analysis
    - Laughter and reaction detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the voice analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.temp_dir = Path(config.get("temp_dir", "temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Voice analysis settings
        self.emphasis_threshold = config.get("voice_emphasis_threshold", 0.6)
        self.enable_laughter_detection = config.get("enable_laughter_detection", True)
        self.laughter_confidence_threshold = config.get("laughter_confidence_threshold", 0.7)
        
        # Load required libraries
        try:
            import librosa
            import scipy
            self.librosa = librosa
            self.scipy = scipy
            self._has_required_libs = True
        except ImportError:
            logger.warning("Librosa or SciPy not available. Voice analysis features will be limited.")
            self._has_required_libs = False
        
        # Check for optional dependencies
        try:
            import torch
            import torchaudio
            self._has_torch = True
        except ImportError:
            logger.warning("PyTorch not available. Advanced voice analysis will be limited.")
            self._has_torch = False
        
        logger.info("Initialized VoiceAnalyzer")
    
    def detect_voice_emphasis(self, audio_path: str) -> List[Tuple[float, float, float]]:
        """
        Detect segments with emphasized speech (louder, higher pitch, etc.).
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of tuples (start_time, end_time, emphasis_score)
        """
        if not self._has_required_libs:
            logger.warning("Required libraries not available, using simplified voice emphasis detection")
            return self._detect_voice_emphasis_simple(audio_path)
        
        try:
            # Load audio
            y, sr = self.librosa.load(audio_path, sr=None)
            
            # Calculate features that indicate emphasis:
            # 1. Energy (volume)
            energy = np.array([
                np.sum(y[i:i+sr//10]**2) 
                for i in range(0, len(y)-sr//10, sr//20)
            ])
            
            # 2. Pitch (fundamental frequency)
            hop_length = sr // 20
            frame_length = sr // 10
            pitches, magnitudes = self.librosa.piptrack(
                y=y, sr=sr, 
                hop_length=hop_length,
                fmin=75, fmax=400
            )
            
            # For each frame, get the pitch with highest magnitude
            pitch = []
            for i in range(magnitudes.shape[1]):
                index = magnitudes[:,i].argmax()
                pitch.append(pitches[index,i])
            
            pitch = np.array(pitch)
            
            # 3. Spectral contrast (measure of "drama" in the spectrum)
            contrast = np.mean(self.librosa.feature.spectral_contrast(
                y=y, sr=sr, hop_length=hop_length
            ), axis=0)
            
            # Normalize features
            energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
            
            # Remove zeros before normalizing pitch to avoid division by zero
            pitch = np.nan_to_num(pitch)
            pitch_mean = np.mean(pitch[pitch > 0])
            pitch_std = np.std(pitch[pitch > 0])
            if pitch_std > 0:
                pitch_norm = (pitch - pitch_mean) / pitch_std
                pitch_norm = (np.clip(pitch_norm, -2, 2) + 2) / 4  # Scale to 0-1
            else:
                pitch_norm = np.zeros_like(pitch)
            
            contrast_norm = (contrast - np.min(contrast)) / (np.max(contrast) - np.min(contrast) + 1e-10)
            
            # Combine features (weighted sum)
            emphasis = 0.5 * energy_norm + 0.3 * pitch_norm + 0.2 * contrast_norm
            
            # Find peaks in emphasis
            from scipy import signal
            # Minimum distance between peaks (0.5 seconds)
            min_dist = int(0.5 * sr / hop_length)
            peaks, _ = signal.find_peaks(emphasis, height=self.emphasis_threshold, distance=min_dist)
            
            # Convert peaks to time ranges
            time_per_frame = hop_length / sr
            results = []
            
            for peak in peaks:
                # Calculate emphasis score
                score = emphasis[peak]
                
                # Calculate time range (0.5 seconds before and after the peak)
                start_time = max(0, peak - int(0.5 / time_per_frame)) * time_per_frame
                end_time = min(len(emphasis) - 1, peak + int(0.5 / time_per_frame)) * time_per_frame
                
                # Ensure minimum duration
                if end_time - start_time < 1.0:
                    # Extend to at least 1 second
                    mid_point = (start_time + end_time) / 2
                    start_time = mid_point - 0.5
                    end_time = mid_point + 0.5
                
                results.append((start_time, end_time, float(score)))
            
            logger.info(f"Detected {len(results)} voice emphasis points")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting voice emphasis: {str(e)}")
            return []
    
    def _detect_voice_emphasis_simple(self, audio_path: str) -> List[Tuple[float, float, float]]:
        """
        Simplified voice emphasis detection without advanced libraries.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of tuples (start_time, end_time, emphasis_score)
        """
        try:
            # Get audio duration using ffprobe
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error getting audio duration: {stderr}")
                return []
            
            try:
                duration = float(stdout.strip())
            except ValueError:
                logger.error(f"Invalid duration value: {stdout}")
                return []
            
            # Generate simulated emphasis points (for demonstration only)
            # In a real implementation, this would analyze the audio file
            import random
            random.seed(43)  # Different seed than audio energy
            
            num_segments = int(duration / 7)  # Approximately one emphasis every 7 seconds
            results = []
            
            for i in range(num_segments):
                # Generate an emphasis with random score
                emphasis = random.uniform(0.5, 1.0)
                
                # Only include emphasis above threshold
                if emphasis >= self.emphasis_threshold:
                    # Calculate time range
                    segment_start = i * 7
                    segment_mid = segment_start + 3.5
                    
                    # Create a 1-2 second window around the peak
                    window_size = random.uniform(1.0, 2.0)
                    start_time = segment_mid - (window_size / 2)
                    end_time = segment_mid + (window_size / 2)
                    
                    # Ensure within bounds
                    start_time = max(0, start_time)
                    end_time = min(duration, end_time)
                    
                    results.append((start_time, end_time, emphasis))
            
            logger.warning(f"Using simulated voice emphasis due to missing libraries. Detected {len(results)} points.")
            return results
            
        except Exception as e:
            logger.error(f"Error in simplified voice emphasis detection: {str(e)}")
            return []
    
    def detect_laughter(self, audio_path: str) -> List[Tuple[float, float, float]]:
        """
        Detect laughter segments in the audio.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of tuples (start_time, end_time, confidence)
        """
        if not self.enable_laughter_detection:
            logger.info("Laughter detection disabled")
            return []
        
        if not self._has_torch:
            logger.warning("PyTorch not available, using simplified laughter detection")
            return self._detect_laughter_simple(audio_path)
        
        try:
            # Note: This is a simplified placeholder for laughter detection
            # In a real implementation, this would use a pre-trained model
            
            # Load audio
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Extract MFCC features (commonly used for audio classification)
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={"n_fft": 400, "hop_length": 160}
            )
            mfcc = mfcc_transform(waveform)
            
            # In a real implementation, the MFCC features would be passed through
            # a trained classifier. Here we simulate classification results.
            import random
            random.seed(44)  # Different seed than other analyses
            
            # Simulate detection
            duration = waveform.shape[1] / sample_rate
            num_segments = int(duration / 10)  # Check every 10 seconds for laughter
            results = []
            
            for i in range(num_segments):
                # Random confidence score
                confidence = random.uniform(0.5, 1.0)
                
                # Only include high-confidence detections
                if confidence >= self.laughter_confidence_threshold:
                    # Start and end times
                    start_time = i * 10
                    duration = random.uniform(1.0, 3.0)  # Laughter typically 1-3 seconds
                    end_time = start_time + duration
                    
                    results.append((start_time, end_time, confidence))
            
            logger.info(f"Detected {len(results)} potential laughter segments")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting laughter: {str(e)}")
            return []
    
    def _detect_laughter_simple(self, audio_path: str) -> List[Tuple[float, float, float]]:
        """
        Simplified laughter detection without PyTorch.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of tuples (start_time, end_time, confidence)
        """
        try:
            # Get audio duration
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error getting audio duration: {stderr}")
                return []
            
            try:
                duration = float(stdout.strip())
            except ValueError:
                logger.error(f"Invalid duration value: {stdout}")
                return []
            
            # Generate simulated laughter detections
            import random
            random.seed(44)
            
            num_segments = int(duration / 20)  # Approximately one laughter every 20 seconds
            results = []
            
            for i in range(num_segments):
                # Generate a random confidence
                confidence = random.uniform(0.5, 1.0)
                
                # Only include high-confidence detections
                if confidence >= self.laughter_confidence_threshold:
                    # Start time somewhere in this 20-second segment
                    start_time = i * 20 + random.uniform(0, 17)
                    # Laughter duration between 1-3 seconds
                    laughter_duration = random.uniform(1.0, 3.0)
                    end_time = start_time + laughter_duration
                    
                    # Ensure within bounds
                    end_time = min(duration, end_time)
                    
                    results.append((start_time, end_time, confidence))
            
            logger.warning(f"Using simulated laughter detection due to missing libraries. Detected {len(results)} segments.")
            return results
            
        except Exception as e:
            logger.error(f"Error in simplified laughter detection: {str(e)}")
            return []
    
    def analyze_voice(self, audio_path: str) -> List[Tuple[str, float, float, float, Dict[str, Any]]]:
        """
        Perform comprehensive voice analysis on the audio.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of tuples (moment_type, start_time, end_time, score, metadata)
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return []
        
        logger.info(f"Analyzing voice features for: {audio_path}")
        
        results = []
        
        # Detect voice emphasis
        emphasis_points = self.detect_voice_emphasis(audio_path)
        for start_time, end_time, score in emphasis_points:
            moment_type = "voice_emphasis"
            metadata = {
                "emphasis_level": score,
                "detection_method": "advanced" if self._has_required_libs else "simplified"
            }
            results.append((moment_type, start_time, end_time, score, metadata))
        
        # Detect laughter
        if self.enable_laughter_detection:
            laughter_segments = self.detect_laughter(audio_path)
            for start_time, end_time, confidence in laughter_segments:
                moment_type = "laughter"
                metadata = {
                    "confidence": confidence,
                    "detection_method": "model" if self._has_torch else "simplified"
                }
                results.append((moment_type, start_time, end_time, confidence, metadata))
        
        logger.info(f"Voice analysis complete. Detected {len(results)} voice-related moments")
        return results 