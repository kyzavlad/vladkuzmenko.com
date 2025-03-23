"""
Noise Reduction Module

This module provides functionality for profiling and reducing background noise
in audio recordings using spectral subtraction and other advanced techniques.
"""

import os
import numpy as np
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import asyncio
import time

try:
    import librosa
    import soundfile as sf
    import noisereduce as nr
    from scipy import signal
    from pydub import AudioSegment
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

logger = logging.getLogger(__name__)

class NoiseReducer:
    """
    Reduces background noise in audio using spectral subtraction and other techniques.
    
    This class provides methods for profiling noise characteristics and applying
    noise reduction algorithms to clean audio recordings. It supports both
    automatic noise detection and manual noise profile specification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the noise reducer.
        
        Args:
            config: Configuration options for noise reduction
        """
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("Required audio processing libraries not available. "
                           "Noise reduction will not work.")
        
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        
        # Noise reduction parameters
        self.reduction_strength = self.config.get('reduction_strength', 0.75)  # 0.0 to 1.0
        self.n_fft = self.config.get('n_fft', 2048)  # FFT window size
        self.win_length = self.config.get('win_length', None)  # Window length
        self.hop_length = self.config.get('hop_length', 512)  # Hop length
        self.n_std_thresh = self.config.get('n_std_thresh', 1.5)  # Standard deviation threshold multiplier
        self.freq_mask_smooth_hz = self.config.get('freq_mask_smooth_hz', 500)  # Frequency mask smoothing
        self.time_mask_smooth_ms = self.config.get('time_mask_smooth_ms', 50)  # Time mask smoothing
        self.chunk_size = self.config.get('chunk_size', 60)  # Process audio in chunks of N seconds
        self.padding = self.config.get('padding', 1)  # Padding in seconds to avoid boundary artifacts
        
        # Noise profile storage
        self.noise_profiles = {}
    
    async def analyze_noise_profile(
        self, 
        audio_path: str,
        start_time: float = 0.0,
        duration: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze and create a noise profile from a section of audio.
        
        Args:
            audio_path: Path to the audio file
            start_time: Start time in seconds to extract noise sample
            duration: Duration in seconds of noise sample
            
        Returns:
            Dictionary containing noise profile data
        """
        if not AUDIO_LIBS_AVAILABLE:
            return {"error": "Required audio libraries not available"}
        
        try:
            # Extract noise sample
            y, sr = await self._load_audio_segment(audio_path, start_time, duration)
            
            if y is None or sr is None:
                return {"error": "Failed to load audio file"}
            
            # Compute noise statistics
            noise_stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, 
                                     win_length=self.win_length)
            noise_spec = np.abs(noise_stft) ** 2
            
            # Calculate noise profile statistics
            noise_mean = np.mean(noise_spec, axis=1)
            noise_std = np.std(noise_spec, axis=1)
            
            # Calculate overall noise metrics
            noise_energy = np.mean(noise_mean)
            noise_variability = np.mean(noise_std) / (noise_energy + 1e-10)
            
            # FFT-based spectral analysis
            noise_fft = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            
            # Find dominant noise frequencies
            noise_threshold = np.percentile(noise_fft, 95)
            dominant_idxs = np.where(noise_fft > noise_threshold)[0]
            dominant_freqs = []
            
            if len(dominant_idxs) > 0:
                for idx in dominant_idxs:
                    dominant_freqs.append({
                        "frequency": freqs[idx],
                        "magnitude": float(noise_fft[idx]),
                        "relative_magnitude": float(noise_fft[idx] / np.max(noise_fft))
                    })
            
            # Create profile ID using a hash of the first few values
            profile_id = f"noise_profile_{hash(str(noise_mean[:10].tolist()))}"
            
            # Store the noise profile
            noise_profile = {
                "id": profile_id,
                "sample_rate": sr,
                "noise_mean": noise_mean.tolist(),
                "noise_std": noise_std.tolist(),
                "noise_energy": float(noise_energy),
                "noise_variability": float(noise_variability),
                "dominant_frequencies": dominant_freqs[:5],  # Store top 5 dominant frequencies
                "created_at": time.time()
            }
            
            # Save the profile for future use
            self.noise_profiles[profile_id] = {
                "mean": noise_mean,
                "std": noise_std,
                "sample_rate": sr
            }
            
            return noise_profile
        
        except Exception as e:
            logger.error(f"Error analyzing noise profile: {str(e)}")
            return {"error": f"Failed to analyze noise profile: {str(e)}"}
    
    async def reduce_noise(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        noise_profile_id: Optional[str] = None,
        noise_sample: Optional[Dict[str, Any]] = None,
        auto_detect: bool = True,
        strength: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Reduce noise in an audio file.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path to save the processed audio (if None, auto-generated)
            noise_profile_id: ID of a previously analyzed noise profile
            noise_sample: Dictionary with start_time and duration for a noise sample
            auto_detect: Whether to automatically detect noise in silent sections
            strength: Override the default noise reduction strength (0.0 to 1.0)
            
        Returns:
            Dictionary with result information including output_path
        """
        if not AUDIO_LIBS_AVAILABLE:
            return {"error": "Required audio libraries not available"}
        
        try:
            # Generate output path if not provided
            if output_path is None:
                output_dir = os.path.join(self.temp_dir, "noise_reduced")
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.basename(audio_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_noise_reduced{ext}")
            
            # Load the full audio file
            y, sr = await self._load_audio(audio_path)
            
            if y is None or sr is None:
                return {"error": "Failed to load audio file"}
            
            # Apply the appropriate noise reduction method
            if noise_profile_id and noise_profile_id in self.noise_profiles:
                # Use existing noise profile
                logger.info(f"Using existing noise profile: {noise_profile_id}")
                noise_profile = self.noise_profiles[noise_profile_id]
                y_cleaned = await self._apply_spectral_subtraction(y, sr, 
                                                                  noise_mean=noise_profile["mean"],
                                                                  noise_std=noise_profile["std"],
                                                                  strength=strength)
            
            elif noise_sample:
                # Extract and use noise sample
                logger.info("Using provided noise sample")
                noise_start = noise_sample.get('start_time', 0.0)
                noise_duration = noise_sample.get('duration', 1.0)
                
                # Analyze the specified noise sample
                noise_profile = await self.analyze_noise_profile(
                    audio_path, noise_start, noise_duration)
                
                if "error" in noise_profile:
                    return noise_profile
                
                # Use the profile
                np_id = noise_profile["id"]
                np_data = self.noise_profiles[np_id]
                y_cleaned = await self._apply_spectral_subtraction(y, sr, 
                                                                  noise_mean=np_data["mean"],
                                                                  noise_std=np_data["std"],
                                                                  strength=strength)
            
            elif auto_detect:
                # Auto-detect noise sections
                logger.info("Auto-detecting noise sections")
                noise_sections = await self._detect_noise_sections(y, sr)
                
                if not noise_sections:
                    logger.warning("No noise sections detected, using noisereduce library's algorithm")
                    # Use noisereduce's statistical method
                    reduction_strength = strength if strength is not None else self.reduction_strength
                    prop_decrease = min(max(reduction_strength, 0.0), 1.0)
                    
                    # Process in chunks to avoid memory issues with long files
                    if len(y) > sr * self.chunk_size:
                        y_cleaned = await self._process_in_chunks(y, sr, prop_decrease)
                    else:
                        y_cleaned = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease,
                                                  n_fft=self.n_fft, 
                                                  n_std_thresh=self.n_std_thresh,
                                                  freq_mask_smooth_hz=self.freq_mask_smooth_hz,
                                                  time_mask_smooth_ms=self.time_mask_smooth_ms)
                else:
                    # Use the first detected noise section
                    noise_section = noise_sections[0]
                    noise_start_sample = int(noise_section["start"] * sr)
                    noise_end_sample = int(noise_section["end"] * sr)
                    noise_sample = y[noise_start_sample:noise_end_sample]
                    
                    # Calculate noise profile
                    noise_stft = librosa.stft(noise_sample, n_fft=self.n_fft, 
                                             hop_length=self.hop_length, 
                                             win_length=self.win_length)
                    noise_spec = np.abs(noise_stft) ** 2
                    noise_mean = np.mean(noise_spec, axis=1)
                    noise_std = np.std(noise_spec, axis=1)
                    
                    # Apply spectral subtraction
                    y_cleaned = await self._apply_spectral_subtraction(y, sr, 
                                                                      noise_mean=noise_mean,
                                                                      noise_std=noise_std,
                                                                      strength=strength)
            else:
                # Default to noisereduce's statistical method
                logger.info("Using noisereduce library's algorithm")
                reduction_strength = strength if strength is not None else self.reduction_strength
                prop_decrease = min(max(reduction_strength, 0.0), 1.0)
                
                # Process in chunks to avoid memory issues with long files
                if len(y) > sr * self.chunk_size:
                    y_cleaned = await self._process_in_chunks(y, sr, prop_decrease)
                else:
                    y_cleaned = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease,
                                              n_fft=self.n_fft, 
                                              n_std_thresh=self.n_std_thresh,
                                              freq_mask_smooth_hz=self.freq_mask_smooth_hz,
                                              time_mask_smooth_ms=self.time_mask_smooth_ms)
            
            # Save the processed audio
            sf.write(output_path, y_cleaned, sr)
            
            # Return result
            return {
                "status": "success",
                "input_path": audio_path,
                "output_path": output_path,
                "sample_rate": sr,
                "duration": len(y) / sr,
                "processed_at": time.time()
            }
        
        except Exception as e:
            logger.error(f"Error reducing noise: {str(e)}")
            return {"error": f"Failed to reduce noise: {str(e)}"}
    
    async def _load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Load an audio file using librosa.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Check if the file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None, None
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            return y, sr
        
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            return None, None
    
    async def _load_audio_segment(
        self, 
        audio_path: str, 
        start_time: float, 
        duration: float
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Load a segment of an audio file.
        
        Args:
            audio_path: Path to the audio file
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Tuple of (audio_segment, sample_rate)
        """
        try:
            # Check if the file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None, None
            
            # Use pydub to extract segment
            audio = AudioSegment.from_file(audio_path)
            start_ms = int(start_time * 1000)
            duration_ms = int(duration * 1000)
            
            # Extract segment
            segment = audio[start_ms:start_ms + duration_ms]
            
            # Convert to numpy array
            samples = np.array(segment.get_array_of_samples())
            
            # Convert to float [-1, 1]
            if segment.sample_width == 2:  # 16-bit
                samples = samples / 32768.0
            elif segment.sample_width == 4:  # 32-bit
                samples = samples / 2147483648.0
            
            # Convert stereo to mono if needed
            if segment.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            
            return samples, segment.frame_rate
        
        except Exception as e:
            logger.error(f"Error loading audio segment: {str(e)}")
            return None, None
    
    async def _detect_noise_sections(
        self, 
        y: np.ndarray, 
        sr: int
    ) -> List[Dict[str, float]]:
        """
        Detect sections in audio that are likely to be background noise.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            List of dictionaries with start and end times of detected noise sections
        """
        try:
            # Calculate RMS energy in short windows
            frame_length = int(0.025 * sr)  # 25ms windows
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Calculate energy
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Convert frames to time
            times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
            
            # Find low energy sections (likely to be noise/silence)
            threshold = np.percentile(energy, 15)  # Bottom 15% of energy frames
            is_noise = energy < threshold
            
            # Find continuous sections
            noise_sections = []
            in_noise = False
            current_start = 0
            
            for i, (noise, t) in enumerate(zip(is_noise, times)):
                if noise and not in_noise:
                    # Start of a noise section
                    in_noise = True
                    current_start = t
                elif not noise and in_noise:
                    # End of a noise section
                    in_noise = False
                    current_duration = t - current_start
                    
                    # Only use sections longer than 0.2 seconds
                    if current_duration >= 0.2:
                        noise_sections.append({
                            "start": current_start,
                            "end": t,
                            "duration": current_duration
                        })
            
            # Add the last section if we're still in noise at the end
            if in_noise:
                current_duration = times[-1] - current_start
                if current_duration >= 0.2:
                    noise_sections.append({
                        "start": current_start,
                        "end": times[-1],
                        "duration": current_duration
                    })
            
            # Sort by duration (longest first)
            noise_sections.sort(key=lambda x: x["duration"], reverse=True)
            
            return noise_sections
        
        except Exception as e:
            logger.error(f"Error detecting noise sections: {str(e)}")
            return []
    
    async def _apply_spectral_subtraction(
        self,
        y: np.ndarray,
        sr: int,
        noise_mean: np.ndarray,
        noise_std: np.ndarray,
        strength: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply spectral subtraction using a pre-computed noise profile.
        
        Args:
            y: Audio signal
            sr: Sample rate
            noise_mean: Mean of noise spectrum
            noise_std: Standard deviation of noise spectrum
            strength: Noise reduction strength (0.0 to 1.0)
            
        Returns:
            Cleaned audio signal
        """
        # Set reduction strength
        reduction_strength = strength if strength is not None else self.reduction_strength
        
        # Process in chunks for long files
        if len(y) > sr * self.chunk_size:
            # Process in overlapping chunks to avoid boundary artifacts
            chunk_samples = int(sr * self.chunk_size)
            padding_samples = int(sr * self.padding)
            
            result = np.zeros_like(y)
            total_chunks = int(np.ceil(len(y) / chunk_samples))
            
            for i in range(total_chunks):
                # Calculate chunk boundaries with padding
                start = max(0, i * chunk_samples - padding_samples)
                end = min(len(y), (i + 1) * chunk_samples + padding_samples)
                
                # Extract chunk
                chunk = y[start:end]
                
                # Process chunk with spectral subtraction
                chunk_stft = librosa.stft(chunk, n_fft=self.n_fft, 
                                         hop_length=self.hop_length, 
                                         win_length=self.win_length)
                chunk_spec = np.abs(chunk_stft)
                chunk_angle = np.angle(chunk_stft)
                
                # Apply spectral subtraction 
                gain = 1 - (reduction_strength * (noise_mean[:, np.newaxis] / 
                                               (chunk_spec + 1e-10)))
                
                # Apply flooring to avoid negative values
                gain = np.maximum(gain, 0.1)
                
                # Apply gain
                chunk_spec_clean = chunk_spec * gain
                
                # Convert back to time domain
                chunk_stft_clean = chunk_spec_clean * np.exp(1j * chunk_angle)
                chunk_clean = librosa.istft(chunk_stft_clean, 
                                           hop_length=self.hop_length, 
                                           win_length=self.win_length)
                
                # Calculate the non-padded region
                if i == 0:
                    # First chunk, no left padding to remove
                    chunk_start = 0
                    chunk_end = min(chunk_samples, len(chunk_clean))
                elif i == total_chunks - 1:
                    # Last chunk, no right padding to remove
                    chunk_start = padding_samples
                    chunk_end = len(chunk_clean)
                else:
                    # Middle chunk, remove padding from both sides
                    chunk_start = padding_samples
                    chunk_end = padding_samples + chunk_samples
                
                # Add the processed chunk to the result
                if i == 0:
                    # First chunk
                    result_end = min(chunk_samples, len(chunk_clean) - chunk_start)
                    result[:result_end] = chunk_clean[chunk_start:chunk_start + result_end]
                elif i == total_chunks - 1:
                    # Last chunk
                    result_start = i * chunk_samples
                    result_end = len(y)
                    result[result_start:result_end] = chunk_clean[chunk_start:chunk_start + (result_end - result_start)]
                else:
                    # Middle chunk
                    result_start = i * chunk_samples
                    result_end = min((i + 1) * chunk_samples, len(y))
                    result[result_start:result_end] = chunk_clean[chunk_start:chunk_start + (result_end - result_start)]
            
            return result
        else:
            # Process the entire file at once
            # Compute STFT
            stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            spec = np.abs(stft)
            angle = np.angle(stft)
            
            # Apply spectral subtraction
            gain = 1 - (reduction_strength * (noise_mean[:, np.newaxis] / (spec + 1e-10)))
            
            # Apply flooring to avoid negative values
            gain = np.maximum(gain, 0.1)
            
            # Apply gain
            spec_clean = spec * gain
            
            # Convert back to time domain
            stft_clean = spec_clean * np.exp(1j * angle)
            y_clean = librosa.istft(stft_clean, hop_length=self.hop_length, win_length=self.win_length)
            
            return y_clean
    
    async def _process_in_chunks(
        self, 
        y: np.ndarray, 
        sr: int, 
        prop_decrease: float
    ) -> np.ndarray:
        """
        Process audio in chunks using noisereduce library.
        
        Args:
            y: Audio signal
            sr: Sample rate
            prop_decrease: Noise reduction strength (0.0 to 1.0)
            
        Returns:
            Cleaned audio signal
        """
        # Process in overlapping chunks to avoid boundary artifacts
        chunk_samples = int(sr * self.chunk_size)
        padding_samples = int(sr * self.padding)
        
        result = np.zeros_like(y)
        total_chunks = int(np.ceil(len(y) / chunk_samples))
        
        for i in range(total_chunks):
            # Calculate chunk boundaries with padding
            start = max(0, i * chunk_samples - padding_samples)
            end = min(len(y), (i + 1) * chunk_samples + padding_samples)
            
            # Extract chunk
            chunk = y[start:end]
            
            # Process chunk with noisereduce
            chunk_clean = nr.reduce_noise(y=chunk, sr=sr, 
                                         prop_decrease=prop_decrease,
                                         n_fft=self.n_fft,
                                         n_std_thresh=self.n_std_thresh,
                                         freq_mask_smooth_hz=self.freq_mask_smooth_hz,
                                         time_mask_smooth_ms=self.time_mask_smooth_ms)
            
            # Calculate the non-padded region
            if i == 0:
                # First chunk, no left padding to remove
                chunk_start = 0
                chunk_end = min(chunk_samples, len(chunk_clean))
            elif i == total_chunks - 1:
                # Last chunk, no right padding to remove
                chunk_start = padding_samples
                chunk_end = len(chunk_clean)
            else:
                # Middle chunk, remove padding from both sides
                chunk_start = padding_samples
                chunk_end = padding_samples + chunk_samples
            
            # Add the processed chunk to the result
            if i == 0:
                # First chunk
                result_end = min(chunk_samples, len(chunk_clean) - chunk_start)
                result[:result_end] = chunk_clean[chunk_start:chunk_start + result_end]
            elif i == total_chunks - 1:
                # Last chunk
                result_start = i * chunk_samples
                result_end = len(y)
                result[result_start:result_end] = chunk_clean[chunk_start:chunk_start + (result_end - result_start)]
            else:
                # Middle chunk
                result_start = i * chunk_samples
                result_end = min((i + 1) * chunk_samples, len(y))
                result[result_start:result_end] = chunk_clean[chunk_start:chunk_start + (result_end - result_start)]
        
        return result 