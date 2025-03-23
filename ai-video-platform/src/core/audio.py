import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import torch
import torchaudio
from scipy import signal
import noisereduce as nr

class AudioProcessor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def load_audio(
        self,
        audio_path: str,
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform and sample rate."""
        waveform, sample_rate = librosa.load(audio_path, sr=sr)
        return waveform, sample_rate
    
    def save_audio(
        self,
        waveform: np.ndarray,
        output_path: str,
        sr: int
    ):
        """Save audio waveform to file."""
        sf.write(output_path, waveform, sr)
    
    def enhance_audio(
        self,
        waveform: np.ndarray,
        sr: int,
        noise_reduction: float = 0.7,
        voice_enhancement: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Enhance audio quality."""
        # Convert to float32 if needed
        waveform = waveform.astype(np.float32)
        
        # Noise reduction
        if noise_reduction > 0:
            waveform = nr.reduce_noise(
                y=waveform,
                sr=sr,
                prop_decrease=noise_reduction
            )
        
        # Voice enhancement
        if voice_enhancement:
            waveform = self._enhance_voice(waveform, sr)
        
        # Normalize audio
        if normalize:
            waveform = self._normalize_audio(waveform)
        
        return waveform
    
    def _enhance_voice(
        self,
        waveform: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """Enhance voice frequencies."""
        # Convert to frequency domain
        D = librosa.stft(waveform)
        
        # Apply frequency-dependent gain
        freqs = librosa.fft_frequencies(sr=sr)
        gain = np.ones_like(freqs)
        voice_freqs = (freqs >= 85) & (freqs <= 255)
        gain[voice_freqs] = 1.2
        
        # Apply gain
        D_enhanced = D * gain[:, np.newaxis]
        
        # Convert back to time domain
        waveform_enhanced = librosa.istft(D_enhanced)
        
        return waveform_enhanced
    
    def _normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to target LUFS."""
        target_lufs = -14.0
        
        # Calculate current LUFS
        current_lufs = self._calculate_lufs(waveform)
        
        # Calculate gain adjustment
        gain_adjustment = target_lufs - current_lufs
        
        # Apply gain
        waveform_normalized = waveform * (10 ** (gain_adjustment / 20))
        
        return waveform_normalized
    
    def _calculate_lufs(self, waveform: np.ndarray) -> float:
        """Calculate integrated LUFS value."""
        # Convert to power
        power = waveform ** 2
        
        # Calculate mean power
        mean_power = np.mean(power)
        
        # Convert to LUFS
        lufs = 10 * np.log10(mean_power) + 70
        
        return lufs
    
    def extract_features(
        self,
        waveform: np.ndarray,
        sr: int
    ) -> dict:
        """Extract audio features."""
        # Calculate basic features
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(waveform)
        
        # Calculate statistics
        features = {
            "mfcc_mean": np.mean(mfcc, axis=1),
            "mfcc_std": np.std(mfcc, axis=1),
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_rolloff_mean": np.mean(spectral_rolloff),
            "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
            "loudness": self._calculate_lufs(waveform),
            "dynamic_range": self._calculate_dynamic_range(waveform)
        }
        
        return features
    
    def _calculate_dynamic_range(self, waveform: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        # Calculate RMS values
        rms = librosa.feature.rms(y=waveform)
        
        # Calculate dynamic range
        dynamic_range = 20 * np.log10(np.max(rms) / np.min(rms))
        
        return dynamic_range
    
    def detect_silence(
        self,
        waveform: np.ndarray,
        sr: int,
        threshold_db: float = -30,
        min_silence_duration: float = 0.3
    ) -> list:
        """Detect silence segments in audio."""
        # Convert to dB
        db = librosa.amplitude_to_db(np.abs(waveform))
        
        # Find silence regions
        silence_mask = db < threshold_db
        
        # Find silence boundaries
        silence_starts = np.where(np.diff(silence_mask.astype(int)) == 1)[0]
        silence_ends = np.where(np.diff(silence_mask.astype(int)) == -1)[0]
        
        # Convert to time
        silence_segments = []
        for start, end in zip(silence_starts, silence_ends):
            duration = (end - start) / sr
            if duration >= min_silence_duration:
                silence_segments.append({
                    "start": start / sr,
                    "end": end / sr,
                    "duration": duration
                })
        
        return silence_segments 