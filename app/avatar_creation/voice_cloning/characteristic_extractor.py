import os
import numpy as np
import librosa
import torch
import time
from typing import Dict, List, Tuple, Optional, Union, Any

class VoiceCharacteristicExtractor:
    """
    Class for extracting voice characteristics from audio samples.
    Analyzes pitch, timbre, rhythm, and other vocal characteristics
    that define a unique voice identity.
    """
    
    def __init__(self, 
                sample_rate: int = 22050,
                min_sample_duration: float = 15.0,
                use_gpu: bool = True,
                feature_dimensionality: int = 256):
        """
        Initialize the voice characteristic extractor.
        
        Args:
            sample_rate: Sample rate for audio processing
            min_sample_duration: Minimum duration (in seconds) required for reliable extraction
            use_gpu: Whether to use GPU for computation
            feature_dimensionality: Dimension of the extracted feature vector
        """
        self.sample_rate = sample_rate
        self.min_sample_duration = min_sample_duration
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.feature_dimensionality = feature_dimensionality
        
        # State tracking
        self.extraction_history = []
        self.last_extraction_time = 0
        
        # Feature extraction parameters
        self.n_mfcc = 40
        self.n_mels = 80
        self.frame_length = 1024
        self.hop_length = 256
        self.f0_min = 60  # Hz
        self.f0_max = 700  # Hz
        
        print(f"Voice Characteristic Extractor initialized")
        print(f"  - Sample rate: {self.sample_rate} Hz")
        print(f"  - Minimum sample duration: {self.min_sample_duration} seconds")
        print(f"  - Using device: {self.device}")
        print(f"  - Feature dimensionality: {self.feature_dimensionality}")
    
    def extract_characteristics(self, 
                               audio_path: str, 
                               vad_threshold: float = 0.3,
                               save_features: bool = True,
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract voice characteristics from an audio file.
        
        Args:
            audio_path: Path to the audio file
            vad_threshold: Voice activity detection threshold (0-1)
            save_features: Whether to save extracted features
            output_dir: Directory to save features (if None, uses directory of audio_path)
            
        Returns:
            Dictionary of extracted voice characteristics
        """
        start_time = time.time()
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Check audio duration
            duration = len(y) / sr
            if duration < self.min_sample_duration:
                print(f"Warning: Audio sample ({duration:.2f}s) is shorter than the recommended minimum " 
                      f"duration ({self.min_sample_duration}s). Extraction may be less reliable.")
            
            # Voice activity detection to remove silence
            y_voiced = self._apply_vad(y, vad_threshold)
            if len(y_voiced) / sr < 3.0:  # At least 3 seconds of speech required
                print(f"Warning: Not enough voiced content detected. Using original audio.")
                y_voiced = y
            
            # Extract basic characteristics
            characteristics = {}
            
            # Pitch (F0) statistics
            characteristics['pitch'] = self._extract_pitch_features(y_voiced)
            
            # Spectral features (timbre)
            characteristics['spectral'] = self._extract_spectral_features(y_voiced)
            
            # Temporal features (rhythm, speaking rate)
            characteristics['temporal'] = self._extract_temporal_features(y_voiced)
            
            # Voice quality features
            characteristics['quality'] = self._extract_voice_quality_features(y_voiced)
            
            # Calculate overall voice feature vector (for similarity comparison)
            characteristics['feature_vector'] = self._compute_feature_vector(characteristics)
            
            # Save extraction info
            extraction_info = {
                'timestamp': time.time(),
                'audio_path': audio_path,
                'duration': duration,
                'voiced_duration': len(y_voiced) / sr,
                'characteristics': characteristics.copy()
            }
            self.extraction_history.append(extraction_info)
            self.last_extraction_time = time.time()
            
            # Save features if requested
            if save_features:
                out_dir = output_dir if output_dir else os.path.dirname(audio_path)
                os.makedirs(out_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(audio_path))[0]
                
                # Save as numpy file
                feature_path = os.path.join(out_dir, f"{basename}_voice_features.npz")
                np.savez(feature_path, 
                         feature_vector=characteristics['feature_vector'],
                         pitch_features=characteristics['pitch'],
                         spectral_features=characteristics['spectral']['mfcc_stats'],
                         temporal_features=characteristics['temporal'])
                
                characteristics['feature_path'] = feature_path
            
            # Compute processing time
            processing_time = time.time() - start_time
            characteristics['processing_time'] = processing_time
            
            print(f"Voice characteristics extracted in {processing_time:.2f} seconds")
            return characteristics
            
        except Exception as e:
            print(f"Error extracting voice characteristics: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _apply_vad(self, y: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply voice activity detection to remove silence.
        
        Args:
            y: Audio signal
            threshold: Energy threshold for speech detection
            
        Returns:
            Audio with silence removed
        """
        # Calculate signal energy
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Compute short-time energy
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Normalize energy
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)
        
        # Mark frames as voiced where energy > threshold
        voiced_frames = energy_norm > threshold
        
        # Convert frame indicators to sample-level mask
        voiced_samples = np.zeros_like(y, dtype=bool)
        for i, voiced in enumerate(voiced_frames):
            start = i * hop_length
            end = min(start + frame_length, len(y))
            if voiced:
                voiced_samples[start:end] = True
        
        # Extract only voiced samples
        y_voiced = y[voiced_samples]
        
        return y_voiced
    
    def _extract_pitch_features(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Extract pitch (F0) related features.
        
        Args:
            y: Audio signal
            
        Returns:
            Dictionary of pitch features
        """
        # Extract fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate
        )
        
        # Remove unvoiced frames (NaN values)
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) == 0:
            # Handle case where no pitch could be detected
            return {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'voiced_ratio': 0.0
            }
        
        # Calculate statistics
        pitch_features = {
            'mean': float(np.mean(f0_voiced)),
            'std': float(np.std(f0_voiced)),
            'median': float(np.median(f0_voiced)),
            'min': float(np.min(f0_voiced)),
            'max': float(np.max(f0_voiced)),
            'range': float(np.max(f0_voiced) - np.min(f0_voiced)),
            'voiced_ratio': float(np.mean(voiced_flag))
        }
        
        # Pitch dynamics (how much pitch varies over time)
        if len(f0_voiced) > 1:
            f0_diff = np.diff(f0_voiced)
            pitch_features['variation'] = float(np.mean(np.abs(f0_diff)))
        else:
            pitch_features['variation'] = 0.0
        
        return pitch_features
    
    def _extract_spectral_features(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Extract spectral features (timbre).
        
        Args:
            y: Audio signal
            
        Returns:
            Dictionary of spectral features
        """
        spectral_features = {}
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Calculate statistics of MFCCs
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Store MFCC statistics
        spectral_features['mfcc_stats'] = np.concatenate([mfcc_means, mfcc_stds])
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Calculate statistics of mel spectrogram
        spectral_features['mel_mean'] = float(np.mean(mel_spec_db))
        spectral_features['mel_std'] = float(np.std(mel_spec_db))
        
        # Spectral centroid (brightness of sound)
        cent = librosa.feature.spectral_centroid(
            y=y, 
            sr=self.sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        spectral_features['centroid_mean'] = float(np.mean(cent))
        spectral_features['centroid_std'] = float(np.std(cent))
        
        # Spectral bandwidth (width of the spectrum)
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, 
            sr=self.sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        spectral_features['bandwidth_mean'] = float(np.mean(bandwidth))
        
        # Spectral flatness (how tone-like vs noise-like)
        flatness = librosa.feature.spectral_flatness(
            y=y,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        spectral_features['flatness_mean'] = float(np.mean(flatness))
        
        # Spectral rolloff (frequency below which is N% of the energy)
        rolloff = librosa.feature.spectral_rolloff(
            y=y, 
            sr=self.sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            roll_percent=0.85
        )[0]
        spectral_features['rolloff_mean'] = float(np.mean(rolloff))
        
        return spectral_features
    
    def _extract_temporal_features(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Extract temporal features (rhythm, speaking rate).
        
        Args:
            y: Audio signal
            
        Returns:
            Dictionary of temporal features
        """
        temporal_features = {}
        
        # Zero crossing rate (proxy for speaking rate/consonant frequency)
        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        temporal_features['zcr_mean'] = float(np.mean(zcr))
        temporal_features['zcr_std'] = float(np.std(zcr))
        
        # Onset strength (for speech rhythm)
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        temporal_features['onset_mean'] = float(np.mean(onset_env))
        temporal_features['onset_std'] = float(np.std(onset_env))
        
        # Detect tempo and rhythmic regularity
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        temporal_features['tempo'] = float(tempo)
        
        # Estimate speaking rate (syllables per second)
        # This is a crude approximation using energy peaks
        y_harmonic, _ = librosa.effects.hpss(y)
        energy = np.abs(librosa.stft(y_harmonic, n_fft=self.frame_length, hop_length=self.hop_length))
        energy_mean = np.mean(energy, axis=0)
        peaks = librosa.util.peak_pick(energy_mean, 3, 3, 3, 5, 0.5, 10)
        
        # Estimate syllables per second
        if len(peaks) > 1:
            duration = len(y) / self.sample_rate
            temporal_features['syllable_rate'] = float(len(peaks) / duration)
        else:
            temporal_features['syllable_rate'] = 0.0
        
        return temporal_features
    
    def _extract_voice_quality_features(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Extract voice quality features (breathiness, creakiness, etc.).
        
        Args:
            y: Audio signal
            
        Returns:
            Dictionary of voice quality features
        """
        quality_features = {}
        
        # Harmonics-to-noise ratio (proxy for breathiness/clarity)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        
        if percussive_energy > 0:
            hnr = 10 * np.log10(harmonic_energy / percussive_energy)
        else:
            hnr = 30.0  # Default high value if no noise detected
        
        quality_features['hnr'] = float(hnr)
        
        # Jitter (variation in period lengths)
        # This would typically require pitch mark detection,
        # but we'll use a simplified approximation
        if len(y) > self.sample_rate // 40:  # At least 25ms
            y_centered = y - np.mean(y)
            y_normalized = y_centered / (np.std(y_centered) + 1e-8)
            
            # Find zero crossings as approximation for period boundaries
            zero_crossings = np.where(np.diff(np.signbit(y_normalized)))[0]
            if len(zero_crossings) > 2:
                periods = np.diff(zero_crossings)
                if len(periods) > 1:
                    jitter = np.std(periods) / np.mean(periods)
                    quality_features['jitter'] = float(jitter)
                else:
                    quality_features['jitter'] = 0.0
            else:
                quality_features['jitter'] = 0.0
        else:
            quality_features['jitter'] = 0.0
        
        # Shimmer (variation in amplitude)
        if len(y) > self.sample_rate // 20:  # At least 50ms
            # Calculate local maxima of the absolute signal
            peaks = librosa.util.peak_pick(np.abs(y), 5, 5, 5, 5, 0.3, 10)
            if len(peaks) > 2:
                peak_amps = np.abs(y[peaks])
                shimmer = np.std(peak_amps) / (np.mean(peak_amps) + 1e-8)
                quality_features['shimmer'] = float(shimmer)
            else:
                quality_features['shimmer'] = 0.0
        else:
            quality_features['shimmer'] = 0.0
        
        return quality_features
    
    def _compute_feature_vector(self, characteristics: Dict[str, Any]) -> np.ndarray:
        """
        Compute a unified feature vector from all characteristics.
        
        Args:
            characteristics: Dictionary of extracted characteristics
            
        Returns:
            Feature vector as numpy array
        """
        # Extract the most relevant features for voice identity
        components = []
        
        # Pitch features: mean, std, median, range
        pitch_vector = np.array([
            characteristics['pitch']['mean'],
            characteristics['pitch']['std'],
            characteristics['pitch']['median'],
            characteristics['pitch']['range'],
            characteristics['pitch']['variation']
        ])
        components.append(pitch_vector)
        
        # Spectral features: MFCCs (already a vector)
        components.append(characteristics['spectral']['mfcc_stats'])
        
        # Add selected spectral features
        spectral_vector = np.array([
            characteristics['spectral']['centroid_mean'],
            characteristics['spectral']['bandwidth_mean'],
            characteristics['spectral']['flatness_mean'],
            characteristics['spectral']['rolloff_mean']
        ])
        components.append(spectral_vector)
        
        # Temporal features
        temporal_vector = np.array([
            characteristics['temporal']['zcr_mean'],
            characteristics['temporal']['onset_mean'],
            characteristics['temporal']['syllable_rate']
        ])
        components.append(temporal_vector)
        
        # Voice quality features
        quality_vector = np.array([
            characteristics['quality']['hnr'],
            characteristics['quality']['jitter'],
            characteristics['quality']['shimmer']
        ])
        components.append(quality_vector)
        
        # Concatenate all components
        feature_vector = np.concatenate(components)
        
        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        
        # Normalize the feature vector
        feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-8)
        
        # If dimensionality is specified, adjust the vector size
        if self.feature_dimensionality > 0:
            current_dim = feature_vector.shape[0]
            
            if current_dim > self.feature_dimensionality:
                # Reduce dimensionality (simple approach: take first N dimensions)
                feature_vector = feature_vector[:self.feature_dimensionality]
            elif current_dim < self.feature_dimensionality:
                # Pad with zeros to reach target dimensionality
                padding = np.zeros(self.feature_dimensionality - current_dim)
                feature_vector = np.concatenate([feature_vector, padding])
        
        return feature_vector
    
    def compare_voices(self, 
                      feature_vector1: np.ndarray, 
                      feature_vector2: np.ndarray) -> Dict[str, float]:
        """
        Compare two voice feature vectors for similarity.
        
        Args:
            feature_vector1: First voice feature vector
            feature_vector2: Second voice feature vector
            
        Returns:
            Dictionary of similarity metrics
        """
        metrics = {}
        
        # Ensure vectors are of the same length
        min_length = min(len(feature_vector1), len(feature_vector2))
        v1 = feature_vector1[:min_length]
        v2 = feature_vector2[:min_length]
        
        # Euclidean distance (smaller = more similar)
        euclidean_dist = np.linalg.norm(v1 - v2)
        metrics['euclidean_distance'] = float(euclidean_dist)
        
        # Cosine similarity (larger = more similar)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
            metrics['cosine_similarity'] = float(cosine_sim)
        else:
            metrics['cosine_similarity'] = 0.0
        
        # Correlation coefficient
        if np.std(v1) > 0 and np.std(v2) > 0:
            correlation = np.corrcoef(v1, v2)[0, 1]
            metrics['correlation'] = float(correlation)
        else:
            metrics['correlation'] = 0.0
        
        return metrics
    
    def load_feature_vector(self, feature_path: str) -> np.ndarray:
        """
        Load a feature vector from a file.
        
        Args:
            feature_path: Path to the feature file (.npz)
            
        Returns:
            Feature vector as numpy array
        """
        try:
            data = np.load(feature_path)
            return data['feature_vector']
        except Exception as e:
            print(f"Error loading feature vector: {e}")
            return np.array([])
    
    def reset(self) -> None:
        """
        Reset the extractor state.
        """
        self.extraction_history = []
        self.last_extraction_time = 0
        print("Voice characteristic extractor reset to initial state") 