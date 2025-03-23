"""
Dynamics Processor Module

This module provides functionality for processing audio dynamics, including
compression, limiting, expansion, and gating to improve audio quality.
"""

import os
import numpy as np
import logging
import tempfile
import librosa
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

class DynamicsProcessor:
    """
    A class for processing audio dynamics to improve audio quality.
    
    This class provides methods for compressing, limiting, expanding, and gating
    audio signals to control dynamic range and improve overall quality.
    """
    
    def __init__(
        self,
        # Compressor parameters
        comp_threshold: float = -24.0,
        comp_ratio: float = 2.0,
        comp_attack: float = 0.01,
        comp_release: float = 0.15,
        comp_makeup_gain: float = 0.0,
        comp_knee: float = 6.0,
        # Limiter parameters
        limit_threshold: float = -0.5,
        limit_release: float = 0.01,
        # Expander parameters
        exp_threshold: float = -45.0,
        exp_ratio: float = 1.5,
        exp_attack: float = 0.2,
        exp_release: float = 0.4,
        # Gate parameters
        gate_threshold: float = -60.0,
        gate_ratio: float = 10.0,
        gate_attack: float = 0.01,
        gate_release: float = 0.2,
        gate_hold: float = 0.02,
        # General parameters
        detect_mode: str = "RMS",
        auto_makeup_gain: bool = True,
        lookahead: float = 0.005,
        preset: Optional[str] = None
    ):
        """
        Initialize the DynamicsProcessor with customizable parameters.
        
        Args:
            comp_threshold: Threshold in dB for compressor
            comp_ratio: Ratio for compressor
            comp_attack: Attack time in seconds for compressor
            comp_release: Release time in seconds for compressor
            comp_makeup_gain: Manual makeup gain in dB for compressor
            comp_knee: Knee width in dB for compressor
            limit_threshold: Threshold in dB for limiter
            limit_release: Release time in seconds for limiter
            exp_threshold: Threshold in dB for expander
            exp_ratio: Ratio for expander
            exp_attack: Attack time in seconds for expander
            exp_release: Release time in seconds for expander
            gate_threshold: Threshold in dB for gate
            gate_ratio: Ratio for gate
            gate_attack: Attack time in seconds for gate
            gate_release: Release time in seconds for gate
            gate_hold: Hold time in seconds for gate
            detect_mode: Envelope detection mode ("RMS" or "peak")
            auto_makeup_gain: Whether to automatically apply makeup gain
            lookahead: Lookahead time in seconds for peak limiting
            preset: Optional preset name to use predefined settings
        """
        # Apply presets if specified
        if preset:
            self._apply_preset(preset)
        else:
            # Store compressor parameters
            self.comp_threshold = comp_threshold
            self.comp_ratio = comp_ratio
            self.comp_attack = comp_attack
            self.comp_release = comp_release
            self.comp_makeup_gain = comp_makeup_gain
            self.comp_knee = comp_knee
            
            # Store limiter parameters
            self.limit_threshold = limit_threshold
            self.limit_release = limit_release
            
            # Store expander parameters
            self.exp_threshold = exp_threshold
            self.exp_ratio = exp_ratio
            self.exp_attack = exp_attack
            self.exp_release = exp_release
            
            # Store gate parameters
            self.gate_threshold = gate_threshold
            self.gate_ratio = gate_ratio
            self.gate_attack = gate_attack
            self.gate_release = gate_release
            self.gate_hold = gate_hold
            
            # Store general parameters
            self.detect_mode = detect_mode
            self.auto_makeup_gain = auto_makeup_gain
            self.lookahead = lookahead
    
    def _apply_preset(self, preset: str):
        """
        Apply a predefined preset for dynamics processing.
        
        Args:
            preset: Name of the preset to apply
        """
        presets = {
            "voice_broadcast": {
                "comp_threshold": -20.0,
                "comp_ratio": 3.0,
                "comp_attack": 0.01,
                "comp_release": 0.1,
                "comp_knee": 6.0,
                "limit_threshold": -1.0,
                "exp_threshold": -50.0,
                "gate_threshold": -70.0,
                "auto_makeup_gain": True
            },
            "voice_intimate": {
                "comp_threshold": -24.0,
                "comp_ratio": 2.0,
                "comp_attack": 0.02,
                "comp_release": 0.2,
                "comp_knee": 8.0,
                "limit_threshold": -3.0,
                "exp_threshold": -45.0,
                "gate_threshold": -60.0,
                "auto_makeup_gain": True
            },
            "music_master": {
                "comp_threshold": -18.0,
                "comp_ratio": 1.5,
                "comp_attack": 0.05,
                "comp_release": 0.2,
                "comp_knee": 12.0,
                "limit_threshold": -0.3,
                "exp_threshold": -90.0,  # Effectively disabled
                "gate_threshold": -90.0,  # Effectively disabled
                "auto_makeup_gain": True
            },
            "dialog_leveler": {
                "comp_threshold": -22.0,
                "comp_ratio": 3.0,
                "comp_attack": 0.015,
                "comp_release": 0.4,
                "comp_knee": 10.0,
                "limit_threshold": -1.0,
                "exp_threshold": -50.0,
                "gate_threshold": -65.0,
                "auto_makeup_gain": True
            },
            "transparent": {
                "comp_threshold": -24.0,
                "comp_ratio": 1.5,
                "comp_attack": 0.1,
                "comp_release": 0.3,
                "comp_knee": 18.0,
                "limit_threshold": -0.1,
                "exp_threshold": -90.0,  # Effectively disabled
                "gate_threshold": -90.0,  # Effectively disabled
                "auto_makeup_gain": False
            }
        }
        
        # Apply preset if it exists, otherwise log warning
        if preset in presets:
            preset_values = presets[preset]
            for key, value in preset_values.items():
                setattr(self, key, value)
            
            # Set default values for any parameters not in the preset
            if "comp_makeup_gain" not in preset_values:
                self.comp_makeup_gain = 0.0
            if "limit_release" not in preset_values:
                self.limit_release = 0.01
            if "exp_ratio" not in preset_values:
                self.exp_ratio = 1.5
            if "exp_attack" not in preset_values:
                self.exp_attack = 0.2
            if "exp_release" not in preset_values:
                self.exp_release = 0.4
            if "gate_ratio" not in preset_values:
                self.gate_ratio = 10.0
            if "gate_attack" not in preset_values:
                self.gate_attack = 0.01
            if "gate_release" not in preset_values:
                self.gate_release = 0.2
            if "gate_hold" not in preset_values:
                self.gate_hold = 0.02
            if "detect_mode" not in preset_values:
                self.detect_mode = "RMS"
            if "lookahead" not in preset_values:
                self.lookahead = 0.005
                
            logger.info(f"Applied '{preset}' dynamics processing preset")
        else:
            logger.warning(f"Preset '{preset}' not found, using default settings")
            # Set default values
            self.comp_threshold = -24.0
            self.comp_ratio = 2.0
            self.comp_attack = 0.01
            self.comp_release = 0.15
            self.comp_makeup_gain = 0.0
            self.comp_knee = 6.0
            self.limit_threshold = -0.5
            self.limit_release = 0.01
            self.exp_threshold = -45.0
            self.exp_ratio = 1.5
            self.exp_attack = 0.2
            self.exp_release = 0.4
            self.gate_threshold = -60.0
            self.gate_ratio = 10.0
            self.gate_attack = 0.01
            self.gate_release = 0.2
            self.gate_hold = 0.02
            self.detect_mode = "RMS"
            self.auto_makeup_gain = True
            self.lookahead = 0.005
    
    def process_audio(
        self,
        audio_path: str,
        output_path: str,
        apply_compression: bool = True,
        apply_limiting: bool = True,
        apply_expansion: bool = False,
        apply_gating: bool = False,
        target_loudness: Optional[float] = None,
        dry_wet_mix: float = 1.0
    ) -> Dict[str, Any]:
        """
        Process audio dynamics in an audio file.
        
        Args:
            audio_path: Path to the input audio file
            output_path: Path to save the processed audio
            apply_compression: Whether to apply compression
            apply_limiting: Whether to apply limiting
            apply_expansion: Whether to apply expansion
            apply_gating: Whether to apply gating
            target_loudness: Target integrated loudness in LUFS (optional)
            dry_wet_mix: Mix of original and processed signal (0.0-1.0)
            
        Returns:
            Dictionary with process info and statistics
        """
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Keep track of processing steps
            processing_steps = []
            
            # Measure input levels
            input_peak = np.max(np.abs(y))
            input_rms = np.sqrt(np.mean(y**2))
            input_loudness = self._estimate_loudness(y, sr)
            
            # Keep a copy of the original signal for dry/wet mixing
            y_original = y.copy()
            
            # Process stages in order: gate -> expander -> compressor -> limiter
            if apply_gating and self.gate_threshold > -90:
                y = self._apply_gate(y, sr)
                processing_steps.append("gate")
                
            if apply_expansion and self.exp_threshold > -90:
                y = self._apply_expander(y, sr)
                processing_steps.append("expander")
                
            if apply_compression and self.comp_threshold < 0:
                y = self._apply_compressor(y, sr)
                processing_steps.append("compressor")
                
            if apply_limiting and self.limit_threshold < 0:
                y = self._apply_limiter(y, sr)
                processing_steps.append("limiter")
            
            # Apply target loudness if specified (after all dynamics processing)
            if target_loudness is not None:
                y = self._match_loudness(y, sr, target_loudness)
                processing_steps.append(f"loudness_matching_to_{target_loudness}LUFS")
            
            # Apply dry/wet mix if not fully wet
            if dry_wet_mix < 1.0:
                y = (1 - dry_wet_mix) * y_original + dry_wet_mix * y
                processing_steps.append(f"dry_wet_mix_{dry_wet_mix:.2f}")
            
            # Measure output levels
            output_peak = np.max(np.abs(y))
            output_rms = np.sqrt(np.mean(y**2))
            output_loudness = self._estimate_loudness(y, sr)
            
            # Ensure no clipping
            if output_peak > 0.999:
                y = y * (0.999 / output_peak)
                logger.info("Applied safety gain reduction to prevent clipping")
            
            # Save the processed audio
            sf.write(output_path, y, sr)
            
            # Return processing info
            return {
                "status": "success",
                "input_path": audio_path,
                "output_path": output_path,
                "processing_steps": processing_steps,
                "sample_rate": sr,
                "duration": librosa.get_duration(y=y, sr=sr),
                "levels": {
                    "input": {
                        "peak_dB": 20 * np.log10(input_peak + 1e-10),
                        "rms_dB": 20 * np.log10(input_rms + 1e-10),
                        "loudness_LUFS": input_loudness
                    },
                    "output": {
                        "peak_dB": 20 * np.log10(output_peak + 1e-10),
                        "rms_dB": 20 * np.log10(output_rms + 1e-10),
                        "loudness_LUFS": output_loudness
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing dynamics: {str(e)}")
            return {
                "status": "error",
                "input_path": audio_path,
                "error": str(e)
            }
    
    def _get_envelope(self, y: np.ndarray, sr: int, attack_time: float, release_time: float) -> np.ndarray:
        """
        Calculate envelope of the audio signal using specified attack and release times.
        
        Args:
            y: Audio signal
            sr: Sample rate
            attack_time: Attack time in seconds
            release_time: Release time in seconds
            
        Returns:
            Signal envelope
        """
        # Convert times to samples
        attack_samples = int(attack_time * sr)
        release_samples = int(release_time * sr)
        
        # Ensure minimum of 1 sample
        attack_samples = max(1, attack_samples)
        release_samples = max(1, release_samples)
        
        # Initialize envelope
        if self.detect_mode.upper() == "RMS":
            # For RMS detection, square the signal first
            y_abs = y ** 2
        else:
            # For peak detection, take absolute value
            y_abs = np.abs(y)
        
        envelope = np.zeros_like(y_abs)
        
        # Time constants
        attack_coef = np.exp(-1.0 / attack_samples)
        release_coef = np.exp(-1.0 / release_samples)
        
        # First sample
        envelope[0] = y_abs[0]
        
        # Apply attack/release envelope follower
        for i in range(1, len(y_abs)):
            if self.detect_mode.upper() == "RMS":
                # For RMS mode
                if y_abs[i] > envelope[i-1]:
                    envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * y_abs[i]
                else:
                    envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * y_abs[i]
            else:
                # For peak mode
                if y_abs[i] > envelope[i-1]:
                    envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * y_abs[i]
                else:
                    envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * y_abs[i]
        
        # Convert back to amplitude for RMS mode
        if self.detect_mode.upper() == "RMS":
            envelope = np.sqrt(envelope)
        
        return envelope
    
    def _apply_compressor(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply compression to the audio signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Compressed audio signal
        """
        # Get envelope
        envelope = self._get_envelope(y, sr, self.comp_attack, self.comp_release)
        
        # Convert threshold to linear
        threshold_linear = 10 ** (self.comp_threshold / 20)
        knee_range_linear = 10 ** (self.comp_knee / 20)
        
        # Initialize gain reduction array
        gain_reduction = np.ones_like(envelope)
        
        # Calculate gain reduction with knee
        for i in range(len(envelope)):
            if envelope[i] <= threshold_linear / knee_range_linear:
                # Below threshold and knee range, no compression
                gain_reduction[i] = 1.0
            elif envelope[i] >= threshold_linear * knee_range_linear:
                # Above threshold and knee range, full compression
                gain_reduction[i] = (threshold_linear / envelope[i]) ** (1 - 1/self.comp_ratio)
            else:
                # Within knee range, gradual compression
                # This is a simplified soft knee calculation
                knee_factor = ((20 * np.log10(envelope[i]) - self.comp_threshold) + self.comp_knee/2) / self.comp_knee
                knee_ratio = 1 + (self.comp_ratio - 1) * knee_factor
                gain_reduction[i] = (threshold_linear / envelope[i]) ** (1 - 1/knee_ratio)
        
        # Apply gain reduction
        y_compressed = y * gain_reduction
        
        # Apply makeup gain if auto or manual
        if self.auto_makeup_gain:
            # Calculate appropriate makeup gain based on compression
            orig_rms = np.sqrt(np.mean(y**2))
            compressed_rms = np.sqrt(np.mean(y_compressed**2))
            if compressed_rms > 0:
                auto_makeup = orig_rms / compressed_rms
                y_compressed *= auto_makeup
        elif self.comp_makeup_gain != 0:
            # Apply manual makeup gain (convert from dB)
            makeup_linear = 10 ** (self.comp_makeup_gain / 20)
            y_compressed *= makeup_linear
        
        return y_compressed
    
    def _apply_limiter(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply peak limiting to the audio signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Limited audio signal
        """
        # Convert threshold to linear
        threshold_linear = 10 ** (self.limit_threshold / 20)
        
        # Implement lookahead
        lookahead_samples = int(self.lookahead * sr)
        if lookahead_samples > 0:
            # Pad the signal for lookahead processing
            y_padded = np.concatenate([y, np.zeros(lookahead_samples)])
            
            # Look ahead to find peaks
            gain_reduction = np.ones(len(y_padded))
            for i in range(len(y)):
                # Find max peak in the lookahead window
                window_end = min(i + lookahead_samples, len(y_padded))
                peak_in_window = np.max(np.abs(y_padded[i:window_end]))
                
                if peak_in_window > threshold_linear:
                    gain_reduction[i] = threshold_linear / peak_in_window
            
            # Apply attack/release smoothing to gain reduction
            release_samples = int(self.limit_release * sr)
            gain_reduction = gaussian_filter1d(gain_reduction, sigma=release_samples/4)
            
            # Apply gain reduction to original signal
            y_limited = y * gain_reduction[:len(y)]
        else:
            # No lookahead, simple limiting
            gain_reduction = np.ones_like(y)
            mask = np.abs(y) > threshold_linear
            gain_reduction[mask] = threshold_linear / np.abs(y[mask])
            
            # Apply release smoothing
            release_samples = int(self.limit_release * sr)
            gain_reduction = gaussian_filter1d(gain_reduction, sigma=release_samples/4)
            
            # Apply gain reduction
            y_limited = y * gain_reduction
        
        return y_limited
    
    def _apply_expander(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply expansion to the audio signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Expanded audio signal
        """
        # Get envelope
        envelope = self._get_envelope(y, sr, self.exp_attack, self.exp_release)
        
        # Convert threshold to linear
        threshold_linear = 10 ** (self.exp_threshold / 20)
        
        # Initialize gain reduction array
        gain_reduction = np.ones_like(envelope)
        
        # Calculate gain reduction - for expander, we reduce gain below threshold
        mask = envelope < threshold_linear
        if np.any(mask):
            ratio = 1 / self.exp_ratio  # Invert ratio for expansion
            gain_reduction[mask] = (envelope[mask] / threshold_linear) ** (1 - ratio)
        
        # Apply gain reduction
        y_expanded = y * gain_reduction
        
        return y_expanded
    
    def _apply_gate(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply noise gate to the audio signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Gated audio signal
        """
        # Get envelope
        envelope = self._get_envelope(y, sr, self.gate_attack, self.gate_release)
        
        # Convert threshold to linear
        threshold_linear = 10 ** (self.gate_threshold / 20)
        
        # Initialize gain reduction array
        gain_reduction = np.ones_like(envelope)
        
        # Calculate hold time in samples
        hold_samples = int(self.gate_hold * sr)
        
        # Apply gate with hold time
        gate_open = False
        hold_counter = 0
        
        for i in range(len(envelope)):
            if envelope[i] >= threshold_linear:
                # Signal above threshold, open gate
                gate_open = True
                hold_counter = hold_samples
                gain_reduction[i] = 1.0
            elif gate_open and hold_counter > 0:
                # Gate still open due to hold time
                hold_counter -= 1
                gain_reduction[i] = 1.0
            else:
                # Gate closed, apply gain reduction
                gate_open = False
                ratio = 1 / self.gate_ratio  # Invert ratio for extreme reduction
                gain_reduction[i] = (envelope[i] / threshold_linear) ** (1 - ratio)
        
        # Apply gain reduction
        y_gated = y * gain_reduction
        
        return y_gated
    
    def _estimate_loudness(self, y: np.ndarray, sr: int) -> float:
        """
        Estimate integrated loudness (LUFS) of the audio signal.
        This is a simplified approximation since true LUFS calculation is complex.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Estimated loudness in LUFS
        """
        # This is a simplified estimate
        # For accurate LUFS, use a dedicated library like pyloudnorm
        
        # Apply K-weighting filter approximation
        # This is a very rough approximation of the K-weighting curve
        b, a = self._get_k_weighting_filter(sr)
        y_weighted = librosa.filtfilt(b, a, y)
        
        # Calculate mean square
        ms = np.mean(y_weighted**2)
        
        # Convert to LUFS (very approximate)
        loudness = -0.691 + 10 * np.log10(ms) - 10
        
        return loudness
    
    def _get_k_weighting_filter(self, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get simplified K-weighting filter coefficients for loudness estimation.
        
        Args:
            sr: Sample rate
            
        Returns:
            Filter coefficients (b, a)
        """
        # This is a simplified approximation of the K-weighting curve
        # For accurate K-weighting, use a dedicated library
        
        # Pre-filter: high-pass filter simulating the head-related transfer function
        b1, a1 = librosa.filters.butter(2, 60.0, 'highpass', fs=sr)
        
        # RLB filter (revised low-frequency B-curve)
        b2, a2 = librosa.filters.butter(2, 120.0, 'highpass', fs=sr)
        
        # High-frequency shelf filter
        b3, a3 = librosa.filters.shelf(3, 1000.0, fs=sr, gain_db=4.0)
        
        # Combine filters (simplified approach)
        b = np.convolve(np.convolve(b1, b2), b3)
        a = np.convolve(np.convolve(a1, a2), a3)
        
        return b, a
    
    def _match_loudness(self, y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
        """
        Match the audio loudness to a target LUFS value.
        
        Args:
            y: Audio signal
            sr: Sample rate
            target_lufs: Target loudness in LUFS
            
        Returns:
            Loudness-adjusted audio signal
        """
        # Estimate current loudness
        current_lufs = self._estimate_loudness(y, sr)
        
        # Calculate gain needed
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        y_adjusted = y * gain_linear
        
        # Prevent clipping
        peak = np.max(np.abs(y_adjusted))
        if peak > 0.999:
            y_adjusted = y_adjusted * (0.999 / peak)
            logger.info(f"Reduced gain to prevent clipping during loudness matching")
        
        return y_adjusted
    
    def analyze_dynamics(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio dynamics to suggest processing parameters.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with analysis results and suggested parameters
        """
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Measure levels
            peak = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))
            peak_db = 20 * np.log10(peak + 1e-10)
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Estimate loudness
            loudness = self._estimate_loudness(y, sr)
            
            # Calculate crest factor (peak-to-average ratio)
            crest_factor = peak / (rms + 1e-10)
            crest_factor_db = 20 * np.log10(crest_factor)
            
            # Calculate dynamic range
            frame_length = int(0.05 * sr)  # 50ms frames
            hop_length = int(0.025 * sr)  # 25ms hop
            
            # Compute frame-level RMS
            rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Calculate dynamic range from frame RMS values
            if len(rms_frames) > 0:
                dynamic_range = 20 * np.log10((np.max(rms_frames) + 1e-10) / (np.mean(rms_frames) + 1e-10))
            else:
                dynamic_range = 0
                
            # Calculate low/mid/high balance
            spec = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Define frequency bands
            low_idx = np.where(freqs <= 250)[0]
            mid_idx = np.where((freqs > 250) & (freqs <= 4000))[0]
            high_idx = np.where(freqs > 4000)[0]
            
            # Calculate energy in each band
            if len(low_idx) > 0:
                low_energy = np.mean(np.sum(spec[low_idx, :], axis=1))
            else:
                low_energy = 0
                
            if len(mid_idx) > 0:
                mid_energy = np.mean(np.sum(spec[mid_idx, :], axis=1))
            else:
                mid_energy = 0
                
            if len(high_idx) > 0:
                high_energy = np.mean(np.sum(spec[high_idx, :], axis=1))
            else:
                high_energy = 0
                
            # Normalize energies
            total_energy = low_energy + mid_energy + high_energy
            if total_energy > 0:
                low_energy_pct = (low_energy / total_energy) * 100
                mid_energy_pct = (mid_energy / total_energy) * 100
                high_energy_pct = (high_energy / total_energy) * 100
            else:
                low_energy_pct = mid_energy_pct = high_energy_pct = 0
            
            # Determine content type based on analysis
            is_speech = False
            is_music = False
            is_mixed = False
            
            # Very simple heuristic based on spectral balance and dynamics
            if mid_energy_pct > 50 and 5 < dynamic_range < 20:
                is_speech = True
            elif low_energy_pct > 30 and high_energy_pct > 15 and dynamic_range > 12:
                is_music = True
            else:
                is_mixed = True
            
            # Suggest parameters based on content type and measurements
            suggested_settings = {}
            
            if is_speech:
                # Speech typically needs more compression
                suggested_settings["content_type"] = "speech"
                suggested_settings["preset"] = "voice_broadcast" if dynamic_range > 15 else "voice_intimate"
                suggested_settings["compression"] = {
                    "threshold": max(-30, min(-20, rms_db - 6)),
                    "ratio": 3.0 if dynamic_range > 15 else 2.0,
                    "attack": 0.01,
                    "release": 0.1 if dynamic_range > 15 else 0.2
                }
                suggested_settings["limiting"] = True
                suggested_settings["target_loudness"] = -16 if dynamic_range > 15 else -18
                
            elif is_music:
                # Music typically needs less compression
                suggested_settings["content_type"] = "music"
                suggested_settings["preset"] = "music_master"
                suggested_settings["compression"] = {
                    "threshold": max(-24, min(-16, rms_db - 3)),
                    "ratio": 1.5,
                    "attack": 0.05,
                    "release": 0.2
                }
                suggested_settings["limiting"] = True
                suggested_settings["target_loudness"] = -14
                
            else:
                # Mixed content - middle ground
                suggested_settings["content_type"] = "mixed"
                suggested_settings["preset"] = "dialog_leveler"
                suggested_settings["compression"] = {
                    "threshold": max(-26, min(-18, rms_db - 4)),
                    "ratio": 2.0,
                    "attack": 0.02,
                    "release": 0.15
                }
                suggested_settings["limiting"] = True
                suggested_settings["target_loudness"] = -16
            
            # Determine if gating is needed (for noisy recordings)
            noise_floor_db = rms_db - dynamic_range
            if noise_floor_db > -50:
                suggested_settings["gating"] = True
                suggested_settings["gate_threshold"] = max(-70, min(-50, noise_floor_db - 10))
            else:
                suggested_settings["gating"] = False
            
            # Return analysis results
            return {
                "status": "success",
                "levels": {
                    "peak_db": peak_db,
                    "rms_db": rms_db,
                    "loudness_LUFS": loudness,
                    "crest_factor_db": crest_factor_db,
                    "dynamic_range_db": dynamic_range,
                    "noise_floor_db": noise_floor_db
                },
                "spectral_balance": {
                    "low_pct": low_energy_pct,
                    "mid_pct": mid_energy_pct,
                    "high_pct": high_energy_pct
                },
                "content_type": {
                    "is_speech": is_speech,
                    "is_music": is_music,
                    "is_mixed": is_mixed
                },
                "suggested_settings": suggested_settings
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dynamics: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 