"""
Voice Enhancer Module

This module provides functionality for enhancing voice quality in audio recordings
by applying various processing techniques such as equalization, compression, 
de-essing, and spectral enhancement.
"""

import os
import numpy as np
import logging
import tempfile
import librosa
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple
from scipy import signal
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

class VoiceEnhancer:
    """
    A class for enhancing voice clarity and quality in audio recordings.
    
    This class provides methods to improve the intelligibility and quality of
    voice recordings using techniques like equalization, compression, de-essing,
    and harmonic enhancement.
    """
    
    def __init__(
        self,
        eq_low_shelf_gain: float = -2.0,
        eq_high_shelf_gain: float = 3.0,
        eq_presence_gain: float = 4.0,
        eq_presence_freq: float = 3500,
        eq_presence_q: float = 0.8,
        eq_high_shelf_freq: float = 7500,
        eq_low_shelf_freq: float = 200,
        compression_threshold: float = -18.0,
        compression_ratio: float = 2.5,
        compression_attack: float = 0.01,
        compression_release: float = 0.15,
        harmonic_enhancement: float = 0.3,
        male_voice_boost: bool = False,
        female_voice_boost: bool = False,
        de_essing_strength: float = 0.5,
        warmth: float = 0.3,
        clarity: float = 0.4,
    ):
        """
        Initialize the VoiceEnhancer with customizable parameters.
        
        Args:
            eq_low_shelf_gain: Gain for low shelf EQ in dB
            eq_high_shelf_gain: Gain for high shelf EQ in dB
            eq_presence_gain: Gain for presence boost EQ in dB
            eq_presence_freq: Center frequency for presence boost
            eq_presence_q: Q factor for presence boost
            eq_high_shelf_freq: Frequency for high shelf filter
            eq_low_shelf_freq: Frequency for low shelf filter
            compression_threshold: Threshold in dB for compressor
            compression_ratio: Ratio for compressor
            compression_attack: Attack time in seconds
            compression_release: Release time in seconds
            harmonic_enhancement: Amount of harmonic enhancement (0.0-1.0)
            male_voice_boost: Whether to apply EQ optimized for male voices
            female_voice_boost: Whether to apply EQ optimized for female voices
            de_essing_strength: Strength of de-essing (0.0-1.0)
            warmth: Amount of warmth to add (0.0-1.0)
            clarity: Amount of clarity to add (0.0-1.0)
        """
        # Store parameters
        self.eq_low_shelf_gain = eq_low_shelf_gain
        self.eq_high_shelf_gain = eq_high_shelf_gain
        self.eq_presence_gain = eq_presence_gain
        self.eq_presence_freq = eq_presence_freq
        self.eq_presence_q = eq_presence_q
        self.eq_high_shelf_freq = eq_high_shelf_freq
        self.eq_low_shelf_freq = eq_low_shelf_freq
        self.compression_threshold = compression_threshold
        self.compression_ratio = compression_ratio
        self.compression_attack = compression_attack
        self.compression_release = compression_release
        self.harmonic_enhancement = harmonic_enhancement
        self.male_voice_boost = male_voice_boost
        self.female_voice_boost = female_voice_boost
        self.de_essing_strength = de_essing_strength
        self.warmth = warmth
        self.clarity = clarity
        
        # Validate that we don't have both male and female boost active
        if self.male_voice_boost and self.female_voice_boost:
            logger.warning("Both male and female voice boost enabled; defaulting to male")
            self.female_voice_boost = False
            
    def enhance_voice(
        self,
        audio_path: str,
        output_path: str,
        apply_eq: bool = True,
        apply_compression: bool = True,
        apply_de_essing: bool = True,
        apply_harmonic_enhancement: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhance the voice in an audio file.
        
        Args:
            audio_path: Path to the input audio file
            output_path: Path to save the enhanced audio
            apply_eq: Whether to apply equalization
            apply_compression: Whether to apply compression
            apply_de_essing: Whether to apply de-essing
            apply_harmonic_enhancement: Whether to apply harmonic enhancement
            
        Returns:
            Dictionary with process info and statistics
        """
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Keep track of processing steps
            processing_steps = []
            
            # Apply processing in order
            if apply_eq:
                y = self._apply_equalization(y, sr)
                processing_steps.append("equalization")
                
            if apply_compression:
                y = self._apply_compression(y, sr)
                processing_steps.append("compression")
                
            if apply_de_essing:
                y = self._apply_de_essing(y, sr)
                processing_steps.append("de_essing")
                
            if apply_harmonic_enhancement:
                y = self._apply_harmonic_enhancement(y, sr)
                processing_steps.append("harmonic_enhancement")
                
            # Apply warmth if requested
            if self.warmth > 0:
                y = self._add_warmth(y, sr)
                processing_steps.append("warmth")
                
            # Apply clarity if requested
            if self.clarity > 0:
                y = self._add_clarity(y, sr)
                processing_steps.append("clarity")
                
            # Normalize the output
            y = librosa.util.normalize(y)
            processing_steps.append("normalization")
            
            # Save the processed audio
            sf.write(output_path, y, sr)
            
            # Return processing info
            return {
                "status": "success",
                "input_path": audio_path,
                "output_path": output_path,
                "processing_steps": processing_steps,
                "sample_rate": sr,
                "duration": librosa.get_duration(y=y, sr=sr)
            }
            
        except Exception as e:
            logger.error(f"Error enhancing voice: {str(e)}")
            return {
                "status": "error",
                "input_path": audio_path,
                "error": str(e)
            }
        
    def _apply_equalization(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply equalization to enhance voice frequencies.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Equalized audio signal
        """
        # Apply different EQ curves based on voice type
        if self.male_voice_boost:
            # Male voice EQ: boost around 120Hz for depth, cut around 500Hz,
            # boost around 4-5kHz for presence
            low_shelf_freq = 120
            presence_freq = 4500
            cut_freq = 500
            
            # Create low shelf filter for depth
            sos_low = signal.butter(2, low_shelf_freq, btype='lowshelf', 
                                   fs=sr, output='sos', gain=2.0)
            y = signal.sosfilt(sos_low, y)
            
            # Create cut filter to reduce muddiness
            sos_cut = signal.butter(2, [400, 600], btype='bandstop', 
                                   fs=sr, output='sos')
            y = signal.sosfilt(sos_cut, y)
            
        elif self.female_voice_boost:
            # Female voice EQ: less low end, boost around 200-300Hz for warmth,
            # boost higher around 5-6kHz for presence
            low_shelf_freq = 180
            presence_freq = 5500
            
            # Create band for warmth
            sos_warm = signal.butter(2, [180, 300], btype='bandpass', 
                                    fs=sr, output='sos')
            y_warm = signal.sosfilt(sos_warm, y)
            y = y + (y_warm * 0.3)  # Blend in warmth
        
        else:
            # Standard voice EQ
            low_shelf_freq = self.eq_low_shelf_freq
            presence_freq = self.eq_presence_freq
        
        # Common EQ for all voice types
        # 1. Low shelf filter to reduce rumble
        sos_low_cut = signal.butter(2, low_shelf_freq, btype='lowshelf', 
                                   fs=sr, output='sos', gain=self.eq_low_shelf_gain)
        y = signal.sosfilt(sos_low_cut, y)
        
        # 2. Presence boost for clarity
        q = self.eq_presence_q
        bw = presence_freq / q
        sos_presence = signal.butter(2, [presence_freq - bw/2, presence_freq + bw/2], 
                                    btype='bandpass', fs=sr, output='sos')
        y_presence = signal.sosfilt(sos_presence, y)
        y = y + (y_presence * (10**(self.eq_presence_gain/20) - 1))
        
        # 3. High shelf for air
        sos_high = signal.butter(2, self.eq_high_shelf_freq, btype='highshelf', 
                                fs=sr, output='sos', gain=self.eq_high_shelf_gain)
        y = signal.sosfilt(sos_high, y)
        
        return y
        
    def _apply_compression(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply dynamic range compression to the voice signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Compressed audio signal
        """
        # Convert parameters
        threshold = 10 ** (self.compression_threshold / 20)
        attack_samples = int(self.compression_attack * sr)
        release_samples = int(self.compression_release * sr)
        
        # Compute signal envelope
        y_abs = np.abs(y)
        envelope = np.zeros_like(y_abs)
        
        # Attack and release envelope follower
        for i in range(1, len(y_abs)):
            if y_abs[i] > envelope[i-1]:
                envelope[i] = y_abs[i] + (envelope[i-1] - y_abs[i]) * np.exp(-1 / attack_samples)
            else:
                envelope[i] = y_abs[i] + (envelope[i-1] - y_abs[i]) * np.exp(-1 / release_samples)
        
        # Compute gain reduction
        gain_reduction = np.ones_like(envelope)
        mask = envelope > threshold
        gain_reduction[mask] = threshold + (envelope[mask] - threshold) / self.compression_ratio
        gain_reduction[mask] /= envelope[mask]
        
        # Smooth gain reduction to avoid artifacts
        gain_reduction = gaussian_filter1d(gain_reduction, sigma=attack_samples/4)
        
        # Apply gain reduction
        y_compressed = y * gain_reduction
        
        # Apply makeup gain to bring level back up
        makeup_gain = 1 / (threshold ** (1 - 1/self.compression_ratio))
        y_compressed *= makeup_gain * 0.95  # Add a safety factor to avoid clipping
        
        return y_compressed
        
    def _apply_de_essing(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply de-essing to reduce sibilance in voice recordings.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            De-essed audio signal
        """
        if self.de_essing_strength <= 0:
            return y
        
        # Sibilance is typically in the 5-8kHz range
        lower_freq = 5000
        upper_freq = 8000
        
        # Create a bandpass filter for sibilance
        sos_sibilance = signal.butter(4, [lower_freq, upper_freq], btype='bandpass', 
                                      fs=sr, output='sos')
        
        # Extract sibilance
        y_sibilance = signal.sosfilt(sos_sibilance, y)
        
        # Compute envelope of sibilance
        y_sib_env = np.abs(y_sibilance)
        y_sib_env = gaussian_filter1d(y_sib_env, sigma=int(0.005 * sr))  # 5ms smoothing
        
        # Determine threshold (adjust as needed)
        threshold = np.mean(y_sib_env) * 2.5
        
        # Compute gain reduction for sibilance
        gain_reduction = np.ones_like(y_sib_env)
        mask = y_sib_env > threshold
        
        # Apply gain reduction based on strength
        gain_reduction[mask] = threshold + (y_sib_env[mask] - threshold) / (1 + self.de_essing_strength * 5)
        gain_reduction[mask] /= y_sib_env[mask]
        
        # Smooth gain changes
        gain_reduction = gaussian_filter1d(gain_reduction, sigma=int(0.01 * sr))  # 10ms smoothing
        
        # Create sibilance-reduced version by applying frequency-dependent gain reduction
        # First, create a full spectrum copy with reduced sibilance
        y_sib_reduced = y_sibilance * gain_reduction
        
        # Then, filter out the sibilance frequencies from the original
        sos_notch = signal.butter(4, [lower_freq, upper_freq], btype='bandstop', 
                                  fs=sr, output='sos')
        y_no_sib = signal.sosfilt(sos_notch, y)
        
        # Combine the band-stopped signal with the processed sibilance
        y_processed = y_no_sib + y_sib_reduced
        
        return y_processed
    
    def _apply_harmonic_enhancement(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply harmonic enhancement to add presence and clarity to voice.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Harmonically enhanced audio signal
        """
        if self.harmonic_enhancement <= 0:
            return y
        
        # Create a subtle harmonic distortion
        # This is a simplified version of a more complex harmonic exciter
        
        # First, high-pass the signal to focus on mid-high frequencies
        sos_hp = signal.butter(2, 1000, btype='highpass', fs=sr, output='sos')
        y_high = signal.sosfilt(sos_hp, y)
        
        # Apply soft saturation to generate harmonics
        # Using a cubic soft-clipper
        drive = 1.5 + (self.harmonic_enhancement * 3)  # Scale enhancement to drive amount
        y_harm = np.tanh(y_high * drive)
        
        # Remove DC offset that might have been introduced
        y_harm = y_harm - np.mean(y_harm)
        
        # Filter the harmonics to focus on the desirable frequencies
        sos_harm_filter = signal.butter(2, [1500, 10000], btype='bandpass', fs=sr, output='sos')
        y_harm = signal.sosfilt(sos_harm_filter, y_harm)
        
        # Mix back with original
        mix_amount = self.harmonic_enhancement * 0.5  # Scale down to avoid overdoing it
        y_enhanced = y + (y_harm * mix_amount)
        
        # Normalize to avoid clipping
        y_enhanced = y_enhanced / (np.max(np.abs(y_enhanced)) + 1e-8)
        
        return y_enhanced
    
    def _add_warmth(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Add warmth to the voice by enhancing lower harmonics.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Warmer audio signal
        """
        if self.warmth <= 0:
            return y
        
        # Extract lower mids (200-500Hz)
        sos_warm = signal.butter(2, [200, 600], btype='bandpass', fs=sr, output='sos')
        y_warm = signal.sosfilt(sos_warm, y)
        
        # Apply subtle saturation for harmonic richness
        y_warm = np.tanh(y_warm * 2.0)
        
        # Mix back with original
        mix_ratio = self.warmth * 0.3  # Adjust mix ratio based on warmth parameter
        y_processed = y + (y_warm * mix_ratio)
        
        # Normalize
        y_processed = librosa.util.normalize(y_processed)
        
        return y_processed
    
    def _add_clarity(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Add clarity to the voice by enhancing upper mids.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Clearer audio signal
        """
        if self.clarity <= 0:
            return y
        
        # Boost upper mids (2-5kHz) for clarity
        sos_clarity = signal.butter(2, [2000, 5000], btype='bandpass', fs=sr, output='sos')
        y_clarity = signal.sosfilt(sos_clarity, y)
        
        # Mix back with original
        mix_ratio = self.clarity * 0.4  # Adjust mix ratio based on clarity parameter
        y_processed = y + (y_clarity * mix_ratio)
        
        # Normalize
        y_processed = librosa.util.normalize(y_processed)
        
        return y_processed
        
    def analyze_voice(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze voice characteristics in audio file to suggest enhancements.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with voice analysis and enhancement suggestions
        """
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Calculate spectrum
            S = np.abs(librosa.stft(y))
            
            # Convert to dB scale
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            
            # Calculate frequency bands
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            
            # Define ranges for analysis
            ranges = {
                'low': (80, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 5000),
                'high': (5000, 10000),
                'presence': (3000, 5000),
                'sibilance': (5000, 8000)
            }
            
            # Analyze energy in each range
            range_energy = {}
            for name, (low, high) in ranges.items():
                low_idx = np.argmin(np.abs(freq_bins - low))
                high_idx = np.argmin(np.abs(freq_bins - high))
                energy = np.mean(S_db[:, low_idx:high_idx+1])
                range_energy[name] = energy
            
            # Detect sibilance
            sibilance_energy = range_energy['sibilance']
            presence_energy = range_energy['presence']
            sibilance_ratio = 10 ** ((sibilance_energy - presence_energy) / 20)
            has_sibilance = sibilance_ratio > 1.1
            
            # Estimate if it's male or female voice (very simplistic)
            low_energy = range_energy['low']
            high_mid_energy = range_energy['high_mid']
            likely_male = low_energy > (high_mid_energy - 6)  # Male voices have stronger low end
            
            # Calculate dynamic range
            rms = librosa.feature.rms(y=y)[0]
            dynamic_range = 20 * np.log10(np.max(rms) / (np.mean(rms) + 1e-8))
            needs_compression = dynamic_range > 20  # If dynamic range is high
            
            # Analyze clarity
            clarity_score = high_mid_energy - range_energy['mid']
            needs_clarity = clarity_score < -10
            
            # Generate suggestions
            suggestions = []
            if has_sibilance:
                suggestions.append("Apply de-essing to reduce sibilance")
            
            if needs_compression:
                suggestions.append("Apply compression to control dynamics")
            
            if needs_clarity:
                suggestions.append("Boost presence frequencies for clarity")
            
            if likely_male:
                suggestions.append("Apply male voice EQ profile")
            else:
                suggestions.append("Apply female voice EQ profile")
            
            # Return analysis results
            return {
                "status": "success",
                "frequency_balance": range_energy,
                "has_sibilance": has_sibilance,
                "dynamic_range_db": dynamic_range,
                "likely_male_voice": likely_male,
                "needs_compression": needs_compression,
                "needs_clarity": needs_clarity,
                "suggestions": suggestions,
                "recommended_settings": {
                    "eq_presence_gain": 4.0 if needs_clarity else 2.0,
                    "compression_ratio": 3.0 if needs_compression else 2.0,
                    "de_essing_strength": 0.7 if has_sibilance else 0.3,
                    "male_voice_boost": likely_male,
                    "female_voice_boost": not likely_male
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing voice: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 