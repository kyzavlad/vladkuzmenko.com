"""
Silence and Unnecessary Audio Detection Module

This module integrates various audio analysis components to detect
silences and unnecessary audio segments that can be removed to
improve the quality of video content.
"""

import os
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from pathlib import Path
import json
import tempfile

from app.clip_generation.services.audio_analysis.audio_analyzer import AudioSegment, AudioAnalysisConfig
from app.clip_generation.services.audio_analysis.vad import VADProcessor
from app.clip_generation.services.audio_analysis.filler_detector import FillerWordDetector
from app.clip_generation.services.audio_analysis.spectral_analyzer import SpectralAnalyzer, SoundType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SilenceDetectorConfig:
    """Configuration for silence and unnecessary audio detection."""
    
    def __init__(
        self,
        # General settings
        min_silence_duration: float = 0.3,  # Minimum silence duration to detect (seconds)
        max_silence_duration: float = 2.0,  # Maximum silence duration to keep (seconds)
        silence_threshold: float = -35,  # dB threshold for silence
        adaptive_threshold: bool = True,  # Use adaptive threshold
        visualize: bool = False,  # Generate visualizations
        
        # VAD settings
        vad_model: str = "silero",  # VAD model to use
        vad_mode: int = 3,  # VAD aggressiveness (0-3)
        vad_threshold: float = 0.5,  # VAD detection threshold
        
        # Filler detection settings
        enable_filler_detection: bool = True,  # Enable filler word detection
        language: str = "en",  # Language for filler detection
        custom_filler_words: Optional[List[str]] = None,  # Custom filler words
        
        # Spectral analysis settings
        enable_spectral_analysis: bool = True,  # Enable spectral analysis
        noise_profiling: bool = True,  # Enable background noise profiling
        
        # Advanced settings
        enable_speaker_diarization: bool = False,  # Enable speaker diarization
        context_window: float = 3.0,  # Context window for analysis (seconds)
        temp_dir: Optional[str] = None,  # Directory for temporary files
        confidence_threshold: float = 0.7,  # Confidence threshold for detection
        model_dir: Optional[str] = None,  # Directory for models
        device: str = "cpu"  # Device for inference
    ):
        """
        Initialize silence detector configuration.
        
        Args:
            min_silence_duration: Minimum silence duration to detect (seconds)
            max_silence_duration: Maximum silence duration to keep (seconds)
            silence_threshold: dB threshold for silence
            adaptive_threshold: Use adaptive threshold based on noise
            visualize: Generate visualizations
            vad_model: VAD model to use (silero, webrtc, pyannote)
            vad_mode: VAD aggressiveness level
            vad_threshold: VAD detection threshold
            enable_filler_detection: Enable filler word detection
            language: Language for filler detection
            custom_filler_words: Custom filler words to detect
            enable_spectral_analysis: Enable spectral analysis
            noise_profiling: Enable background noise profiling
            enable_speaker_diarization: Enable speaker diarization
            context_window: Context window for analysis (seconds)
            temp_dir: Directory for temporary files
            confidence_threshold: Confidence threshold for detection
            model_dir: Directory for models
            device: Device for inference (cpu, cuda)
        """
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        self.silence_threshold = silence_threshold
        self.adaptive_threshold = adaptive_threshold
        self.visualize = visualize
        
        self.vad_model = vad_model
        self.vad_mode = vad_mode
        self.vad_threshold = vad_threshold
        
        self.enable_filler_detection = enable_filler_detection
        self.language = language
        self.custom_filler_words = custom_filler_words or []
        
        self.enable_spectral_analysis = enable_spectral_analysis
        self.noise_profiling = noise_profiling
        
        self.enable_speaker_diarization = enable_speaker_diarization
        self.context_window = context_window
        self.confidence_threshold = confidence_threshold
        
        # Set up temp directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "clip_generation" / "audio_analysis"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up model directory
        self.model_dir = Path(model_dir) if model_dir else None
        
        self.device = device
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "min_silence_duration": self.min_silence_duration,
            "max_silence_duration": self.max_silence_duration,
            "silence_threshold": self.silence_threshold,
            "adaptive_threshold": self.adaptive_threshold,
            "visualize": self.visualize,
            "vad_model": self.vad_model,
            "vad_mode": self.vad_mode,
            "vad_threshold": self.vad_threshold,
            "enable_filler_detection": self.enable_filler_detection,
            "language": self.language,
            "custom_filler_words": self.custom_filler_words,
            "enable_spectral_analysis": self.enable_spectral_analysis,
            "noise_profiling": self.noise_profiling,
            "enable_speaker_diarization": self.enable_speaker_diarization,
            "context_window": self.context_window,
            "temp_dir": str(self.temp_dir),
            "confidence_threshold": self.confidence_threshold,
            "device": self.device
        }


class SilenceDetector:
    """
    Integrated detector for silence and unnecessary audio.
    
    This class combines multiple audio analysis techniques to detect:
    - Silence segments with adaptive thresholds
    - Filler words and hesitations
    - Non-speech sounds (optional removal)
    
    It provides methods to analyze audio and generate segments that
    can be removed to improve content quality.
    """
    
    def __init__(self, config: Optional[SilenceDetectorConfig] = None):
        """
        Initialize the silence detector.
        
        Args:
            config: Configuration for silence detection
        """
        self.config = config or SilenceDetectorConfig()
        
        # Initialize analyzers
        self._initialize_analyzers()
        
        # Result storage
        self.segments = []
        self.removable_segments = []
        self.noise_profile = None
        
        logger.info(f"Initialized SilenceDetector with config: {json.dumps(self.config.to_dict(), indent=2)}")
    
    def _initialize_analyzers(self) -> None:
        """Initialize all analysis components."""
        # VAD processor
        self.vad_processor = VADProcessor(
            model_type=self.config.vad_model,
            vad_mode=self.config.vad_mode,
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=int(self.config.min_silence_duration * 1000),
            min_silence_duration_ms=int(self.config.min_silence_duration * 1000),
            device=self.config.device,
            visualize=self.config.visualize
        )
        
        # Filler word detector (if enabled)
        if self.config.enable_filler_detection:
            self.filler_detector = FillerWordDetector(
                language=self.config.language,
                custom_filler_words=self.config.custom_filler_words,
                model_dir=self.config.model_dir
            )
        else:
            self.filler_detector = None
        
        # Spectral analyzer (if enabled)
        if self.config.enable_spectral_analysis:
            self.spectral_analyzer = SpectralAnalyzer(
                visualize=self.config.visualize,
                model_dir=self.config.model_dir
            )
        else:
            self.spectral_analyzer = None
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from a file.
        
        Args:
            file_path: Path to audio or video file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            import librosa
            
            logger.info(f"Loading audio from {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return np.array([]), 0
            
            # Load audio using librosa
            audio_data, sample_rate = librosa.load(
                file_path, 
                sr=None,  # Use original sample rate
                mono=True  # Convert to mono
            )
            
            duration = len(audio_data) / sample_rate
            logger.info(f"Loaded audio: {duration:.2f}s, {sample_rate}Hz")
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            return np.array([]), 0
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file, or None if failed
        """
        try:
            import subprocess
            
            # Check if video file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None
            
            # Generate output path
            video_filename = os.path.basename(video_path)
            video_name = os.path.splitext(video_filename)[0]
            output_path = str(self.config.temp_dir / f"{video_name}_audio.wav")
            
            # Extract audio using ffmpeg
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-q:a", "0",  # Best quality
                "-map", "a",  # Extract audio only
                "-ac", "1",   # Convert to mono
                output_path
            ]
            
            logger.info(f"Extracting audio from {video_path} to {output_path}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error extracting audio: {result.stderr}")
                return None
            
            logger.info(f"Successfully extracted audio to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return None
    
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Analyze audio to detect silence and unnecessary sounds.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of audio segments
        """
        start_time = time.time()
        logger.info(f"Analyzing audio ({len(audio_data) / sample_rate:.2f}s)")
        
        # Reset results
        self.segments = []
        self.removable_segments = []
        
        # 1. Perform background noise profiling (if enabled)
        if self.config.noise_profiling and self.config.enable_spectral_analysis:
            self.noise_profile = self.spectral_analyzer.profile_background_noise(audio_data, sample_rate)
            
            # Update silence threshold if adaptive threshold is enabled
            if self.config.adaptive_threshold:
                self.config.silence_threshold = self.spectral_analyzer.suggest_silence_threshold(self.noise_profile)
                logger.info(f"Adaptive silence threshold: {self.config.silence_threshold:.2f} dB")
        
        # 2. Perform Voice Activity Detection
        vad_segments = self.vad_processor.process_audio(audio_data, sample_rate)
        self.segments.extend(vad_segments)
        
        # 3. Detect filler sounds (if enabled)
        if self.config.enable_filler_detection and self.filler_detector:
            filler_segments = self.filler_detector.process_audio(audio_data, sample_rate)
            self.segments.extend(filler_segments)
        
        # 4. Perform spectral analysis for non-speech sounds (if enabled)
        if self.config.enable_spectral_analysis and self.spectral_analyzer:
            spectral_segments = self.spectral_analyzer.segment_and_classify(
                audio_data, 
                sample_rate, 
                segment_length_ms=500  # Use smaller segments for fine-grained analysis
            )
            
            # Only add non-speech segments
            for segment in spectral_segments:
                if segment.segment_type not in ["speech", "silence"]:
                    self.segments.append(segment)
        
        # 5. Sort segments by start time
        self.segments.sort(key=lambda s: s.start_time)
        
        # 6. Merge overlapping segments of the same type
        merged_segments = self._merge_segments(self.segments)
        
        # 7. Determine which segments are removable
        self.removable_segments = self._identify_removable_segments(merged_segments)
        
        logger.info(f"Audio analysis completed in {time.time() - start_time:.2f}s")
        logger.info(f"Identified {len(merged_segments)} segments, {len(self.removable_segments)} removable")
        
        return merged_segments
    
    def _merge_segments(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """
        Merge overlapping segments of the same type.
        
        Args:
            segments: List of segments to merge
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        
        merged = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            last = merged[-1]
            
            # If segments overlap and are of the same type
            if (segment.start_time <= last.end_time and 
                segment.segment_type == last.segment_type):
                
                # Update end time to the later of the two
                last.end_time = max(last.end_time, segment.end_time)
                
                # Use higher confidence
                if segment.confidence > last.confidence:
                    last.confidence = segment.confidence
                    
                # Merge metadata
                last.metadata.update(segment.metadata)
                
            else:
                # Add as a new segment
                merged.append(segment)
        
        return merged
    
    def _identify_removable_segments(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """
        Identify segments that can be removed.
        
        Args:
            segments: List of segments to analyze
            
        Returns:
            List of removable segments
        """
        removable = []
        
        for segment in segments:
            # Case 1: Silence segments beyond max duration
            if segment.segment_type == "silence" and segment.duration > self.config.max_silence_duration:
                # Only mark the excess part as removable
                excess_start = segment.start_time + self.config.max_silence_duration
                removable.append(AudioSegment(
                    start_time=excess_start,
                    end_time=segment.end_time,
                    segment_type="excess_silence",
                    confidence=segment.confidence,
                    metadata={
                        "original_duration": segment.duration,
                        "kept_duration": self.config.max_silence_duration,
                        "removal_reason": "silence_too_long"
                    }
                ))
            
            # Case 2: Filler sounds
            elif segment.segment_type == "filler":
                removable.append(segment)
            
            # Case 3: Non-speech sounds with removable flag
            elif segment.segment_type not in ["speech", "silence"]:
                # Only remove if confidence is high enough
                if segment.confidence >= self.config.confidence_threshold:
                    removable.append(AudioSegment(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        segment_type=f"unnecessary_{segment.segment_type}",
                        confidence=segment.confidence,
                        metadata={
                            "original_type": segment.segment_type,
                            "removal_reason": "unnecessary_sound"
                        }
                    ))
            
            # Case 4: Any segment explicitly marked as removable
            elif segment.metadata.get("removable", False):
                removable.append(segment)
        
        # Sort by start time
        removable.sort(key=lambda s: s.start_time)
        
        return removable
    
    def get_segments(self) -> List[AudioSegment]:
        """
        Get all audio segments.
        
        Returns:
            List of all audio segments
        """
        return self.segments
    
    def get_removable_segments(self) -> List[AudioSegment]:
        """
        Get segments that can be removed.
        
        Returns:
            List of removable segments
        """
        return self.removable_segments
    
    def get_segments_by_type(self, segment_type: str) -> List[AudioSegment]:
        """
        Get segments of a specific type.
        
        Args:
            segment_type: Type of segments to retrieve
            
        Returns:
            List of segments matching the type
        """
        return [s for s in self.segments if s.segment_type == segment_type]
    
    def process_file(self, file_path: str) -> Tuple[List[AudioSegment], List[AudioSegment]]:
        """
        Process an audio or video file.
        
        Args:
            file_path: Path to audio or video file
            
        Returns:
            Tuple of (all_segments, removable_segments)
        """
        # Check file type
        is_video = file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
        
        if is_video:
            # Extract audio from video
            audio_path = self.extract_audio_from_video(file_path)
            if not audio_path:
                logger.error(f"Failed to extract audio from {file_path}")
                return [], []
        else:
            # Use audio file directly
            audio_path = file_path
        
        # Load audio
        audio_data, sample_rate = self.load_audio(audio_path)
        if len(audio_data) == 0:
            logger.error(f"Failed to load audio from {audio_path}")
            return [], []
        
        # Analyze audio
        segments = self.analyze_audio(audio_data, sample_rate)
        
        # Clean up extracted audio if needed
        if is_video and os.path.exists(audio_path) and audio_path != file_path:
            try:
                os.unlink(audio_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary audio file: {str(e)}")
        
        return segments, self.removable_segments
    
    def generate_silence_mask(self, audio_length: int, sample_rate: int) -> np.ndarray:
        """
        Generate a boolean mask for removable segments.
        
        This is useful for audio processing operations.
        
        Args:
            audio_length: Length of audio in samples
            sample_rate: Sample rate of the audio
            
        Returns:
            Boolean mask (True = keep, False = remove)
        """
        # Initialize mask (keep everything)
        mask = np.ones(audio_length, dtype=bool)
        
        # Mark removable segments
        for segment in self.removable_segments:
            start_sample = int(segment.start_time * sample_rate)
            end_sample = int(segment.end_time * sample_rate)
            
            # Ensure within bounds
            start_sample = max(0, min(start_sample, audio_length - 1))
            end_sample = max(0, min(end_sample, audio_length - 1))
            
            # Mark as removable (False)
            mask[start_sample:end_sample] = False
        
        return mask
    
    def visualize_results(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """
        Generate visualization of analysis results.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Path to visualization image, or None if failed
        """
        if not self.config.visualize:
            return None
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            import librosa.display
            
            # Create temporary directory for visualizations
            vis_dir = self.config.temp_dir / "visualization"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up figure
            plt.figure(figsize=(15, 8))
            
            # Plot waveform
            plt.subplot(2, 1, 1)
            times = np.arange(len(audio_data)) / sample_rate
            plt.plot(times, audio_data, color='gray', alpha=0.7)
            
            # Add segment markers
            colors = {
                "speech": "green",
                "silence": "blue",
                "filler": "red",
                "excess_silence": "purple",
                "music": "orange",
                "noise": "brown",
                "ambient": "cyan",
                "effect": "magenta"
            }
            
            legend_elements = []
            used_types = set()
            
            for segment in self.segments:
                segment_type = segment.segment_type
                color = colors.get(segment_type, "gray")
                
                # Add to plot
                plt.axvspan(segment.start_time, segment.end_time, 
                           alpha=0.3, color=color)
                
                # Track for legend
                if segment_type not in used_types:
                    used_types.add(segment_type)
                    legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                                       color=color, alpha=0.3,
                                                       label=segment_type))
            
            # Plot removable segments with diagonal pattern
            for segment in self.removable_segments:
                plt.axvspan(segment.start_time, segment.end_time, 
                           alpha=0.5, color='red', hatch='//')
            
            # Add legend for segment types
            if legend_elements:
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                                   color='red', alpha=0.5, hatch='//',
                                                   label='removable'))
                plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Audio Segments")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            
            # Plot spectrogram
            plt.subplot(2, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
            librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Spectrogram with Silence Detection")
            
            # Add segment markers to spectrogram
            for segment in self.removable_segments:
                plt.axvspan(segment.start_time, segment.end_time, 
                           alpha=0.3, color='red', hatch='//')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = int(time.time())
            output_path = vis_dir / f"silence_analysis_{timestamp}.png"
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Analysis visualization saved to {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")
            return None
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a report of the analysis results.
        
        Returns:
            Dictionary with analysis report
        """
        total_duration = 0
        if self.segments:
            # Find the end time of the last segment
            total_duration = max(segment.end_time for segment in self.segments)
        
        removable_duration = sum(segment.duration for segment in self.removable_segments)
        
        # Count segments by type
        segment_counts = {}
        for segment in self.segments:
            segment_type = segment.segment_type
            segment_counts[segment_type] = segment_counts.get(segment_type, 0) + 1
        
        # Removable segments by reason
        removal_reasons = {}
        for segment in self.removable_segments:
            reason = segment.metadata.get("removal_reason", "unspecified")
            removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
        
        return {
            "total_duration": total_duration,
            "total_segments": len(self.segments),
            "removable_segments": len(self.removable_segments),
            "removable_duration": removable_duration,
            "removal_percentage": (removable_duration / total_duration * 100) if total_duration > 0 else 0,
            "segment_counts": segment_counts,
            "removal_reasons": removal_reasons,
            "silence_threshold": self.config.silence_threshold,
            "adaptive_threshold_used": self.config.adaptive_threshold,
            "noise_floor": self.noise_profile.get("noise_floor_db", None) if self.noise_profile else None,
            "timestamp": time.time()
        } 