"""
Audio Analyzer Module

This module provides the foundation for audio analysis in the Clip Generation Service,
enabling spectral analysis, silence detection, and filler sound identification.
"""

import os
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Represents a segment of audio with specific characteristics."""
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    segment_type: str  # Type of segment (silence, speech, music, etc.)
    confidence: float = 1.0  # Confidence score for the classification
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    @property
    def duration(self) -> float:
        """Get the duration of the segment in seconds."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "start": self.start_time,
            "end": self.end_time,
            "type": self.segment_type,
            "duration": self.duration,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    def overlaps(self, other: 'AudioSegment') -> bool:
        """Check if this segment overlaps with another segment."""
        return (self.start_time < other.end_time and 
                self.end_time > other.start_time)
    
    def merge(self, other: 'AudioSegment') -> 'AudioSegment':
        """Merge this segment with another overlapping segment."""
        if not self.overlaps(other):
            raise ValueError("Cannot merge non-overlapping segments")
        
        # Take the wider time range
        start_time = min(self.start_time, other.start_time)
        end_time = max(self.end_time, other.end_time)
        
        # Keep the segment type with higher confidence
        if self.confidence >= other.confidence:
            segment_type = self.segment_type
            confidence = self.confidence
            metadata = self.metadata.copy()
            # Merge metadata from other segment
            metadata.update(other.metadata)
        else:
            segment_type = other.segment_type
            confidence = other.confidence
            metadata = other.metadata.copy()
            # Merge metadata from this segment
            metadata.update(self.metadata)
        
        return AudioSegment(
            start_time=start_time,
            end_time=end_time,
            segment_type=segment_type,
            confidence=confidence,
            metadata=metadata
        )


class AudioAnalysisConfig:
    """Configuration for audio analysis."""
    
    def __init__(
        self,
        min_silence_duration: float = 0.3,  # Minimum silence duration to detect
        max_silence_duration: float = 2.0,  # Maximum silence duration to keep
        silence_threshold: float = -35,  # dB threshold for silence
        adaptive_threshold: bool = True,  # Use adaptive threshold
        vad_mode: int = 3,  # VAD aggressiveness (0-3)
        enable_speaker_diarization: bool = False,  # Enable speaker diarization
        enable_noise_profiling: bool = True,  # Enable background noise profiling
        enable_filler_detection: bool = True,  # Enable filler word detection
        language: str = "en",  # Language for filler word detection
        temp_dir: Optional[str] = None,  # Directory for temporary files
        ffmpeg_path: Optional[str] = None,  # Path to ffmpeg binary
        device: str = "cpu"  # Device for ML inference
    ):
        """
        Initialize audio analysis configuration.
        
        Args:
            min_silence_duration: Minimum silence duration to detect (seconds)
            max_silence_duration: Maximum silence duration to keep (seconds)
            silence_threshold: dB threshold for silence detection
            adaptive_threshold: Use adaptive threshold based on audio content
            vad_mode: VAD aggressiveness level (0=least aggressive, 3=most aggressive)
            enable_speaker_diarization: Enable identification of different speakers
            enable_noise_profiling: Enable background noise profiling
            enable_filler_detection: Enable detection of filler words
            language: Language code for speech processing
            temp_dir: Directory for temporary files
            ffmpeg_path: Path to ffmpeg binary
            device: Device for ML inference (cpu, cuda)
        """
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        self.silence_threshold = silence_threshold
        self.adaptive_threshold = adaptive_threshold
        self.vad_mode = vad_mode
        self.enable_speaker_diarization = enable_speaker_diarization
        self.enable_noise_profiling = enable_noise_profiling
        self.enable_filler_detection = enable_filler_detection
        self.language = language
        
        # Set temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "clip_generation" / "audio_analysis"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set ffmpeg path
        if ffmpeg_path:
            self.ffmpeg_path = ffmpeg_path
        else:
            # Try to find ffmpeg in system path
            try:
                result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
                if result.returncode == 0:
                    self.ffmpeg_path = result.stdout.strip()
                else:
                    self.ffmpeg_path = "ffmpeg"  # Default to system path
            except Exception:
                self.ffmpeg_path = "ffmpeg"  # Default to system path
        
        self.device = device
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "min_silence_duration": self.min_silence_duration,
            "max_silence_duration": self.max_silence_duration,
            "silence_threshold": self.silence_threshold,
            "adaptive_threshold": self.adaptive_threshold,
            "vad_mode": self.vad_mode,
            "enable_speaker_diarization": self.enable_speaker_diarization,
            "enable_noise_profiling": self.enable_noise_profiling,
            "enable_filler_detection": self.enable_filler_detection,
            "language": self.language,
            "temp_dir": str(self.temp_dir),
            "ffmpeg_path": self.ffmpeg_path,
            "device": self.device
        }


class AudioAnalyzer:
    """
    Base class for audio analysis in the Clip Generation Service.
    
    This class provides the foundation for:
    - Loading and processing audio from video files
    - Spectral analysis and background noise profiling
    - Voice activity detection (VAD)
    - Speaker diarization
    - Non-speech sound classification
    - Intelligent silence detection
    - Filler sound identification
    """
    
    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        """
        Initialize the audio analyzer.
        
        Args:
            config: Audio analysis configuration
        """
        self.config = config or AudioAnalysisConfig()
        
        # Internal state
        self._audio_data = None
        self._sample_rate = None
        self._duration = None
        self._noise_profile = None
        self._segments = []
        
        logger.info(f"Initialized AudioAnalyzer with config: {self.config.to_dict()}")
    
    def load_audio(self, file_path: str) -> bool:
        """
        Load audio from a file.
        
        Args:
            file_path: Path to audio or video file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import librosa
            
            logger.info(f"Loading audio from {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Load audio using librosa
            self._audio_data, self._sample_rate = librosa.load(
                file_path, 
                sr=None,  # Use original sample rate
                mono=True  # Convert to mono
            )
            
            # Calculate duration
            self._duration = librosa.get_duration(
                y=self._audio_data, 
                sr=self._sample_rate
            )
            
            logger.info(f"Loaded audio: {self._duration:.2f}s, {self._sample_rate}Hz")
            return True
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            return False
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to video file
            output_path: Path to save extracted audio (optional)
            
        Returns:
            Path to extracted audio file, or None if failed
        """
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None
            
            # Generate output path if not provided
            if output_path is None:
                video_filename = os.path.basename(video_path)
                video_name = os.path.splitext(video_filename)[0]
                output_path = str(self.config.temp_dir / f"{video_name}_audio.wav")
            
            # Extract audio using ffmpeg
            cmd = [
                self.config.ffmpeg_path,
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
    
    def analyze(self, audio_path: str) -> List[AudioSegment]:
        """
        Analyze audio to detect silences, speech, and other sound types.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of audio segments
        """
        # Load the audio
        if not self.load_audio(audio_path):
            return []
        
        # Analysis pipeline
        self._segments = []
        
        # 1. Perform background noise profiling
        if self.config.enable_noise_profiling:
            self._analyze_background_noise()
        
        # 2. Perform voice activity detection
        self._perform_vad()
        
        # 3. Classify non-speech sounds
        self._classify_non_speech()
        
        # 4. Perform speaker diarization if enabled
        if self.config.enable_speaker_diarization:
            self._perform_speaker_diarization()
        
        # 5. Detect filler sounds if enabled
        if self.config.enable_filler_detection:
            self._detect_filler_sounds()
        
        # 6. Post-process segments
        self._post_process_segments()
        
        return self._segments
    
    def _analyze_background_noise(self) -> None:
        """
        Analyze background noise profile.
        
        This method identifies the ambient noise level in the audio
        to set adaptive thresholds for silence detection.
        """
        logger.info("Analyzing background noise profile")
        
        try:
            # This is a placeholder for the actual implementation
            # In a real implementation, we would:
            # 1. Find segments with lowest energy
            # 2. Analyze their spectral characteristics
            # 3. Create a noise profile to use for adaptive thresholding
            
            # For now, we'll use a simple approach based on percentiles
            if self._audio_data is not None:
                import librosa
                
                # Calculate energy
                energy = librosa.feature.rms(y=self._audio_data)[0]
                
                # Estimate noise floor as the 10th percentile of energy
                noise_floor = np.percentile(energy, 10)
                
                # Store noise profile
                self._noise_profile = {
                    "noise_floor_db": 20 * np.log10(noise_floor + 1e-10),
                    "energy_percentiles": {
                        "10": np.percentile(energy, 10),
                        "25": np.percentile(energy, 25),
                        "50": np.percentile(energy, 50),
                        "75": np.percentile(energy, 75),
                        "90": np.percentile(energy, 90)
                    }
                }
                
                logger.info(f"Noise floor: {self._noise_profile['noise_floor_db']:.2f} dB")
                
                # Adjust silence threshold if adaptive threshold is enabled
                if self.config.adaptive_threshold and self._noise_profile is not None:
                    # Set threshold 10dB above noise floor
                    self.config.silence_threshold = self._noise_profile["noise_floor_db"] + 10
                    logger.info(f"Adaptive silence threshold: {self.config.silence_threshold:.2f} dB")
        
        except Exception as e:
            logger.error(f"Error analyzing background noise: {str(e)}")
    
    def _perform_vad(self) -> None:
        """
        Perform Voice Activity Detection (VAD).
        
        This method identifies segments with speech activity.
        """
        logger.info("Performing Voice Activity Detection (VAD)")
        
        try:
            # This is a placeholder for the actual implementation
            # In a real implementation, we would:
            # 1. Apply a VAD algorithm (e.g., webrtcvad, Silero VAD)
            # 2. Identify speech segments with timestamps
            # 3. Add them to the segments list
            
            # For now, we'll use a simple energy-based approach
            if self._audio_data is not None and self._sample_rate is not None:
                import librosa
                
                # Calculate energy
                energy = librosa.feature.rms(y=self._audio_data)[0]
                
                # Convert to dB
                energy_db = 20 * np.log10(energy + 1e-10)
                
                # Time frames
                frames = np.arange(len(energy)) * 512 / self._sample_rate
                
                # Apply threshold
                threshold = self.config.silence_threshold
                is_speech = energy_db > threshold
                
                # Find speech segments
                speech_segments = self._find_contiguous_segments(is_speech, frames)
                
                # Filter short segments
                speech_segments = [
                    segment for segment in speech_segments 
                    if segment[1] - segment[0] >= self.config.min_silence_duration
                ]
                
                # Add speech segments
                for start, end in speech_segments:
                    self._segments.append(AudioSegment(
                        start_time=start,
                        end_time=end,
                        segment_type="speech",
                        confidence=0.8,  # Placeholder
                        metadata={"energy_db": float(np.mean(energy_db))}
                    ))
                
                # Add silence segments (gaps between speech)
                if speech_segments:
                    # Add initial silence if needed
                    if speech_segments[0][0] > 0:
                        self._segments.append(AudioSegment(
                            start_time=0,
                            end_time=speech_segments[0][0],
                            segment_type="silence",
                            confidence=0.9,
                            metadata={}
                        ))
                    
                    # Add silences between speech segments
                    for i in range(len(speech_segments) - 1):
                        self._segments.append(AudioSegment(
                            start_time=speech_segments[i][1],
                            end_time=speech_segments[i+1][0],
                            segment_type="silence",
                            confidence=0.9,
                            metadata={}
                        ))
                    
                    # Add final silence if needed
                    if speech_segments[-1][1] < self._duration:
                        self._segments.append(AudioSegment(
                            start_time=speech_segments[-1][1],
                            end_time=self._duration,
                            segment_type="silence",
                            confidence=0.9,
                            metadata={}
                        ))
                else:
                    # No speech detected, mark entire audio as silence
                    self._segments.append(AudioSegment(
                        start_time=0,
                        end_time=self._duration,
                        segment_type="silence",
                        confidence=0.9,
                        metadata={}
                    ))
                
                logger.info(f"VAD identified {len([s for s in self._segments if s.segment_type == 'speech'])} speech segments")
                logger.info(f"VAD identified {len([s for s in self._segments if s.segment_type == 'silence'])} silence segments")
        
        except Exception as e:
            logger.error(f"Error performing VAD: {str(e)}")
    
    def _find_contiguous_segments(self, mask: np.ndarray, times: np.ndarray) -> List[Tuple[float, float]]:
        """
        Find contiguous segments of True values in a boolean mask.
        
        Args:
            mask: Boolean mask of segments
            times: Time values corresponding to each mask position
            
        Returns:
            List of (start_time, end_time) tuples
        """
        if len(mask) == 0:
            return []
        
        # Find indices where mask changes
        change_points = np.where(np.diff(np.concatenate(([False], mask, [False]))))[0]
        
        # Pair start and end indices
        pairs = change_points.reshape(-1, 2)
        
        # Convert to time values
        segments = []
        for start_idx, end_idx in pairs:
            if start_idx < len(times) and end_idx - 1 < len(times):
                segments.append((times[start_idx], times[end_idx - 1]))
        
        return segments
    
    def _classify_non_speech(self) -> None:
        """
        Classify non-speech sounds.
        
        This method identifies different types of non-speech sounds
        (music, effects, ambient noise, etc.)
        """
        logger.info("Classifying non-speech sounds")
        
        # Placeholder for actual implementation
        # In a real implementation, we would:
        # 1. Use a classifier to identify music, effects, ambient noise
        # 2. Update the segment types accordingly
        pass
    
    def _perform_speaker_diarization(self) -> None:
        """
        Perform speaker diarization.
        
        This method identifies different speakers in the audio.
        """
        logger.info("Performing speaker diarization")
        
        # Placeholder for actual implementation
        # In a real implementation, we would:
        # 1. Use a diarization model to cluster speech segments by speaker
        # 2. Update the speech segments with speaker IDs
        pass
    
    def _detect_filler_sounds(self) -> None:
        """
        Detect filler sounds.
        
        This method identifies filler words, hesitations, etc.
        """
        logger.info("Detecting filler sounds")
        
        # Placeholder for actual implementation
        # In a real implementation, we would:
        # 1. Use ASR to transcribe speech segments
        # 2. Identify filler words and hesitations
        # 3. Mark the corresponding segments
        pass
    
    def _post_process_segments(self) -> None:
        """
        Post-process segments.
        
        This method applies rules to refine the segments:
        1. Merge adjacent segments of the same type
        2. Filter out segments that are too short
        3. Apply contextual rules for silence preservation
        """
        logger.info("Post-processing segments")
        
        if not self._segments:
            return
        
        # Sort segments by start time
        self._segments.sort(key=lambda s: s.start_time)
        
        # Merge adjacent segments of the same type
        merged_segments = []
        current_segment = self._segments[0]
        
        for segment in self._segments[1:]:
            # If segment types match and they're adjacent or overlapping
            if (segment.segment_type == current_segment.segment_type and 
                segment.start_time <= current_segment.end_time + 0.05):  # Allow small gaps (50ms)
                
                # Merge by extending the end time
                current_segment.end_time = max(current_segment.end_time, segment.end_time)
            else:
                # Add the current segment and start a new one
                merged_segments.append(current_segment)
                current_segment = segment
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        # Apply silence duration rules
        processed_segments = []
        
        for segment in merged_segments:
            # Skip silences that are too short
            if segment.segment_type == "silence" and segment.duration < self.config.min_silence_duration:
                continue
                
            # Cap silences that are too long
            if segment.segment_type == "silence" and segment.duration > self.config.max_silence_duration:
                # Only keep the first part of the silence
                segment.end_time = segment.start_time + self.config.max_silence_duration
            
            processed_segments.append(segment)
        
        self._segments = processed_segments
        logger.info(f"After post-processing: {len(self._segments)} segments")
    
    def get_segments(self) -> List[AudioSegment]:
        """
        Get the list of audio segments.
        
        Returns:
            List of audio segments
        """
        return self._segments
    
    def get_silence_segments(self) -> List[AudioSegment]:
        """
        Get silence segments.
        
        Returns:
            List of silence segments
        """
        return [segment for segment in self._segments if segment.segment_type == "silence"]
    
    def get_speech_segments(self) -> List[AudioSegment]:
        """
        Get speech segments.
        
        Returns:
            List of speech segments
        """
        return [segment for segment in self._segments if segment.segment_type == "speech"]
    
    def get_filler_segments(self) -> List[AudioSegment]:
        """
        Get filler sound segments.
        
        Returns:
            List of filler sound segments
        """
        return [segment for segment in self._segments if segment.segment_type == "filler"]
    
    def get_segments_by_type(self, segment_type: str) -> List[AudioSegment]:
        """
        Get segments of a specific type.
        
        Args:
            segment_type: Type of segments to retrieve
            
        Returns:
            List of segments matching the type
        """
        return [segment for segment in self._segments if segment.segment_type == segment_type]
    
    def get_removable_segments(self) -> List[AudioSegment]:
        """
        Get segments that can be removed from the audio.
        
        This includes:
        - Extended silences (beyond max_silence_duration)
        - Filler sounds
        - Other non-speech sounds marked for removal
        
        Returns:
            List of removable segments
        """
        removable = []
        
        # Add silence segments beyond the max duration
        for segment in self.get_silence_segments():
            if segment.duration > self.config.max_silence_duration:
                # Create a segment for the excess silence
                excess_start = segment.start_time + self.config.max_silence_duration
                removable.append(AudioSegment(
                    start_time=excess_start,
                    end_time=segment.end_time,
                    segment_type="excess_silence",
                    confidence=segment.confidence,
                    metadata=segment.metadata
                ))
        
        # Add all filler segments
        removable.extend(self.get_filler_segments())
        
        # Add other segments marked for removal
        for segment in self._segments:
            if segment.metadata.get("removable", False):
                removable.append(segment)
        
        # Sort by start time
        removable.sort(key=lambda s: s.start_time)
        
        return removable 