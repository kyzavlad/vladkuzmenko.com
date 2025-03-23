"""
Content Analysis Module

This module handles the multi-faceted content analysis for detecting interesting moments,
including audio, visual, and textual analysis.
"""

import os
import numpy as np
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from app.clip_generation.services.moment_detection.moment_analyzer import MomentType, MomentScore
from app.clip_generation.services.moment_detection.voice_analysis import VoiceAnalyzer
from app.clip_generation.services.moment_detection.transcript_analysis import TranscriptAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ContentAnalysisConfig:
    """Configuration for content analysis components."""
    # General settings
    temp_dir: str = "temp"
    ffmpeg_path: str = "ffmpeg"
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Audio analysis settings
    audio_sample_rate: int = 16000
    audio_window_size: float = 0.5  # seconds
    audio_window_overlap: float = 0.25  # seconds
    audio_energy_threshold: float = 0.7  # 0.0 to 1.0, percentile of energy distribution
    min_peak_distance: float = 1.0  # seconds
    
    # Voice analysis settings
    voice_emphasis_threshold: float = 0.6  # 0.0 to 1.0
    enable_laughter_detection: bool = True
    laughter_confidence_threshold: float = 0.7  # 0.0 to 1.0
    
    # Transcript analysis settings
    enable_sentiment_analysis: bool = True
    sentiment_window_size: int = 50  # tokens
    sentiment_threshold: float = 0.7  # 0.0 to 1.0
    
    # Visual analysis settings
    enable_gesture_detection: bool = True
    enable_expression_detection: bool = True
    visual_analysis_interval: int = 5  # frames
    expression_intensity_threshold: float = 0.6  # 0.0 to 1.0


class AudioAnalyzer:
    """
    Analyzes audio aspects of content to detect interesting moments.
    
    Features:
    - Audio energy peak detection
    - Voice tone and emphasis analysis
    - Laughter and reaction detection
    """
    
    def __init__(self, config: ContentAnalysisConfig):
        """
        Initialize the audio analyzer.
        
        Args:
            config: Configuration for content analysis
        """
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load required libraries only when needed
        # This allows for modular dependencies
        try:
            import librosa
            self.librosa = librosa
            self._has_librosa = True
        except ImportError:
            logger.warning("Librosa not available. Some audio analysis features will be limited.")
            self._has_librosa = False
        
        logger.info("Initialized AudioAnalyzer")
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file, or None if extraction failed
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Create a temporary file for the extracted audio
        audio_path = self.temp_dir / f"audio_{os.path.basename(video_path)}.wav"
        
        # Build FFmpeg command
        cmd = [
            self.config.ffmpeg_path,
            "-i", video_path,
            "-ac", "1",  # Mono audio
            "-ar", str(self.config.audio_sample_rate),  # Sample rate
            "-vn",  # Disable video
            "-y",  # Overwrite output file
            str(audio_path)
        ]
        
        try:
            # Run FFmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error extracting audio: {stderr}")
                return None
            
            if not os.path.exists(audio_path):
                logger.error(f"Audio extraction failed, output file not created: {audio_path}")
                return None
            
            logger.info(f"Audio extracted successfully: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            logger.error(f"Error running FFmpeg: {str(e)}")
            return None
    
    def detect_audio_energy_peaks(self, audio_path: str) -> List[Tuple[float, float, float]]:
        """
        Detect peaks in audio energy.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of tuples (start_time, end_time, energy_score)
        """
        if not self._has_librosa:
            logger.warning("Librosa not available, using simplified audio energy detection")
            return self._detect_audio_energy_peaks_simple(audio_path)
        
        try:
            # Load audio
            y, sr = self.librosa.load(audio_path, sr=self.config.audio_sample_rate)
            
            # Convert window sizes from seconds to samples
            window_size = int(self.config.audio_window_size * sr)
            hop_length = int((self.config.audio_window_size - self.config.audio_window_overlap) * sr)
            min_peak_distance_frames = int(self.config.min_peak_distance / 
                                           (self.config.audio_window_size - self.config.audio_window_overlap))
            
            # Compute energy
            energy = np.array([
                np.sum(y[i:i+window_size]**2) 
                for i in range(0, len(y)-window_size, hop_length)
            ])
            
            # Normalize energy
            if len(energy) > 0:
                energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
            
            # Find energy threshold
            threshold = np.percentile(energy, self.config.audio_energy_threshold * 100)
            
            # Find peaks above threshold
            from scipy import signal
            peaks, _ = signal.find_peaks(energy, height=threshold, distance=min_peak_distance_frames)
            
            # Convert peaks to time ranges
            time_per_frame = (self.config.audio_window_size - self.config.audio_window_overlap)
            
            results = []
            for peak in peaks:
                # Calculate actual peak energy score (0.0 to 1.0)
                peak_energy = energy[peak]
                
                # Calculate time range
                start_time = max(0, peak - 1) * time_per_frame
                end_time = min(len(energy)-1, peak + 1) * time_per_frame
                
                # Ensure minimum duration
                if end_time - start_time < 1.0:
                    # Extend to at least 1 second
                    mid_point = (start_time + end_time) / 2
                    start_time = mid_point - 0.5
                    end_time = mid_point + 0.5
                
                results.append((start_time, end_time, float(peak_energy)))
            
            logger.info(f"Detected {len(results)} audio energy peaks")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting audio energy peaks: {str(e)}")
            return []
    
    def _detect_audio_energy_peaks_simple(self, audio_path: str) -> List[Tuple[float, float, float]]:
        """
        Simplified version of audio peak detection without librosa.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of tuples (start_time, end_time, energy_score)
        """
        try:
            # This is a placeholder implementation using subprocess
            # to get audio duration and generate simulated peaks
            
            # Get audio duration using ffprobe
            cmd = [
                self.config.ffmpeg_path.replace("ffmpeg", "ffprobe"),
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
            
            # Generate simulated peaks (for demonstration only)
            # In a real implementation, this would analyze the audio file
            import random
            random.seed(42)  # For reproducible demo results
            
            num_segments = int(duration / 5)  # Approximately one peak every 5 seconds
            results = []
            
            for i in range(num_segments):
                # Generate a peak with random energy
                energy = random.uniform(0.6, 1.0)
                
                # Only include peaks above threshold
                if energy >= self.config.audio_energy_threshold:
                    # Calculate time range
                    segment_start = i * 5
                    segment_mid = segment_start + 2.5
                    
                    # Create a 1-2 second window around the peak
                    window_size = random.uniform(1.0, 2.0)
                    start_time = segment_mid - (window_size / 2)
                    end_time = segment_mid + (window_size / 2)
                    
                    # Ensure within bounds
                    start_time = max(0, start_time)
                    end_time = min(duration, end_time)
                    
                    results.append((start_time, end_time, energy))
            
            logger.warning(f"Using simulated audio peaks due to missing libraries. Detected {len(results)} peaks.")
            return results
            
        except Exception as e:
            logger.error(f"Error in simplified audio peak detection: {str(e)}")
            return []
    
    def analyze_audio(self, video_path: str) -> List[Tuple[MomentType, float, float, float, Dict[str, Any]]]:
        """
        Perform audio analysis on a video to detect various audio-based moments.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of tuples (moment_type, start_time, end_time, score, metadata)
        """
        # Extract audio from video
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            logger.error("Audio extraction failed, cannot analyze audio.")
            return []
        
        # Detect audio energy peaks
        energy_peaks = self.detect_audio_energy_peaks(audio_path)
        
        # Convert peaks to moment format
        results = []
        for start_time, end_time, energy_score in energy_peaks:
            moment_type = MomentType.AUDIO_PEAK
            metadata = {
                "energy_level": energy_score,
                "detection_method": "librosa" if self._has_librosa else "simplified"
            }
            results.append((moment_type, start_time, end_time, energy_score, metadata))
        
        # Note: Voice tone and laughter detection will be implemented in subsequent steps
        
        return results


class ContentAnalyzer:
    """
    Main class for multi-faceted content analysis.
    
    Coordinates various analysis components to detect interesting moments
    through audio, visual, and textual analysis.
    """
    
    def __init__(self, config: Optional[ContentAnalysisConfig] = None):
        """
        Initialize the content analyzer.
        
        Args:
            config: Configuration for content analysis
        """
        self.config = config or ContentAnalysisConfig()
        
        # Set up directory
        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize audio analyzer
        self.audio_analyzer = AudioAnalyzer(self.config)
        
        # Initialize voice analyzer
        voice_config = {
            "temp_dir": str(self.temp_dir / "voice_analysis"),
            "voice_emphasis_threshold": self.config.voice_emphasis_threshold,
            "enable_laughter_detection": self.config.enable_laughter_detection,
            "laughter_confidence_threshold": self.config.laughter_confidence_threshold
        }
        self.voice_analyzer = VoiceAnalyzer(voice_config)
        
        # Initialize transcript analyzer
        transcript_config = {
            "temp_dir": str(self.temp_dir / "transcript_analysis"),
            "sentiment_threshold": self.config.sentiment_threshold,
            "sentiment_window_size": self.config.sentiment_window_size,
            "enable_sentiment_analysis": self.config.enable_sentiment_analysis
        }
        self.transcript_analyzer = TranscriptAnalyzer(transcript_config)
        
        # Other analyzers will be implemented in subsequent steps
        self.visual_analyzer = None
        
        logger.info("Initialized ContentAnalyzer")
    
    def extract_transcript(self, video_path: str) -> Optional[str]:
        """
        Extract transcript from a video or generate one using speech recognition.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the transcript file, or None if extraction failed
        """
        # Check if a transcript file already exists
        base_path = os.path.splitext(video_path)[0]
        transcript_candidates = [
            f"{base_path}.vtt",
            f"{base_path}.srt",
            f"{base_path}.json",
            f"{base_path}.txt"
        ]
        
        for candidate in transcript_candidates:
            if os.path.exists(candidate):
                logger.info(f"Found existing transcript: {candidate}")
                return candidate
        
        # No transcript found - create a dummy one for now
        # In a real implementation, this would use speech recognition
        logger.warning("No transcript found. Creating a placeholder transcript.")
        
        transcript_path = self.temp_dir / f"transcript_{os.path.basename(video_path)}.json"
        
        # Get video duration
        try:
            cmd = [
                self.config.ffmpeg_path.replace("ffmpeg", "ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error getting video duration: {stderr}")
                return None
            
            duration = float(stdout.strip())
            
            # Create a simple dummy transcript with segments
            import json
            import random
            random.seed(42)  # For reproducible results
            
            # Create segments approximately every 5 seconds
            segment_duration = 5.0
            num_segments = int(duration / segment_duration)
            
            segments = []
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min(duration, (i + 1) * segment_duration)
                
                # Random placeholder text
                text = f"This is placeholder text for segment {i+1}."
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })
            
            # Save as JSON
            with open(transcript_path, 'w') as f:
                json.dump(segments, f, indent=2)
            
            logger.info(f"Created placeholder transcript: {transcript_path}")
            return str(transcript_path)
            
        except Exception as e:
            logger.error(f"Error creating placeholder transcript: {str(e)}")
            return None
    
    def analyze_content(
        self, 
        video_path: str,
        transcript_path: Optional[str] = None
    ) -> List[Tuple[MomentType, float, float, float, Dict[str, Any]]]:
        """
        Perform comprehensive content analysis on a video.
        
        Args:
            video_path: Path to the video file
            transcript_path: Optional path to a transcript file
            
        Returns:
            List of tuples (moment_type, start_time, end_time, score, metadata)
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
        
        logger.info(f"Starting content analysis for: {video_path}")
        
        # Extract audio first since multiple analyzers need it
        audio_path = self.audio_analyzer.extract_audio(video_path)
        if not audio_path:
            logger.error("Audio extraction failed, analysis will be limited.")
            audio_path = None
        
        # Get transcript if not provided
        if not transcript_path:
            transcript_path = self.extract_transcript(video_path)
        
        all_moments = []
        
        # Analyze audio aspects
        if audio_path:
            # Audio energy peaks
            audio_moments = self.audio_analyzer.analyze_audio(video_path)
            all_moments.extend(audio_moments)
            
            # Voice analysis (tone, emphasis, laughter)
            voice_results = self.voice_analyzer.analyze_voice(audio_path)
            
            for moment_type_str, start_time, end_time, score, metadata in voice_results:
                # Convert string moment type to enum
                if moment_type_str == "voice_emphasis":
                    moment_type = MomentType.VOICE_EMPHASIS
                elif moment_type_str == "laughter":
                    moment_type = MomentType.LAUGHTER
                else:
                    moment_type = MomentType.REACTION
                
                all_moments.append((moment_type, start_time, end_time, score, metadata))
        
        # Analyze transcript
        if transcript_path and os.path.exists(transcript_path):
            transcript_results = self.transcript_analyzer.analyze_transcript(transcript_path)
            
            for moment_type_str, start_time, end_time, score, metadata in transcript_results:
                # Convert string moment type to enum
                if moment_type_str == "sentiment_peak":
                    moment_type = MomentType.SENTIMENT_PEAK
                elif moment_type_str == "keyword":
                    moment_type = MomentType.KEYWORD
                else:
                    moment_type = MomentType.KEY_POINT  # Default for now
                
                all_moments.append((moment_type, start_time, end_time, score, metadata))
        
        # Note: Additional analyses will be added in subsequent steps
        
        logger.info(f"Content analysis complete. Detected {len(all_moments)} potential moments")
        return all_moments
    
    def get_moment_scores(
        self,
        video_path: str,
        transcript_path: Optional[str] = None
    ) -> List[Tuple[float, float, List[MomentScore]]]:
        """
        Convert content analysis results to moment scores.
        
        Args:
            video_path: Path to the video file
            transcript_path: Optional path to a transcript file
            
        Returns:
            List of tuples (start_time, end_time, list_of_scores)
        """
        # Perform content analysis
        moment_tuples = self.analyze_content(video_path, transcript_path)
        
        # Group by time ranges
        # This is a simplified approach - a more sophisticated implementation
        # would merge overlapping ranges and handle conflicts
        moment_groups = {}
        
        for moment_type, start_time, end_time, score, metadata in moment_tuples:
            # Create a key based on time range
            # Round to 2 decimal places for easier grouping
            key = (round(start_time, 2), round(end_time, 2))
            
            if key not in moment_groups:
                moment_groups[key] = []
            
            # Create a moment score
            moment_score = MomentScore(
                score=score,
                confidence=0.8,  # Default confidence, will be refined later
                moment_type=moment_type,
                metadata=metadata
            )
            
            moment_groups[key].append(moment_score)
        
        # Convert groups to result format
        results = []
        for (start_time, end_time), scores in moment_groups.items():
            results.append((start_time, end_time, scores))
        
        return results 