"""
Moment Analyzer Module

This module contains the main analyzer for detecting interesting moments in videos,
coordinating various detection strategies and algorithms.
"""

import os
import logging
import dataclasses
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MomentType(Enum):
    """Types of interesting moments that can be detected."""
    AUDIO_PEAK = "audio_peak"
    VOICE_EMPHASIS = "voice_emphasis"
    LAUGHTER = "laughter"
    REACTION = "reaction"
    SENTIMENT_PEAK = "sentiment_peak"
    KEYWORD = "keyword"
    GESTURE = "gesture"
    EXPRESSION = "expression"
    HIGH_ENGAGEMENT = "high_engagement"
    KEY_POINT = "key_point"
    TOPIC_BOUNDARY = "topic_boundary"
    CAUSE_EFFECT = "cause_effect"
    COMPLETE_THOUGHT = "complete_thought"


@dataclass
class MomentScore:
    """Score for a detected moment with confidence and metadata."""
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    moment_type: MomentType
    metadata: Dict[str, Any]


@dataclass
class DetectedMoment:
    """A detected interesting moment with timing and scores."""
    start_time: float  # seconds
    end_time: float  # seconds
    scores: List[MomentScore]
    combined_score: float  # 0.0 to 1.0
    transcript: Optional[str] = None
    preview_image_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time,
            "combined_score": self.combined_score,
            "scores": [
                {
                    "type": score.moment_type.value,
                    "score": score.score,
                    "confidence": score.confidence,
                    "metadata": score.metadata
                }
                for score in self.scores
            ],
            "transcript": self.transcript,
            "preview_image_path": self.preview_image_path
        }


@dataclass
class MomentAnalyzerConfig:
    """Configuration for the moment analyzer."""
    # General settings
    temp_dir: str = "temp"
    output_dir: str = "output"
    ffmpeg_path: str = "ffmpeg"
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Component enablement
    enable_audio_analysis: bool = True
    enable_transcript_analysis: bool = True
    enable_visual_analysis: bool = True
    enable_engagement_prediction: bool = True
    enable_narrative_analysis: bool = True
    
    # Thresholds and parameters
    min_moment_duration: float = 1.0  # seconds
    max_moment_duration: float = 15.0  # seconds
    min_detection_score: float = 0.6  # 0.0 to 1.0
    overlap_threshold: float = 0.5  # 0.0 to 1.0
    
    # Format preferences
    target_platform: str = "general"  # "general", "tiktok", "instagram", "youtube"
    content_category: str = "general"  # "general", "educational", "entertainment", etc.
    
    # Additional settings
    generate_previews: bool = True
    save_intermediate_results: bool = False
    verbose_logging: bool = False
    
    # Transcript analysis settings
    sentiment_threshold: float = 0.7  # 0.0 to 1.0
    sentiment_window_size: int = 50  # tokens
    keyword_importance_threshold: float = 0.6  # 0.0 to 1.0


class MomentAnalyzer:
    """
    Main analyzer for detecting interesting moments in videos.
    
    This class coordinates the different analysis components and combines
    their results to identify the most engaging moments in a video.
    """
    
    def __init__(self, config: Optional[MomentAnalyzerConfig] = None):
        """
        Initialize the moment analyzer.
        
        Args:
            config: Configuration for the analyzer
        """
        self.config = config or MomentAnalyzerConfig()
        
        # Set up directories
        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        if self.config.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Initialized MomentAnalyzer")
        logger.info(f"Configuration: {dataclasses.asdict(self.config)}")
    
    def _init_components(self):
        """Initialize the analysis components based on configuration."""
        # Import analyzers
        from app.clip_generation.services.moment_detection.content_analysis import ContentAnalyzer, ContentAnalysisConfig
        from app.clip_generation.services.moment_detection.transcript_analysis import TranscriptAnalyzer
        
        # Create content analysis config from main config
        content_config = ContentAnalysisConfig(
            temp_dir=str(self.temp_dir / "content_analysis"),
            ffmpeg_path=self.config.ffmpeg_path,
            device=self.config.device,
            # Pass transcript analysis settings
            sentiment_threshold=self.config.sentiment_threshold,
            sentiment_window_size=self.config.sentiment_window_size,
            enable_sentiment_analysis=self.config.enable_transcript_analysis
        )
        
        # Initialize content analyzer (which contains the transcript analyzer)
        self.content_analyzer = ContentAnalyzer(content_config)
        
        # Save a direct reference to the transcript analyzer for easier access
        self.transcript_analyzer = self.content_analyzer.transcript_analyzer
        
        # Other components will be initialized in subsequent steps
        self.engagement_predictor = None
        self.narrative_analyzer = None
        self.highlight_selector = None
        
        logger.debug("Analysis components initialized")
    
    def _calculate_combined_score(self, scores: List[MomentScore]) -> float:
        """
        Calculate a combined score from multiple moment scores.
        
        Args:
            scores: List of individual moment scores
            
        Returns:
            Combined score (0.0 to 1.0)
        """
        if not scores:
            return 0.0
        
        # Simple weighted average based on confidence
        total_weight = sum(score.confidence for score in scores)
        
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(score.score * score.confidence for score in scores)
        return weighted_sum / total_weight
    
    def _extract_frame(self, video_path: str, timestamp: float) -> Optional[str]:
        """
        Extract a frame from a video at a specific timestamp.
        
        Args:
            video_path: Path to the video file
            timestamp: Timestamp in seconds
            
        Returns:
            Path to the extracted frame image, or None if extraction failed
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        if not self.config.generate_previews:
            return None
        
        # Create output path
        frame_filename = f"frame_{os.path.basename(video_path)}_{timestamp:.2f}.jpg"
        frame_path = self.temp_dir / "frames" / frame_filename
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        cmd = [
            self.config.ffmpeg_path,
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            str(frame_path)
        ]
        
        try:
            # Run FFmpeg
            import subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error extracting frame: {stderr}")
                return None
            
            if not os.path.exists(frame_path):
                logger.error(f"Frame extraction failed, output file not created: {frame_path}")
                return None
            
            logger.debug(f"Frame extracted: {frame_path}")
            return str(frame_path)
            
        except Exception as e:
            logger.error(f"Error running FFmpeg: {str(e)}")
            return None
    
    def analyze_video(
        self, 
        video_path: str,
        transcript_path: Optional[str] = None
    ) -> List[DetectedMoment]:
        """
        Analyze a video to detect interesting moments.
        
        Args:
            video_path: Path to the video file
            transcript_path: Optional path to a transcript file
            
        Returns:
            List of detected interesting moments
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
        
        logger.info(f"Analyzing video for interesting moments: {video_path}")
        
        # Get moment scores from content analyzer
        moment_tuples = self.content_analyzer.get_moment_scores(video_path, transcript_path)
        
        # Convert to DetectedMoment objects
        detected_moments = []
        
        for start_time, end_time, scores in moment_tuples:
            # Calculate combined score
            combined_score = self._calculate_combined_score(scores)
            
            # Skip moments below threshold
            if combined_score < self.config.min_detection_score:
                continue
            
            # Extract preview frame (at the middle of the moment)
            preview_timestamp = (start_time + end_time) / 2
            preview_path = self._extract_frame(video_path, preview_timestamp) if self.config.generate_previews else None
            
            # Create DetectedMoment object
            moment = DetectedMoment(
                start_time=start_time,
                end_time=end_time,
                scores=scores,
                combined_score=combined_score,
                transcript=None,  # Will be added in subsequent steps
                preview_image_path=preview_path
            )
            
            detected_moments.append(moment)
        
        # Sort by combined score (descending)
        detected_moments.sort(key=lambda m: m.combined_score, reverse=True)
        
        logger.info(f"Detected {len(detected_moments)} interesting moments")
        return detected_moments
    
    def extract_highlights(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        max_highlights: int = 5,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract highlight clips based on detected interesting moments.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save highlight clips
            max_highlights: Maximum number of highlights to extract
            min_duration: Minimum duration for each highlight
            max_duration: Maximum duration for each highlight
            
        Returns:
            List of dictionaries with highlight information
        """
        # Use provided output directory or default
        output_path = Path(output_dir) if output_dir else self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use provided duration limits or config defaults
        min_dur = min_duration if min_duration is not None else self.config.min_moment_duration
        max_dur = max_duration if max_duration is not None else self.config.max_moment_duration
        
        logger.info(f"Extracting up to {max_highlights} highlights from {video_path}")
        
        # First, analyze the video to detect moments
        moments = self.analyze_video(video_path)
        
        if not moments:
            logger.warning(f"No interesting moments detected in {video_path}")
            return []
        
        # Filter moments based on duration constraints
        valid_moments = [
            m for m in moments 
            if min_dur <= (m.end_time - m.start_time) <= max_dur
        ]
        
        # Sort by combined score (descending)
        valid_moments.sort(key=lambda m: m.combined_score, reverse=True)
        
        # Take the top N moments
        top_moments = valid_moments[:max_highlights]
        
        # Extract clips for each moment
        results = []
        
        for i, moment in enumerate(top_moments):
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_file = output_path / f"{base_name}_highlight_{i+1}.mp4"
            
            # Extract clip
            clip_info = self._extract_clip(
                video_path, 
                str(output_file), 
                moment.start_time, 
                moment.end_time
            )
            
            if clip_info:
                # Add moment information to result
                clip_info.update(moment.to_dict())
                results.append(clip_info)
        
        logger.info(f"Extracted {len(results)} highlight clips")
        return results
    
    def _extract_clip(
        self, 
        video_path: str, 
        output_path: str, 
        start_time: float, 
        end_time: float
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a clip from a video.
        
        Args:
            video_path: Path to the video file
            output_path: Path for the output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Dictionary with clip information, or None if extraction failed
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Build FFmpeg command
        cmd = [
            self.config.ffmpeg_path,
            "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            output_path
        ]
        
        try:
            # Run FFmpeg
            import subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error extracting clip: {stderr}")
                return None
            
            if not os.path.exists(output_path):
                logger.error(f"Clip extraction failed, output file not created: {output_path}")
                return None
            
            logger.info(f"Clip extracted: {output_path}")
            
            # Return clip information
            return {
                "clip_path": output_path,
                "source_video": video_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            }
            
        except Exception as e:
            logger.error(f"Error running FFmpeg: {str(e)}")
            return None 