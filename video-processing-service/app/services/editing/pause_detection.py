import logging
import asyncio
import os
import json
import numpy as np
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PauseType(str, Enum):
    """Types of pauses that can be detected in videos."""
    NATURAL = "natural"  # Pauses that should be preserved (e.g., between sentences)
    UNNECESSARY = "unnecessary"  # Pauses that can be removed
    EXTENDED = "extended"  # Pauses that should be shortened but not removed
    SCENE_BOUNDARY = "scene_boundary"  # Pauses that coincide with scene changes


@dataclass
class PauseInfo:
    """Information about a detected pause."""
    start_time: float
    end_time: float
    duration: float
    type: PauseType
    confidence: float
    scene_change: bool = False
    context: Dict[str, Any] = None


class PauseDetectionService:
    """
    Service for detecting and analyzing pauses in video content.
    
    Features:
    - Adaptive silence threshold detection
    - Content-aware pause analysis
    - Scene-aware trimming
    - Configurable pause length thresholds
    """
    
    def __init__(
        self,
        min_pause_duration: float = 0.3,
        max_pause_duration: float = 2.0,
        silence_threshold_db: float = -35.0,
        adaptive_threshold: bool = True,
        content_aware: bool = True,
        scene_aware: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the pause detection service.
        
        Args:
            min_pause_duration: Minimum duration (seconds) to consider as a pause
            max_pause_duration: Maximum duration (seconds) to analyze as a pause
            silence_threshold_db: Initial silence threshold in dB (lower is quieter)
            adaptive_threshold: Whether to adaptively calibrate silence threshold
            content_aware: Whether to perform content-aware analysis
            scene_aware: Whether to consider scene boundaries
            config: Additional configuration parameters
        """
        self.min_pause_duration = min_pause_duration
        self.max_pause_duration = max_pause_duration
        self.silence_threshold_db = silence_threshold_db
        self.adaptive_threshold = adaptive_threshold
        self.content_aware = content_aware
        self.scene_aware = scene_aware
        
        self.config = config or {}
        
        # Configure audio analysis parameters
        self.audio_analyzer = AudioAnalyzer(
            silence_threshold_db=silence_threshold_db,
            min_silence_duration=min_pause_duration
        )
        
        # Configure linguistic analysis if content-aware
        self.linguistic_analyzer = None
        if self.content_aware:
            self.linguistic_analyzer = LinguisticPauseAnalyzer()
        
        self.logger = logging.getLogger(__name__)
    
    async def detect_pauses(
        self,
        video_path: str,
        transcript: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect pauses in a video file.
        
        Args:
            video_path: Path to the video file
            transcript: Optional transcript with timing information
            
        Returns:
            Dictionary containing detected pauses and analysis information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Detecting pauses in video: {video_path}")
        
        try:
            # Extract audio from video
            audio_path = await self._extract_audio(video_path)
            
            # Calibrate silence threshold if adaptive mode is enabled
            if self.adaptive_threshold:
                calibrated_threshold = await self.audio_analyzer.calibrate_threshold(audio_path)
                self.logger.info(f"Calibrated silence threshold: {calibrated_threshold} dB")
            
            # Detect silence segments in audio
            silence_segments = await self.audio_analyzer.detect_silence(audio_path)
            
            # Convert silence segments to pause information
            pause_infos = [
                PauseInfo(
                    start_time=segment["start"],
                    end_time=segment["end"],
                    duration=segment["end"] - segment["start"],
                    type=PauseType.UNNECESSARY,  # Default type, will refine later
                    confidence=segment["confidence"]
                )
                for segment in silence_segments
                if segment["end"] - segment["start"] <= self.max_pause_duration
            ]
            
            # Enhance pause information with content awareness if enabled and transcript available
            if self.content_aware and transcript:
                pause_infos = await self._enhance_with_content_awareness(
                    pause_infos, transcript
                )
            
            # Enhance with scene awareness if enabled
            if self.scene_aware:
                pause_infos = await self._enhance_with_scene_awareness(
                    pause_infos, video_path
                )
            
            # Filter pauses based on configured criteria
            filtered_pauses = self._filter_pauses(pause_infos)
            
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            self.logger.info(
                f"Detected {len(filtered_pauses)} actionable pauses "
                f"out of {len(pause_infos)} total pauses"
            )
            
            return {
                "video_path": video_path,
                "total_pauses": len(pause_infos),
                "actionable_pauses": len(filtered_pauses),
                "pauses": [
                    {
                        "start_time": pause.start_time,
                        "end_time": pause.end_time,
                        "duration": pause.duration,
                        "type": pause.type,
                        "confidence": pause.confidence,
                        "scene_change": pause.scene_change,
                        "context": pause.context
                    }
                    for pause in filtered_pauses
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting pauses: {str(e)}")
            raise
    
    async def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        # Create temporary file for audio
        audio_path = os.path.join(
            tempfile.gettempdir(),
            f"{os.path.basename(video_path)}.wav"
        )
        
        # Build FFmpeg command to extract audio
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",  # Disable video
            "-acodec", "pcm_s16le",  # PCM 16-bit little-endian audio
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            audio_path
        ]
        
        self.logger.debug(f"Extracting audio using command: {' '.join(cmd)}")
        
        # Run FFmpeg command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            self.logger.error(f"FFmpeg error: {error_msg}")
            raise Exception(f"Audio extraction failed: {error_msg}")
        
        self.logger.debug(f"Audio extraction complete: {audio_path}")
        return audio_path
    
    async def _enhance_with_content_awareness(
        self,
        pauses: List[PauseInfo],
        transcript: Dict[str, Any]
    ) -> List[PauseInfo]:
        """
        Enhance pause detection with linguistic analysis.
        
        Args:
            pauses: List of detected pauses
            transcript: Transcript with timing information
            
        Returns:
            Enhanced list of pauses with linguistic context
        """
        if not self.linguistic_analyzer:
            return pauses
            
        # Analyze transcript to find linguistic boundaries
        linguistic_boundaries = await self.linguistic_analyzer.analyze(transcript)
        
        enhanced_pauses = []
        
        for pause in pauses:
            # Find any linguistic boundaries that overlap with this pause
            matching_boundaries = [
                boundary for boundary in linguistic_boundaries
                if (
                    abs(boundary["time"] - pause.start_time) < 0.5 or
                    abs(boundary["time"] - pause.end_time) < 0.5 or
                    (boundary["time"] >= pause.start_time and 
                     boundary["time"] <= pause.end_time)
                )
            ]
            
            # If we have matching boundaries, classify the pause
            if matching_boundaries:
                best_match = max(
                    matching_boundaries,
                    key=lambda x: x["strength"]
                )
                
                # Determine pause type based on boundary type
                if best_match["type"] in ["sentence", "paragraph"]:
                    pause_type = PauseType.NATURAL
                elif best_match["type"] == "phrase":
                    # Short phrase boundaries can be unnecessary
                    if pause.duration > 0.8:  # Configurable threshold
                        pause_type = PauseType.EXTENDED
                    else:
                        pause_type = PauseType.NATURAL
                else:
                    pause_type = PauseType.UNNECESSARY
                
                # Update pause with linguistic context
                updated_pause = PauseInfo(
                    start_time=pause.start_time,
                    end_time=pause.end_time,
                    duration=pause.duration,
                    type=pause_type,
                    confidence=max(pause.confidence, best_match["strength"]),
                    scene_change=pause.scene_change,
                    context={
                        "linguistic_boundary": best_match["type"],
                        "strength": best_match["strength"],
                        "text_before": best_match.get("text_before", ""),
                        "text_after": best_match.get("text_after", "")
                    }
                )
                
                enhanced_pauses.append(updated_pause)
            else:
                # No linguistic context, keep as is
                enhanced_pauses.append(pause)
        
        return enhanced_pauses
    
    async def _enhance_with_scene_awareness(
        self,
        pauses: List[PauseInfo],
        video_path: str
    ) -> List[PauseInfo]:
        """
        Enhance pause detection with scene change detection.
        
        Args:
            pauses: List of detected pauses
            video_path: Path to the video file
            
        Returns:
            Enhanced list of pauses with scene change information
        """
        # Detect scene changes
        scene_changes = await self._detect_scene_changes(video_path)
        
        enhanced_pauses = []
        
        for pause in pauses:
            # Check if pause overlaps with any scene change
            scene_change = False
            for change_time in scene_changes:
                # Consider scene change to overlap if it's within or close to pause
                if (
                    change_time >= pause.start_time - 0.2 and
                    change_time <= pause.end_time + 0.2
                ):
                    scene_change = True
                    break
            
            # If scene change detected, mark the pause
            if scene_change:
                # Create a new pause with scene change flag set
                updated_pause = PauseInfo(
                    start_time=pause.start_time,
                    end_time=pause.end_time,
                    duration=pause.duration,
                    type=PauseType.SCENE_BOUNDARY if pause.type == PauseType.UNNECESSARY else pause.type,
                    confidence=pause.confidence,
                    scene_change=True,
                    context=pause.context
                )
                enhanced_pauses.append(updated_pause)
            else:
                enhanced_pauses.append(pause)
        
        return enhanced_pauses
    
    async def _detect_scene_changes(self, video_path: str) -> List[float]:
        """
        Detect scene changes in a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of timestamps where scene changes occur
        """
        # Create temporary file for scene change data
        temp_file = os.path.join(
            tempfile.gettempdir(),
            f"{os.path.basename(video_path)}.scenes.txt"
        )
        
        # Build FFmpeg command for scene detection
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-filter:v", "select='gt(scene,0.3)',showinfo",  # Detect scenes with 30% change
            "-f", "null",
            "-"
        ]
        
        self.logger.debug(f"Detecting scenes using command: {' '.join(cmd)}")
        
        # Run FFmpeg command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            self.logger.error(f"FFmpeg scene detection error: {stderr.decode()}")
            # Don't fail the whole process if scene detection fails
            return []
        
        # Parse scene change timestamps from FFmpeg output
        # Example line: "[Parsed_showinfo_1 @ 0x7f8f5c00] n:   0 pts:      0 pts_time:0 ..."
        scene_changes = []
        for line in stderr.decode().split('\n'):
            if "pts_time:" in line:
                try:
                    # Extract timestamp
                    pts_time_part = line.split("pts_time:")[1].split(" ")[0]
                    timestamp = float(pts_time_part)
                    scene_changes.append(timestamp)
                except (IndexError, ValueError) as e:
                    self.logger.warning(f"Failed to parse scene info: {line}, error: {e}")
        
        self.logger.debug(f"Detected {len(scene_changes)} scene changes")
        return scene_changes
    
    def _filter_pauses(self, pauses: List[PauseInfo]) -> List[PauseInfo]:
        """
        Filter pauses based on configured criteria.
        
        Args:
            pauses: List of detected pauses
            
        Returns:
            Filtered list of actionable pauses
        """
        # Set up filters based on configuration
        min_duration = self.config.get("filter_min_duration", self.min_pause_duration)
        max_duration = self.config.get("filter_max_duration", self.max_pause_duration)
        min_confidence = self.config.get("filter_min_confidence", 0.6)
        
        # Additional filter conditions from config
        preserve_natural = self.config.get("preserve_natural_pauses", True)
        preserve_scene_boundaries = self.config.get("preserve_scene_boundaries", True)
        
        filtered_pauses = []
        
        for pause in pauses:
            # Check basic duration criteria
            if pause.duration < min_duration or pause.duration > max_duration:
                continue
                
            # Check confidence
            if pause.confidence < min_confidence:
                continue
                
            # Apply content-aware filters
            if preserve_natural and pause.type == PauseType.NATURAL:
                continue
                
            # Apply scene-aware filters
            if preserve_scene_boundaries and pause.scene_change:
                continue
            
            # Pause passed all filters, include it
            filtered_pauses.append(pause)
            
        return filtered_pauses


class AudioAnalyzer:
    """
    Analyzes audio to detect silence and calibrate thresholds.
    """
    
    def __init__(
        self,
        silence_threshold_db: float = -35.0,
        min_silence_duration: float = 0.3
    ):
        """
        Initialize the audio analyzer.
        
        Args:
            silence_threshold_db: Silence threshold in dB (lower = quieter)
            min_silence_duration: Minimum duration for silence detection
        """
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.logger = logging.getLogger(__name__)
    
    async def calibrate_threshold(self, audio_path: str) -> float:
        """
        Adaptively calibrate silence threshold based on audio content.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Calibrated silence threshold in dB
        """
        self.logger.info(f"Calibrating silence threshold for {audio_path}")
        
        try:
            # Get audio stats
            audio_stats = await self._get_audio_stats(audio_path)
            
            # Calculate noise floor
            noise_floor = await self._calculate_noise_floor(audio_path)
            
            # Use mean and RMS to set a reasonable threshold above noise floor
            mean_volume = audio_stats.get("mean_volume", -30)
            
            # Set threshold between noise floor and mean volume
            # This makes it adaptive to the specific audio content
            calibrated_threshold = noise_floor + (mean_volume - noise_floor) * 0.3
            
            # Clamp to reasonable values
            calibrated_threshold = max(-50, min(-20, calibrated_threshold))
            
            self.silence_threshold_db = calibrated_threshold
            return calibrated_threshold
            
        except Exception as e:
            self.logger.error(f"Error calibrating threshold: {str(e)}")
            # Fall back to default if calibration fails
            return self.silence_threshold_db
    
    async def _get_audio_stats(self, audio_path: str) -> Dict[str, float]:
        """
        Get statistical information about the audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of audio statistics
        """
        # Use FFmpeg to get volume stats
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", "volumedetect",
            "-f", "null",
            "-"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            self.logger.error(f"FFmpeg error: {stderr.decode()}")
            raise Exception("Failed to analyze audio volume")
        
        # Parse volume statistics
        stats = {}
        stderr_text = stderr.decode()
        
        for line in stderr_text.split('\n'):
            if "mean_volume" in line:
                stats["mean_volume"] = float(line.split(':')[1].strip().split(' ')[0])
            elif "max_volume" in line:
                stats["max_volume"] = float(line.split(':')[1].strip().split(' ')[0])
        
        return stats
    
    async def _calculate_noise_floor(self, audio_path: str) -> float:
        """
        Calculate the noise floor of the audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Estimated noise floor in dB
        """
        # Get histogram of audio levels to identify noise floor
        # This is a simplified version that uses FFmpeg's silencedetect filter
        # with a very low threshold to identify the quietest parts
        
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", "silencedetect=noise=-60dB:d=0.1",
            "-f", "null",
            "-"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Parse silence detection output
        silence_levels = []
        stderr_text = stderr.decode()
        
        for line in stderr_text.split('\n'):
            if "silence_start" in line:
                # Next line should have noise level
                silence_levels.append(-60)  # Default value
        
        # If we found some silence, use as noise floor,
        # otherwise use a conservative default
        if silence_levels:
            # Use slightly higher than the detected floor to account for variations
            return -60 + 5
        else:
            return -50  # Conservative default
    
    async def detect_silence(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Detect silence segments in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of silence segments with start/end times
        """
        self.logger.info(
            f"Detecting silence in {audio_path} "
            f"(threshold: {self.silence_threshold_db}dB, "
            f"min duration: {self.min_silence_duration}s)"
        )
        
        # Use FFmpeg's silencedetect filter
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", f"silencedetect=noise={self.silence_threshold_db}dB:d={self.min_silence_duration}",
            "-f", "null",
            "-"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Parse silence detection output
        silence_segments = []
        start_time = None
        stderr_text = stderr.decode()
        
        for line in stderr_text.split('\n'):
            if "silence_start" in line:
                try:
                    start_time = float(line.split('silence_start: ')[1].split()[0])
                except (IndexError, ValueError):
                    self.logger.warning(f"Failed to parse silence start: {line}")
            
            elif "silence_end" in line and start_time is not None:
                try:
                    parts = line.split('silence_end: ')[1].split('|')
                    end_time = float(parts[0].strip())
                    
                    # Some versions of FFmpeg include silence duration
                    duration = None
                    for part in parts:
                        if "silence_duration" in part:
                            duration = float(part.split(':')[1].strip())
                    
                    if duration is None:
                        duration = end_time - start_time
                    
                    # Calculate confidence based on duration and how far below threshold
                    # Simple linear mapping: longer = more confident
                    confidence = min(1.0, duration / 1.0)
                    
                    silence_segments.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": duration,
                        "confidence": confidence
                    })
                    
                    start_time = None
                    
                except (IndexError, ValueError):
                    self.logger.warning(f"Failed to parse silence end: {line}")
        
        self.logger.info(f"Detected {len(silence_segments)} silence segments")
        return silence_segments


class LinguisticPauseAnalyzer:
    """
    Analyzes transcripts to identify linguistic boundaries for pause analysis.
    """
    
    def __init__(self):
        """Initialize the linguistic analyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Sentence-ending punctuation for boundary detection
        self.sentence_endings = ['.', '!', '?']
        
        # Phrase-level separation for boundary detection
        self.phrase_separators = [',', ';', ':']
        
        # Paragraph markers
        self.paragraph_markers = ['\n\n', '\r\n\r\n']
    
    async def analyze(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze transcript to identify linguistic boundaries.
        
        Args:
            transcript: Transcript with timing information
            
        Returns:
            List of linguistic boundaries
        """
        if not transcript or "segments" not in transcript:
            return []
            
        boundaries = []
        
        # Analyze each segment
        for i, segment in enumerate(transcript["segments"]):
            if "text" not in segment:
                continue
                
            text = segment.get("text", "").strip()
            
            # Find sentence boundaries
            for j, char in enumerate(text):
                if char in self.sentence_endings:
                    # This is a sentence boundary
                    # Estimate the time of this boundary based on character position
                    if j == 0:
                        continue  # Skip if it's the first character
                        
                    # Estimate boundary time based on position in segment
                    segment_duration = segment["end"] - segment["start"]
                    char_position_ratio = j / len(text)
                    boundary_time = segment["start"] + (segment_duration * char_position_ratio)
                    
                    # Get context
                    text_before = text[:j+1].strip()
                    text_after = text[j+1:].strip()
                    
                    boundaries.append({
                        "time": boundary_time,
                        "type": "sentence",
                        "strength": 0.9,  # High confidence for sentence boundaries
                        "text_before": text_before,
                        "text_after": text_after
                    })
            
            # Find phrase boundaries
            for j, char in enumerate(text):
                if char in self.phrase_separators:
                    # This is a phrase boundary
                    # Estimate the time of this boundary based on character position
                    if j == 0:
                        continue  # Skip if it's the first character
                        
                    # Estimate boundary time
                    segment_duration = segment["end"] - segment["start"]
                    char_position_ratio = j / len(text)
                    boundary_time = segment["start"] + (segment_duration * char_position_ratio)
                    
                    # Get context
                    text_before = text[:j+1].strip()
                    text_after = text[j+1:].strip()
                    
                    boundaries.append({
                        "time": boundary_time,
                        "type": "phrase",
                        "strength": 0.6,  # Medium confidence for phrase boundaries
                        "text_before": text_before,
                        "text_after": text_after
                    })
            
            # Check for paragraph boundaries between segments
            if i > 0 and i < len(transcript["segments"]) - 1:
                prev_segment = transcript["segments"][i-1]
                next_segment = transcript["segments"][i+1]
                
                # Check time gap between segments
                gap_before = segment["start"] - prev_segment["end"]
                gap_after = next_segment["start"] - segment["end"]
                
                # If significant gap before or after, this might be a paragraph boundary
                if gap_before > 1.0:  # More than 1 second gap
                    boundaries.append({
                        "time": (prev_segment["end"] + segment["start"]) / 2,
                        "type": "paragraph",
                        "strength": min(1.0, gap_before / 2.0),  # Longer gap = stronger boundary
                        "text_before": prev_segment.get("text", ""),
                        "text_after": segment.get("text", "")
                    })
                
                if gap_after > 1.0:  # More than 1 second gap
                    boundaries.append({
                        "time": (segment["end"] + next_segment["start"]) / 2,
                        "type": "paragraph",
                        "strength": min(1.0, gap_after / 2.0),  # Longer gap = stronger boundary
                        "text_before": segment.get("text", ""),
                        "text_after": next_segment.get("text", "")
                    })
        
        return boundaries 