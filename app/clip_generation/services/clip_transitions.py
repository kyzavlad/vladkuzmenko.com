"""
Enhanced Clip Transitions Module

This module provides advanced content-aware transitions for clip assembly,
including seamless cuts, audio crossfades, and dynamic transition selection.
"""

import os
import logging
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TransitionConfig:
    """Configuration for clip transitions."""
    # Crossfade settings
    enable_video_crossfade: bool = True
    enable_audio_crossfade: bool = True
    video_crossfade_duration: float = 0.5  # seconds
    audio_crossfade_duration: float = 0.75  # seconds
    
    # J-cut/L-cut settings (audio leads/lags video)
    enable_j_l_cuts: bool = True
    j_cut_offset: float = 0.3  # seconds audio starts before video
    l_cut_offset: float = 0.5  # seconds audio ends after video
    
    # Scene detection settings
    enable_scene_detection: bool = True
    scene_detection_threshold: float = 0.3  # 0.0 to 1.0
    
    # Transition type selection
    prefer_silence_based_cuts: bool = True
    prefer_scene_based_cuts: bool = True
    
    # Fade in/out settings
    fade_in_duration: float = 0.5  # seconds
    fade_out_duration: float = 0.75  # seconds


class ClipTransitioner:
    """
    Handles advanced content-aware transitions between clip segments.
    
    This class provides functionality for:
    1. Detecting optimal transition points based on audio and visual content
    2. Applying various transition types (cuts, crossfades, J/L cuts)
    3. Creating seamless transitions between assembled clip segments
    """
    
    def __init__(self, config: Optional[TransitionConfig] = None, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize the clip transitioner.
        
        Args:
            config: Configuration for transitions
            ffmpeg_path: Path to ffmpeg executable
        """
        self.config = config or TransitionConfig()
        self.ffmpeg_path = ffmpeg_path
        
        # Create a temporary directory for transition processing
        self.temp_dir = Path("temp/transitions")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ClipTransitioner initialized")
    
    def detect_scene_changes(self, video_path: str, threshold: Optional[float] = None) -> List[float]:
        """
        Detect scene changes in a video that could serve as good transition points.
        
        Args:
            video_path: Path to the video file
            threshold: Scene change detection threshold (0.0 to 1.0)
            
        Returns:
            List of timestamps (in seconds) where scene changes occur
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Use provided threshold or default from config
        threshold = threshold or self.config.scene_detection_threshold
        
        # Create a temporary file for scene detection output
        temp_output = self.temp_dir / f"scene_changes_{os.path.basename(video_path)}.txt"
        
        try:
            # Run FFmpeg with scene detection filter
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-vf", f"select=gt(scene\\,{threshold}),showinfo",
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Parse the output to extract scene change timestamps
            scene_changes = []
            for line in result.stderr.split('\n'):
                if "pts_time:" in line:
                    try:
                        # Extract timestamp from the line
                        timestamp = float(line.split("pts_time:")[1].split()[0])
                        scene_changes.append(timestamp)
                    except (ValueError, IndexError):
                        pass
            
            logger.info(f"Detected {len(scene_changes)} scene changes in video")
            return scene_changes
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error detecting scene changes: {e.stderr if e.stderr else str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in scene detection: {str(e)}")
            return []
    
    def detect_silence_points(
        self, 
        video_path: str, 
        noise_threshold: float = -30.0,
        min_silence_duration: float = 0.3
    ) -> List[Dict[str, float]]:
        """
        Detect silent points in a video that could serve as good audio transition points.
        
        Args:
            video_path: Path to the video file
            noise_threshold: Threshold in dB below which audio is considered silence
            min_silence_duration: Minimum silence duration in seconds
            
        Returns:
            List of dictionaries with silence start and end times
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Run FFmpeg with silencedetect filter
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-af", f"silencedetect=noise={noise_threshold}dB:d={min_silence_duration}",
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Parse the output to extract silence periods
            silence_periods = []
            silence_start = None
            
            for line in result.stderr.split('\n'):
                if "silence_start" in line:
                    try:
                        silence_start = float(line.split("silence_start: ")[1])
                    except (ValueError, IndexError):
                        silence_start = None
                elif "silence_end" in line and silence_start is not None:
                    try:
                        silence_end = float(line.split("silence_end: ")[1].split()[0])
                        silence_periods.append({
                            "start": silence_start,
                            "end": silence_end,
                            "duration": silence_end - silence_start
                        })
                        silence_start = None
                    except (ValueError, IndexError):
                        pass
            
            logger.info(f"Detected {len(silence_periods)} silence periods in video")
            return silence_periods
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error detecting silence: {e.stderr if e.stderr else str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in silence detection: {str(e)}")
            return []
    
    def find_optimal_transition_point(
        self, 
        video_path: str, 
        target_time: float, 
        search_range: float = 2.0
    ) -> float:
        """
        Find the optimal transition point near the target time.
        
        This method combines scene detection and silence detection to find
        the best point for a transition.
        
        Args:
            video_path: Path to the video file
            target_time: Target time for the transition
            search_range: Range to search around the target time (seconds)
            
        Returns:
            Optimal transition time in seconds
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Define search window
        search_start = max(0, target_time - search_range)
        search_end = target_time + search_range
        
        # Get scene changes within search window
        scene_changes = []
        if self.config.prefer_scene_based_cuts:
            all_scene_changes = self.detect_scene_changes(video_path)
            scene_changes = [
                sc for sc in all_scene_changes 
                if search_start <= sc <= search_end
            ]
        
        # Get silence points within search window
        silence_points = []
        if self.config.prefer_silence_based_cuts:
            all_silence_periods = self.detect_silence_points(video_path)
            
            # Extract both start and end points of silence
            for period in all_silence_periods:
                if search_start <= period["start"] <= search_end:
                    silence_points.append(period["start"])
                if search_start <= period["end"] <= search_end:
                    silence_points.append(period["end"])
        
        # Combine all potential transition points
        all_points = scene_changes + silence_points
        
        if not all_points:
            logger.info(f"No optimal transition points found near {target_time}s, using target time")
            return target_time
        
        # Find point closest to target time
        optimal_point = min(all_points, key=lambda p: abs(p - target_time))
        logger.info(f"Found optimal transition point at {optimal_point}s (target: {target_time}s)")
        
        return optimal_point
    
    def create_transition_filter_complex(
        self, 
        segments: List[Dict[str, Any]], 
        output_path: str
    ) -> str:
        """
        Create an FFmpeg filter complex for advanced transitions between segments.
        
        Args:
            segments: List of segment dictionaries with paths and timestamps
            output_path: Path for the output clip
            
        Returns:
            FFmpeg filter complex string
        """
        if not segments:
            raise ValueError("No segments provided for transition")
        
        # Build the filter complex for transitions
        filter_parts = []
        
        # Label for input streams
        inputs = []
        for i in range(len(segments)):
            inputs.append(f"[{i}:v]")
            inputs.append(f"[{i}:a]")
        
        # Create video transitions
        video_parts = []
        for i in range(len(segments)):
            # Label for this segment's video
            video_parts.append(f"[{i}:v]")
        
        # Create crossfades if enabled
        if self.config.enable_video_crossfade and len(segments) > 1:
            crossfade_dur = self.config.video_crossfade_duration
            
            # Build crossfade filters
            for i in range(len(segments) - 1):
                # Get duration of current segment
                current_duration = segments[i]["end_time"] - segments[i]["start_time"]
                
                # Calculate xfade start time (adjusted for segment duration)
                xfade_start = max(0, current_duration - crossfade_dur)
                
                # Add crossfade filter
                filter_parts.append(
                    f"{video_parts[i]}{video_parts[i+1]}xfade=transition=fade:"
                    f"duration={crossfade_dur}:offset={xfade_start}[v{i}out]"
                )
                
                # Update label for next iteration
                video_parts[i+1] = f"[v{i}out]"
        
        # The last video part is our final video output
        final_video = video_parts[-1]
        
        # Create audio transitions
        audio_parts = []
        for i in range(len(segments)):
            # Label for this segment's audio
            audio_parts.append(f"[{i}:a]")
        
        # Create audio crossfades if enabled
        if self.config.enable_audio_crossfade and len(segments) > 1:
            crossfade_dur = self.config.audio_crossfade_duration
            
            # Build acrossfade filters
            for i in range(len(segments) - 1):
                # Get duration of current segment
                current_duration = segments[i]["end_time"] - segments[i]["start_time"]
                
                # Calculate crossfade start time (adjusted for segment duration)
                xfade_start = max(0, current_duration - crossfade_dur)
                
                # Add crossfade filter
                filter_parts.append(
                    f"{audio_parts[i]}{audio_parts[i+1]}acrossfade=d={crossfade_dur}[a{i}out]"
                )
                
                # Update label for next iteration
                audio_parts[i+1] = f"[a{i}out]"
        
        # The last audio part is our final audio output
        final_audio = audio_parts[-1]
        
        # Add fade in/out if enabled
        if self.config.fade_in_duration > 0:
            # Add fade in to first segment
            filter_parts.append(
                f"{final_video}fade=t=in:st=0:d={self.config.fade_in_duration}[vfaded]"
            )
            final_video = "[vfaded]"
            
            filter_parts.append(
                f"{final_audio}afade=t=in:st=0:d={self.config.fade_in_duration}[afaded]"
            )
            final_audio = "[afaded]"
        
        if self.config.fade_out_duration > 0:
            # Calculate total duration for fade out
            total_duration = sum(seg["end_time"] - seg["start_time"] for seg in segments)
            fade_start = total_duration - self.config.fade_out_duration
            
            # Add fade out to last segment
            filter_parts.append(
                f"{final_video}fade=t=out:st={fade_start}:d={self.config.fade_out_duration}[vfinal]"
            )
            final_video = "[vfinal]"
            
            filter_parts.append(
                f"{final_audio}afade=t=out:st={fade_start}:d={self.config.fade_out_duration}[afinal]"
            )
            final_audio = "[afinal]"
        
        # Complete filter complex
        filter_complex = ";".join(filter_parts)
        
        # Add map outputs
        filter_complex += f" -map {final_video} -map {final_audio}"
        
        return filter_complex
    
    def create_transition_command(
        self, 
        segments: List[Dict[str, Any]], 
        output_path: str,
        audio_normalize: bool = True,
        target_lufs: float = -14.0
    ) -> List[str]:
        """
        Create full FFmpeg command for assembling segments with transitions.
        
        Args:
            segments: List of segment dictionaries with paths and timestamps
            output_path: Path for the output clip
            audio_normalize: Whether to normalize audio
            target_lufs: Target loudness level (LUFS)
            
        Returns:
            FFmpeg command as list of strings
        """
        if not segments:
            raise ValueError("No segments provided for transition")
        
        # Start building command
        cmd = [self.ffmpeg_path, "-y"]
        
        # Add input files
        for segment in segments:
            cmd.extend(["-i", segment["path"]])
        
        # Create filter complex
        filter_complex = self.create_transition_filter_complex(segments, output_path)
        
        # Add filter complex to command
        cmd.extend(["-filter_complex", filter_complex])
        
        # Add audio normalization if requested
        if audio_normalize:
            cmd.extend([
                "-af", f"loudnorm=I={target_lufs}:LRA=11:TP=-1.5"
            ])
        
        # Add output encoding settings
        cmd.extend([
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path
        ])
        
        return cmd 