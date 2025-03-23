"""
Silence Processing Utility Module

This module provides utility functions to process and remove 
unnecessary audio segments from videos, integrating with the 
clip generation pipeline.
"""

import os
import numpy as np
import logging
import time
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import concurrent.futures

from app.clip_generation.services.audio_analysis.audio_analyzer import AudioSegment
from app.clip_generation.services.audio_analysis.silence_detector import SilenceDetector, SilenceDetectorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SilenceProcessorConfig:
    """Configuration for silence processing."""
    
    def __init__(
        self,
        # Basic settings
        output_dir: Optional[str] = None,  # Output directory for processed files
        temp_dir: Optional[str] = None,  # Temporary directory
        ffmpeg_path: str = "ffmpeg",  # Path to ffmpeg binary
        
        # Processing settings
        removal_threshold: float = 0.5,  # Duration threshold for removal (seconds)
        max_segment_gap: float = 0.1,  # Maximum gap between segments to merge (seconds)
        crossfade_duration: float = 0.05,  # Crossfade duration (seconds)
        speed_up_silence: bool = False,  # Speed up instead of removing silence
        speed_factor: float = 2.0,  # Speed factor for silence (when not removing)
        
        # Performance settings
        parallel_processing: bool = True,  # Enable parallel processing
        max_workers: int = 4,  # Maximum number of worker threads
        
        # Audio quality settings
        audio_codec: str = "aac",  # Audio codec for output
        audio_bitrate: str = "192k",  # Audio bitrate for output
        
        # Advanced settings
        preserve_video_quality: bool = True,  # Preserve original video quality
        generate_report: bool = True,  # Generate processing report
        silence_detector_config: Optional[SilenceDetectorConfig] = None  # Silence detector config
    ):
        """
        Initialize silence processor configuration.
        
        Args:
            output_dir: Output directory for processed files
            temp_dir: Temporary directory
            ffmpeg_path: Path to ffmpeg binary
            removal_threshold: Duration threshold for removal (seconds)
            max_segment_gap: Maximum gap between segments to merge (seconds)
            crossfade_duration: Crossfade duration (seconds)
            speed_up_silence: Speed up instead of removing silence
            speed_factor: Speed factor for silence (when not removing)
            parallel_processing: Enable parallel processing
            max_workers: Maximum number of worker threads
            audio_codec: Audio codec for output
            audio_bitrate: Audio bitrate for output
            preserve_video_quality: Preserve original video quality
            generate_report: Generate processing report
            silence_detector_config: Silence detector configuration
        """
        # Set up directories
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "clip_generation" / "silence_processing"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.ffmpeg_path = ffmpeg_path
        
        # Processing settings
        self.removal_threshold = removal_threshold
        self.max_segment_gap = max_segment_gap
        self.crossfade_duration = crossfade_duration
        self.speed_up_silence = speed_up_silence
        self.speed_factor = speed_factor
        
        # Performance settings
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        
        # Audio quality settings
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        
        # Advanced settings
        self.preserve_video_quality = preserve_video_quality
        self.generate_report = generate_report
        
        # Set up silence detector config
        self.silence_detector_config = silence_detector_config or SilenceDetectorConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            "ffmpeg_path": self.ffmpeg_path,
            "removal_threshold": self.removal_threshold,
            "max_segment_gap": self.max_segment_gap,
            "crossfade_duration": self.crossfade_duration,
            "speed_up_silence": self.speed_up_silence,
            "speed_factor": self.speed_factor,
            "parallel_processing": self.parallel_processing,
            "max_workers": self.max_workers,
            "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate,
            "preserve_video_quality": self.preserve_video_quality,
            "generate_report": self.generate_report,
            "silence_detector_config": self.silence_detector_config.to_dict()
        }


class SilenceProcessor:
    """
    Processor for removing or modifying unnecessary audio segments.
    
    This class provides utilities to:
    - Process videos to remove silence and unnecessary sounds
    - Apply different processing strategies (removal, speedup)
    - Generate detailed reports on processing
    - Integrate with the clip generation pipeline
    """
    
    def __init__(self, config: Optional[SilenceProcessorConfig] = None):
        """
        Initialize the silence processor.
        
        Args:
            config: Configuration for silence processing
        """
        self.config = config or SilenceProcessorConfig()
        
        # Initialize silence detector
        self.silence_detector = SilenceDetector(self.config.silence_detector_config)
        
        # Statistics and reporting
        self.processed_files = 0
        self.total_duration = 0.0
        self.removed_duration = 0.0
        
        logger.info(f"Initialized SilenceProcessor with config: {json.dumps(self.config.to_dict(), indent=2)}")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Process a video to remove unnecessary audio segments.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            
        Returns:
            Path to processed video, or None if failed
        """
        start_time = time.time()
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            video_filename = os.path.basename(video_path)
            video_name, ext = os.path.splitext(video_filename)
            output_path = str(self.config.output_dir / f"{video_name}_processed{ext}")
        
        logger.info(f"Processing video: {video_path} -> {output_path}")
        
        # Step 1: Analyze video to find removable segments
        segments, removable_segments = self.silence_detector.process_file(video_path)
        
        if not segments:
            logger.error(f"Failed to analyze video: {video_path}")
            return None
        
        if not removable_segments:
            logger.info(f"No removable segments found in: {video_path}")
            # Copy original file to output path
            try:
                import shutil
                shutil.copy2(video_path, output_path)
                logger.info(f"Copied original file to: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Error copying file: {str(e)}")
                return None
        
        # Step 2: Process the video based on the detected segments
        if self.config.speed_up_silence:
            result_path = self._apply_speedup(video_path, output_path, segments, removable_segments)
        else:
            result_path = self._apply_removal(video_path, output_path, segments, removable_segments)
        
        # Step 3: Generate report if enabled
        if self.config.generate_report and result_path:
            report = self.silence_detector.generate_report()
            report_path = os.path.splitext(output_path)[0] + "_report.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Processing report saved to: {report_path}")
        
        # Update statistics
        if result_path:
            self.processed_files += 1
            
            # Get video duration
            try:
                original_duration = self._get_video_duration(video_path)
                processed_duration = self._get_video_duration(result_path)
                
                self.total_duration += original_duration
                self.removed_duration += (original_duration - processed_duration)
                
                logger.info(f"Reduced duration: {original_duration:.2f}s -> {processed_duration:.2f}s " +
                           f"({(original_duration - processed_duration):.2f}s removed, " +
                           f"{((original_duration - processed_duration) / original_duration * 100):.1f}%)")
            except Exception as e:
                logger.error(f"Error calculating durations: {str(e)}")
        
        logger.info(f"Processing completed in {time.time() - start_time:.2f}s")
        return result_path
    
    def _get_video_duration(self, video_path: str) -> float:
        """
        Get the duration of a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                self.config.ffmpeg_path, 
                "-i", video_path, 
                "-v", "quiet", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            else:
                logger.warning(f"Could not get duration using ffprobe, using fallback method")
                
                # Fallback method using ffmpeg directly
                cmd = [
                    self.config.ffmpeg_path,
                    "-i", video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
                
                # Parse duration from ffmpeg output
                output = result.stdout or ""
                duration_str = ""
                
                for line in output.split('\n'):
                    if "Duration:" in line:
                        duration_part = line.split("Duration:")[1].split(",")[0].strip()
                        h, m, s = duration_part.split(":")
                        duration_str = f"{float(h) * 3600 + float(m) * 60 + float(s)}"
                        break
                
                if duration_str:
                    return float(duration_str)
                else:
                    logger.error(f"Could not determine duration of: {video_path}")
                    return 0.0
                
        except Exception as e:
            logger.error(f"Error getting video duration: {str(e)}")
            return 0.0
    
    def _optimize_segments(self, removable_segments: List[AudioSegment]) -> List[AudioSegment]:
        """
        Optimize removable segments by merging close segments.
        
        Args:
            removable_segments: List of removable segments
            
        Returns:
            Optimized list of segments
        """
        if not removable_segments:
            return []
        
        # Only consider segments above threshold
        segments = [s for s in removable_segments if s.duration >= self.config.removal_threshold]
        
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda s: s.start_time)
        
        # Merge segments that are close to each other
        optimized = [segments[0]]
        
        for segment in segments[1:]:
            last = optimized[-1]
            
            # If this segment starts soon after the last one ends
            if segment.start_time - last.end_time <= self.config.max_segment_gap:
                # Extend the last segment
                last.end_time = segment.end_time
            else:
                # Add as a new segment
                optimized.append(segment)
        
        return optimized
    
    def _apply_removal(self, input_path: str, output_path: str, 
                      all_segments: List[AudioSegment], 
                      removable_segments: List[AudioSegment]) -> Optional[str]:
        """
        Apply segment removal to create a processed video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            all_segments: All detected segments
            removable_segments: Segments to remove
            
        Returns:
            Path to processed video, or None if failed
        """
        try:
            # Optimize segments for removal
            optimized_segments = self._optimize_segments(removable_segments)
            
            if not optimized_segments:
                logger.info(f"No segments to remove after optimization")
                import shutil
                shutil.copy2(input_path, output_path)
                return output_path
            
            # Calculate keep segments (inverse of removable)
            keep_segments = self._calculate_keep_segments(input_path, optimized_segments)
            
            if not keep_segments:
                logger.warning(f"No segments to keep, copying original file")
                import shutil
                shutil.copy2(input_path, output_path)
                return output_path
            
            # Create filter complex for ffmpeg
            filter_complex = self._create_segment_filter(input_path, keep_segments)
            
            # Apply ffmpeg command
            cmd = [
                self.config.ffmpeg_path,
                "-i", input_path,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "copy" if self.config.preserve_video_quality else "libx264",
                "-c:a", self.config.audio_codec,
                "-b:a", self.config.audio_bitrate,
                "-y",  # Overwrite without asking
                output_path
            ]
            
            logger.info(f"Executing ffmpeg command with {len(keep_segments)} segments")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error applying removal: {result.stderr}")
                return None
            
            logger.info(f"Successfully created processed video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying removal: {str(e)}")
            return None
    
    def _calculate_keep_segments(self, video_path: str, 
                               removable_segments: List[AudioSegment]) -> List[Dict[str, float]]:
        """
        Calculate segments to keep based on removable segments.
        
        Args:
            video_path: Path to video file
            removable_segments: Segments to remove
            
        Returns:
            List of segments to keep as dictionaries
        """
        # Get video duration
        duration = self._get_video_duration(video_path)
        
        if duration <= 0:
            logger.error(f"Invalid video duration: {duration}")
            return []
        
        # Sort removable segments
        segments = sorted(removable_segments, key=lambda s: s.start_time)
        
        # Calculate keep segments (inverse of removable)
        keep_segments = []
        
        # Add initial segment if needed
        if segments and segments[0].start_time > 0:
            keep_segments.append({
                "start": 0,
                "end": segments[0].start_time
            })
        
        # Add segments between removable segments
        for i in range(len(segments) - 1):
            current_end = segments[i].end_time
            next_start = segments[i + 1].start_time
            
            if next_start > current_end:
                keep_segments.append({
                    "start": current_end,
                    "end": next_start
                })
        
        # Add final segment if needed
        if segments and segments[-1].end_time < duration:
            keep_segments.append({
                "start": segments[-1].end_time,
                "end": duration
            })
        
        # Handle case where no removable segments
        if not segments:
            keep_segments.append({
                "start": 0,
                "end": duration
            })
        
        return keep_segments
    
    def _create_segment_filter(self, video_path: str, keep_segments: List[Dict[str, float]]) -> str:
        """
        Create ffmpeg filter complex for segment processing.
        
        Args:
            video_path: Path to input video
            keep_segments: Segments to keep
            
        Returns:
            FFmpeg filter complex string
        """
        # Video filters
        v_filters = []
        a_filters = []
        
        # Process each segment
        for i, segment in enumerate(keep_segments):
            start = segment["start"]
            end = segment["end"]
            duration = end - start
            
            # Add segment selection
            v_filters.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
            a_filters.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
        
        # Concatenate filters
        v_concat = "".join([f"[v{i}]" for i in range(len(keep_segments))])
        a_concat = "".join([f"[a{i}]" for i in range(len(keep_segments))])
        
        # Add concat filter
        v_filters.append(f"{v_concat}concat=n={len(keep_segments)}:v=1:a=0[v]")
        a_filters.append(f"{a_concat}concat=n={len(keep_segments)}:v=0:a=1[a]")
        
        # Combine filters
        filter_complex = ";".join(v_filters + a_filters)
        
        return filter_complex
    
    def _apply_speedup(self, input_path: str, output_path: str, 
                     all_segments: List[AudioSegment], 
                     removable_segments: List[AudioSegment]) -> Optional[str]:
        """
        Apply speed adjustment to silence segments.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            all_segments: All detected segments
            removable_segments: Segments to speed up
            
        Returns:
            Path to processed video, or None if failed
        """
        try:
            # Optimize segments for speedup
            optimized_segments = self._optimize_segments(removable_segments)
            
            if not optimized_segments:
                logger.info(f"No segments to speed up after optimization")
                import shutil
                shutil.copy2(input_path, output_path)
                return output_path
            
            # Get video duration
            duration = self._get_video_duration(input_path)
            
            # Create segments list with speed factors
            segments = []
            current_pos = 0.0
            
            # Sort removable segments
            optimized_segments.sort(key=lambda s: s.start_time)
            
            for segment in optimized_segments:
                # Add normal speed segment before this one
                if segment.start_time > current_pos:
                    segments.append({
                        "start": current_pos,
                        "end": segment.start_time,
                        "speed": 1.0
                    })
                
                # Add speed up segment
                segments.append({
                    "start": segment.start_time,
                    "end": segment.end_time,
                    "speed": self.config.speed_factor
                })
                
                current_pos = segment.end_time
            
            # Add final normal speed segment if needed
            if current_pos < duration:
                segments.append({
                    "start": current_pos,
                    "end": duration,
                    "speed": 1.0
                })
            
            # Create filter complex for ffmpeg
            filter_complex = self._create_speedup_filter(input_path, segments)
            
            # Apply ffmpeg command
            cmd = [
                self.config.ffmpeg_path,
                "-i", input_path,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "copy" if self.config.preserve_video_quality else "libx264",
                "-c:a", self.config.audio_codec,
                "-b:a", self.config.audio_bitrate,
                "-y",  # Overwrite without asking
                output_path
            ]
            
            logger.info(f"Executing ffmpeg speedup command with {len(segments)} segments")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error applying speedup: {result.stderr}")
                return None
            
            logger.info(f"Successfully created processed video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying speedup: {str(e)}")
            return None
    
    def _create_speedup_filter(self, video_path: str, segments: List[Dict[str, float]]) -> str:
        """
        Create ffmpeg filter complex for speedup processing.
        
        Args:
            video_path: Path to input video
            segments: Segments with speed factors
            
        Returns:
            FFmpeg filter complex string
        """
        # Video and audio filters
        v_filters = []
        a_filters = []
        
        # Process each segment
        for i, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]
            speed = segment["speed"]
            
            # Add segment selection and speed adjustment
            if speed == 1.0:
                # Normal speed
                v_filters.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
                a_filters.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
            else:
                # Speed up
                v_filters.append(
                    f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,setpts={1/speed}*PTS[v{i}]")
                a_filters.append(
                    f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,atempo={speed}[a{i}]")
        
        # Concatenate filters
        v_concat = "".join([f"[v{i}]" for i in range(len(segments))])
        a_concat = "".join([f"[a{i}]" for i in range(len(segments))])
        
        # Add concat filter
        v_filters.append(f"{v_concat}concat=n={len(segments)}:v=1:a=0[v]")
        a_filters.append(f"{a_concat}concat=n={len(segments)}:v=0:a=1[a]")
        
        # Combine filters
        filter_complex = ";".join(v_filters + a_filters)
        
        return filter_complex
    
    def batch_process(self, video_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple videos in batch.
        
        Args:
            video_paths: List of video paths to process
            output_dir: Output directory (overrides config)
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        # Set output directory
        if output_dir:
            self.config.output_dir = Path(output_dir)
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset statistics
        self.processed_files = 0
        self.total_duration = 0.0
        self.removed_duration = 0.0
        
        results = []
        
        if self.config.parallel_processing and len(video_paths) > 1:
            # Process videos in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Create tasks
                futures = {executor.submit(self.process_video, path): path for path in video_paths}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    video_path = futures[future]
                    try:
                        output_path = future.result()
                        results.append({
                            "input": video_path,
                            "output": output_path,
                            "status": "success" if output_path else "error"
                        })
                    except Exception as e:
                        logger.error(f"Error processing {video_path}: {str(e)}")
                        results.append({
                            "input": video_path,
                            "output": None,
                            "status": "error",
                            "error": str(e)
                        })
        else:
            # Process videos sequentially
            for video_path in video_paths:
                try:
                    output_path = self.process_video(video_path)
                    results.append({
                        "input": video_path,
                        "output": output_path,
                        "status": "success" if output_path else "error"
                    })
                except Exception as e:
                    logger.error(f"Error processing {video_path}: {str(e)}")
                    results.append({
                        "input": video_path,
                        "output": None,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Generate batch report
        report = {
            "summary": {
                "total_files": len(video_paths),
                "processed_files": self.processed_files,
                "success_rate": (self.processed_files / len(video_paths) * 100) if video_paths else 0,
                "total_duration": self.total_duration,
                "removed_duration": self.removed_duration,
                "reduction_percentage": (self.removed_duration / self.total_duration * 100) if self.total_duration > 0 else 0,
                "processing_time": time.time() - start_time
            },
            "results": results
        }
        
        # Save report if enabled
        if self.config.generate_report:
            timestamp = int(time.time())
            report_path = self.config.output_dir / f"batch_report_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Batch processing report saved to: {report_path}")
        
        logger.info(f"Batch processing completed: {self.processed_files}/{len(video_paths)} files " +
                   f"in {time.time() - start_time:.2f}s")
        
        return report
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "processed_files": self.processed_files,
            "total_duration": self.total_duration,
            "removed_duration": self.removed_duration,
            "reduction_percentage": (self.removed_duration / self.total_duration * 100) if self.total_duration > 0 else 0
        } 