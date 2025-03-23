"""
Dynamic Volume Adjustment Module

This module provides functionality for automatically adjusting the volume of music
during speech segments in video content.
"""

import os
import logging
import json
import tempfile
import subprocess
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class VolumeAdjuster:
    """
    Adjusts music volume dynamically to ensure speech intelligibility while maintaining
    appropriate background music levels.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the volume adjuster with configuration options."""
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
        
        # Default volume settings
        self.speech_music_ratio = self.config.get('speech_music_ratio', 0.3)  # Music at 30% of speech volume
        self.default_music_volume = self.config.get('default_music_volume', 0.7)  # 70% of original
        self.ducking_amount = self.config.get('ducking_amount', 0.3)  # Reduce to 30% during speech
        self.fade_in_time = self.config.get('fade_in_time', 0.5)  # Seconds
        self.fade_out_time = self.config.get('fade_out_time', 0.8)  # Seconds
        
        # Speech detection thresholds
        self.speech_detection_threshold = self.config.get('speech_detection_threshold', -25)  # dB
        self.speech_min_duration = self.config.get('speech_min_duration', 0.3)  # Seconds
        self.speech_merge_threshold = self.config.get('speech_merge_threshold', 0.5)  # Seconds
    
    def adjust_music_for_speech(self, 
                               video_path: str, 
                               music_path: str, 
                               output_path: str,
                               speech_segments: Optional[List[Dict[str, Any]]] = None,
                               music_start_time: float = 0.0,
                               music_end_time: Optional[float] = None,
                               keep_original_audio: bool = True) -> Dict[str, Any]:
        """
        Adjust music volume dynamically based on speech presence in the video.
        
        Args:
            video_path: Path to the video file
            music_path: Path to the music file
            output_path: Path for the output video
            speech_segments: Pre-detected speech segments (optional)
            music_start_time: Time to start music in the video (seconds)
            music_end_time: Time to end music (None for end of video)
            keep_original_audio: Whether to keep original audio from video
            
        Returns:
            Dictionary with operation results
        """
        # Check if files exist
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "error": f"Video file not found: {video_path}"
            }
        
        if not os.path.exists(music_path):
            return {
                "status": "error",
                "error": f"Music file not found: {music_path}"
            }
        
        try:
            # Get video info
            video_info = self._get_media_info(video_path)
            video_duration = float(video_info.get('duration', 0))
            
            if video_duration <= 0:
                return {
                    "status": "error",
                    "error": "Unable to determine video duration"
                }
            
            # Get music info
            music_info = self._get_media_info(music_path)
            music_duration = float(music_info.get('duration', 0))
            
            if music_duration <= 0:
                return {
                    "status": "error",
                    "error": "Unable to determine music duration"
                }
            
            # If no speech segments provided, detect speech
            if not speech_segments:
                logger.info("No speech segments provided, detecting speech...")
                speech_segments = self._detect_speech_segments(video_path)
            
            # Set default music end time if not specified
            if music_end_time is None:
                music_end_time = video_duration
            
            # Ensure music doesn't go beyond video duration
            music_end_time = min(music_end_time, video_duration)
            
            # Calculate volume automation
            volume_automation = self._generate_volume_automation(
                speech_segments=speech_segments,
                video_duration=video_duration,
                music_start_time=music_start_time,
                music_end_time=music_end_time
            )
            
            # Apply volume automation to music and mix with video
            result = self._apply_volume_automation(
                video_path=video_path,
                music_path=music_path,
                output_path=output_path,
                volume_automation=volume_automation,
                music_start_time=music_start_time,
                music_end_time=music_end_time,
                keep_original_audio=keep_original_audio
            )
            
            if result.get("status") == "success":
                return {
                    "status": "success",
                    "output_path": output_path,
                    "speech_segments": speech_segments,
                    "volume_points": volume_automation["volume_points"],
                    "music_duration": music_duration,
                    "video_duration": video_duration
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error adjusting music volume: {str(e)}")
            return {
                "status": "error",
                "error": f"Error adjusting music volume: {str(e)}"
            }
    
    def create_crossfade_mix(self,
                           music_files: List[str],
                           output_path: str,
                           crossfade_duration: float = 3.0,
                           target_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a continuous music mix with crossfades between tracks.
        
        Args:
            music_files: List of paths to music files
            output_path: Path for the output audio file
            crossfade_duration: Duration of crossfades in seconds
            target_duration: Target duration for the final mix (optional)
            
        Returns:
            Dictionary with operation results
        """
        if not music_files:
            return {
                "status": "error",
                "error": "No music files provided"
            }
        
        # Check if files exist
        missing_files = []
        for file_path in music_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                
        if missing_files:
            return {
                "status": "error",
                "error": f"Missing music files: {', '.join(missing_files)}"
            }
        
        try:
            # Get info for all music files
            music_info = []
            total_duration = 0
            
            for file_path in music_files:
                info = self._get_media_info(file_path)
                duration = float(info.get('duration', 0))
                
                if duration <= 0:
                    return {
                        "status": "error",
                        "error": f"Unable to determine duration for: {file_path}"
                    }
                
                music_info.append({
                    "path": file_path,
                    "duration": duration
                })
                
                # Add duration minus crossfade (except for last track)
                if total_duration > 0:
                    total_duration -= crossfade_duration
                total_duration += duration
            
            # If target duration specified, trim or loop tracks as needed
            if target_duration and target_duration > 0:
                if total_duration < target_duration:
                    # Need to loop tracks to reach target duration
                    music_info = self._extend_music_to_duration(music_info, target_duration, crossfade_duration)
                elif total_duration > target_duration:
                    # Need to trim tracks to fit target duration
                    music_info = self._trim_music_to_duration(music_info, target_duration, crossfade_duration)
            
            # Create filter complex for crossfade
            filter_complex = self._create_crossfade_filter(music_info, crossfade_duration)
            
            # Generate FFmpeg command
            cmd = [self.ffmpeg_path, "-y"]
            
            # Add input files
            for info in music_info:
                cmd.extend(["-i", info["path"]])
            
            # Add filter complex
            cmd.extend([
                "-filter_complex", filter_complex,
                "-c:a", "aac",
                "-b:a", "192k",
                output_path
            ])
            
            # Run FFmpeg
            logger.info(f"Creating music mix with crossfades: {cmd}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Check if output was created
            if not os.path.exists(output_path):
                return {
                    "status": "error",
                    "error": "Failed to create music mix"
                }
            
            # Get final duration
            output_info = self._get_media_info(output_path)
            output_duration = float(output_info.get('duration', 0))
            
            return {
                "status": "success",
                "output_path": output_path,
                "duration": output_duration,
                "tracks": len(music_info),
                "crossfade_duration": crossfade_duration
            }
            
        except Exception as e:
            logger.error(f"Error creating music mix: {str(e)}")
            return {
                "status": "error",
                "error": f"Error creating music mix: {str(e)}"
            }
    
    def _detect_speech_segments(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detect speech segments in a video using audio analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of speech segments with start and end times
        """
        logger.info(f"Detecting speech segments in: {video_path}")
        
        # Extract audio to temporary file for analysis
        temp_audio = os.path.join(
            self.temp_dir, 
            f"{os.path.splitext(os.path.basename(video_path))[0]}_audio.wav"
        )
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM format
            "-ar", "16000",  # 16kHz sample rate (good for speech)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output
            temp_audio
        ]
        
        try:
            # Extract audio
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Detect speech using silencedetect filter
            cmd = [
                self.ffmpeg_path,
                "-i", temp_audio,
                "-af", f"silencedetect=noise={self.speech_detection_threshold}dB:d={self.speech_min_duration}",
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse silencedetect output
            silence_segments = []
            
            for line in result.stderr.splitlines():
                if "silence_start" in line:
                    try:
                        silence_start = float(line.split("silence_start: ")[1])
                        silence_segments.append({"start": silence_start})
                    except (IndexError, ValueError):
                        pass
                elif "silence_end" in line:
                    try:
                        silence_end = float(line.split("silence_end: ")[1].split(" ")[0])
                        if silence_segments and "end" not in silence_segments[-1]:
                            silence_segments[-1]["end"] = silence_end
                    except (IndexError, ValueError):
                        pass
            
            # Clean up temp file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            # Convert silence segments to speech segments
            video_info = self._get_media_info(video_path)
            video_duration = float(video_info.get('duration', 0))
            
            speech_segments = []
            last_end = 0
            
            for silence in silence_segments:
                if "start" in silence and "end" in silence:
                    # Add speech segment before silence if there's a gap
                    if silence["start"] > last_end:
                        speech_segments.append({
                            "start": last_end,
                            "end": silence["start"]
                        })
                    
                    last_end = silence["end"]
            
            # Add final speech segment if needed
            if last_end < video_duration:
                speech_segments.append({
                    "start": last_end,
                    "end": video_duration
                })
            
            # Merge speech segments that are close together
            merged_segments = self._merge_segments(speech_segments)
            
            logger.info(f"Detected {len(merged_segments)} speech segments")
            return merged_segments
            
        except Exception as e:
            logger.error(f"Error detecting speech: {str(e)}")
            # Return empty list on error
            return []
    
    def _merge_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge speech segments that are close together.
        
        Args:
            segments: List of segments with start and end times
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        # Merge segments
        merged = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            last = merged[-1]
            
            # If current segment starts close to the end of the last one, merge them
            if segment["start"] - last["end"] <= self.speech_merge_threshold:
                last["end"] = segment["end"]
            else:
                merged.append(segment)
        
        return merged
    
    def _generate_volume_automation(self, 
                                   speech_segments: List[Dict[str, Any]],
                                   video_duration: float,
                                   music_start_time: float = 0.0,
                                   music_end_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate volume automation based on speech segments.
        
        Args:
            speech_segments: List of speech segments with start and end times
            video_duration: Duration of the video in seconds
            music_start_time: Time to start music in the video
            music_end_time: Time to end music (defaults to video end)
            
        Returns:
            Dictionary with volume automation data
        """
        if music_end_time is None:
            music_end_time = video_duration
        
        # Initialize volume points with default volume
        volume_points = []
        
        # Add initial volume point
        volume_points.append({
            "time": music_start_time,
            "volume": self.default_music_volume
        })
        
        # Process each speech segment
        for segment in speech_segments:
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Skip segments outside music range
            if end_time < music_start_time or start_time > music_end_time:
                continue
            
            # Adjust segment boundaries to music range
            if start_time < music_start_time:
                start_time = music_start_time
            if end_time > music_end_time:
                end_time = music_end_time
            
            # Add fade out point before speech (if not already at low volume)
            fade_out_start = max(music_start_time, start_time - self.fade_out_time)
            
            # Only add fade out if previous point wasn't already at ducking volume
            if not volume_points or volume_points[-1]["volume"] > self.ducking_amount:
                volume_points.append({
                    "time": fade_out_start,
                    "volume": self.default_music_volume
                })
            
            # Add low volume point at speech start
            volume_points.append({
                "time": start_time,
                "volume": self.ducking_amount
            })
            
            # Add low volume point at speech end
            volume_points.append({
                "time": end_time,
                "volume": self.ducking_amount
            })
            
            # Add fade in point after speech
            fade_in_end = min(music_end_time, end_time + self.fade_in_time)
            volume_points.append({
                "time": fade_in_end,
                "volume": self.default_music_volume
            })
        
        # Add final volume point if needed
        if not volume_points or volume_points[-1]["time"] < music_end_time:
            volume_points.append({
                "time": music_end_time,
                "volume": self.default_music_volume
            })
        
        # Clean up and simplify volume points
        simplified_points = self._simplify_volume_points(volume_points)
        
        return {
            "volume_points": simplified_points,
            "music_start": music_start_time,
            "music_end": music_end_time
        }
    
    def _simplify_volume_points(self, volume_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simplify volume points by removing unnecessary ones.
        
        Args:
            volume_points: List of volume points with time and volume
            
        Returns:
            Simplified list of volume points
        """
        if len(volume_points) <= 2:
            return volume_points
        
        # Sort by time
        sorted_points = sorted(volume_points, key=lambda x: x["time"])
        
        # Simplify by removing points with same volume as neighbors
        simplified = [sorted_points[0]]
        
        for i in range(1, len(sorted_points) - 1):
            current = sorted_points[i]
            prev = simplified[-1]
            next_point = sorted_points[i + 1]
            
            # Keep if volume is different from either neighbor
            if abs(current["volume"] - prev["volume"]) > 0.01 or abs(current["volume"] - next_point["volume"]) > 0.01:
                simplified.append(current)
        
        # Always keep last point
        simplified.append(sorted_points[-1])
        
        return simplified
    
    def _apply_volume_automation(self,
                                video_path: str,
                                music_path: str,
                                output_path: str,
                                volume_automation: Dict[str, Any],
                                music_start_time: float = 0.0,
                                music_end_time: Optional[float] = None,
                                keep_original_audio: bool = True) -> Dict[str, Any]:
        """
        Apply volume automation to music and mix with video.
        
        Args:
            video_path: Path to the video file
            music_path: Path to the music file
            output_path: Path for the output video
            volume_automation: Dictionary with volume automation data
            music_start_time: Time to start music in the video
            music_end_time: Time to end music (defaults to video end)
            keep_original_audio: Whether to keep original audio from video
            
        Returns:
            Dictionary with operation results
        """
        # Get volume points
        volume_points = volume_automation.get("volume_points", [])
        
        if not volume_points:
            return {
                "status": "error",
                "error": "No volume points provided for automation"
            }
        
        # Get video info
        video_info = self._get_media_info(video_path)
        video_duration = float(video_info.get('duration', 0))
        
        if music_end_time is None:
            music_end_time = video_duration
        
        try:
            # Prepare FFmpeg filter complex for volume automation
            volume_expr = self._create_volume_expression(volume_points, music_start_time)
            
            # Prepare full command
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", video_path,  # First input: video
                "-i", music_path,  # Second input: music
                "-filter_complex"
            ]
            
            if keep_original_audio:
                # Mix original audio with volume-adjusted music
                filter_complex = (
                    # Apply volume automation to music
                    f"[1:a]adelay={int(music_start_time*1000)}|{int(music_start_time*1000)},"
                    f"volume='{volume_expr}'[music];"
                    # Mix with original audio
                    f"[0:a][music]amix=inputs=2:duration=first[a]"
                )
            else:
                # Only use volume-adjusted music
                filter_complex = (
                    f"[1:a]adelay={int(music_start_time*1000)}|{int(music_start_time*1000)},"
                    f"volume='{volume_expr}'[a]"
                )
            
            cmd.append(filter_complex)
            
            # Map streams to output
            cmd.extend([
                "-map", "0:v",  # Video from input 0
                "-map", "[a]",  # Audio from filter output
                "-c:v", "copy",  # Copy video codec
                "-c:a", "aac",  # AAC audio codec
                "-b:a", "192k",  # Audio bitrate
                "-shortest",  # End when shortest input ends
                output_path
            ])
            
            # Run FFmpeg command
            logger.info(f"Applying volume automation: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Check if output was created
            if not os.path.exists(output_path):
                return {
                    "status": "error",
                    "error": "Failed to create output video"
                }
            
            return {
                "status": "success",
                "output_path": output_path,
                "volume_points": volume_points
            }
            
        except Exception as e:
            logger.error(f"Error applying volume automation: {str(e)}")
            return {
                "status": "error",
                "error": f"Error applying volume automation: {str(e)}"
            }
    
    def _create_volume_expression(self, volume_points: List[Dict[str, Any]], offset: float = 0.0) -> str:
        """
        Create FFmpeg volume expression for dynamic volume adjustment.
        
        Args:
            volume_points: List of volume points with time and volume
            offset: Time offset to apply to volume points
            
        Returns:
            FFmpeg volume expression string
        """
        # Adjust times for offset and sort by time
        adjusted_points = [
            {"time": max(0, p["time"] - offset), "volume": p["volume"]} 
            for p in volume_points
        ]
        adjusted_points.sort(key=lambda x: x["time"])
        
        # Build expression: if(between(t,t1,t2),vol1+(vol2-vol1)*(t-t1)/(t2-t1),...)
        parts = []
        
        for i in range(len(adjusted_points) - 1):
            t1 = adjusted_points[i]["time"]
            t2 = adjusted_points[i + 1]["time"]
            vol1 = adjusted_points[i]["volume"]
            vol2 = adjusted_points[i + 1]["volume"]
            
            # If times are the same, avoid division by zero
            if abs(t2 - t1) < 0.001:
                expr = f"if(between(t,{t1},{t2}),{vol1}"
            else:
                expr = f"if(between(t,{t1},{t2}),{vol1}+({vol2}-{vol1})*(t-{t1})/({t2}-{t1})"
            
            parts.append(expr)
        
        # Add default value for times outside defined ranges
        if adjusted_points:
            first_time = adjusted_points[0]["time"]
            first_vol = adjusted_points[0]["volume"]
            last_time = adjusted_points[-1]["time"]
            last_vol = adjusted_points[-1]["volume"]
            
            # Before first point
            if first_time > 0:
                parts.insert(0, f"if(lt(t,{first_time}),{first_vol}")
            
            # After last point
            parts.append(f"{last_vol}")
        else:
            # No points, use default volume
            return "1.0"
        
        # Close all if statements
        expression = "".join(parts) + ")" * (len(parts) - 1)
        
        return expression
    
    def _create_crossfade_filter(self, 
                                music_info: List[Dict[str, Any]], 
                                crossfade_duration: float) -> str:
        """
        Create FFmpeg filter complex for crossfading between music tracks.
        
        Args:
            music_info: List of dictionaries with music file info
            crossfade_duration: Duration of crossfades in seconds
            
        Returns:
            FFmpeg filter complex string
        """
        if len(music_info) == 1:
            # Single track, no crossfade needed
            return "[0:a]acopy[a]"
        
        filter_parts = []
        
        # Calculate timestamps for each track
        timestamps = [0]
        for i in range(1, len(music_info)):
            prev_start = timestamps[i-1]
            prev_duration = music_info[i-1]["duration"]
            # Start next track before previous ends by crossfade duration
            next_start = prev_start + prev_duration - crossfade_duration
            timestamps.append(next_start)
        
        # Build filter complex
        for i, (info, timestamp) in enumerate(zip(music_info, timestamps)):
            # Trim and set timestamps
            filter_parts.append(f"[{i}:a]atrim=start=0:duration={info['duration']},asetpts=PTS-STARTPTS[a{i}]")
        
        # Crossfade tracks
        for i in range(len(music_info) - 1):
            if i == 0:
                # First crossfade
                filter_parts.append(
                    f"[a{i}][a{i+1}]acrossfade=d={crossfade_duration}:c1=tri:c2=tri[af{i}]"
                )
            else:
                # Subsequent crossfades
                filter_parts.append(
                    f"[af{i-1}][a{i+1}]acrossfade=d={crossfade_duration}:c1=tri:c2=tri[af{i}]"
                )
        
        # Add final output map
        filter_parts.append(f"[af{len(music_info)-2}]")
        
        return ";".join(filter_parts)
    
    def _extend_music_to_duration(self, 
                                 music_info: List[Dict[str, Any]], 
                                 target_duration: float,
                                 crossfade_duration: float) -> List[Dict[str, Any]]:
        """
        Extend music tracks to reach target duration by looping.
        
        Args:
            music_info: List of dictionaries with music file info
            target_duration: Target duration in seconds
            crossfade_duration: Duration of crossfades in seconds
            
        Returns:
            Extended music info list
        """
        extended_info = music_info.copy()
        current_duration = sum(track["duration"] for track in extended_info)
        
        # Subtract crossfade durations
        current_duration -= crossfade_duration * (len(extended_info) - 1)
        
        # Keep adding tracks until we reach target duration
        while current_duration < target_duration:
            # Choose a track to repeat (for variety, pick the least used one)
            track_counts = {}
            for track in extended_info:
                path = track["path"]
                track_counts[path] = track_counts.get(path, 0) + 1
            
            # Find least used track
            least_used = min(music_info, key=lambda x: track_counts.get(x["path"], 0))
            
            # Add it to extended info
            extended_info.append(least_used)
            
            # Update duration (accounting for crossfade)
            current_duration += least_used["duration"] - crossfade_duration
        
        return extended_info
    
    def _trim_music_to_duration(self, 
                                music_info: List[Dict[str, Any]], 
                                target_duration: float,
                                crossfade_duration: float) -> List[Dict[str, Any]]:
        """
        Trim music tracks to fit target duration.
        
        Args:
            music_info: List of dictionaries with music file info
            target_duration: Target duration in seconds
            crossfade_duration: Duration of crossfades in seconds
            
        Returns:
            Trimmed music info list
        """
        # Calculate current total duration
        total_duration = sum(track["duration"] for track in music_info)
        total_duration -= crossfade_duration * (len(music_info) - 1)  # Account for crossfades
        
        # If already shorter, return as is
        if total_duration <= target_duration:
            return music_info
        
        # Calculate how much to trim
        excess_duration = total_duration - target_duration
        
        # Start by removing tracks from the end if needed
        trimmed_info = music_info.copy()
        
        while len(trimmed_info) > 1:
            last_track = trimmed_info[-1]
            track_duration = last_track["duration"] - crossfade_duration
            
            if excess_duration >= track_duration:
                # Can remove this track entirely
                excess_duration -= track_duration
                trimmed_info.pop()
            else:
                break
        
        # If still need to trim, adjust last track duration
        if excess_duration > 0 and len(trimmed_info) > 0:
            last_track = trimmed_info[-1]
            new_duration = max(crossfade_duration * 2, last_track["duration"] - excess_duration)
            
            trimmed_info[-1] = {
                "path": last_track["path"],
                "duration": new_duration
            }
        
        return trimmed_info
    
    def _get_media_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get media file information using FFprobe.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            Dictionary with media information
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
            "-of", "json",
            file_path
        ]
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data = json.loads(result.stdout)
            
            media_info = {}
            
            # Extract format information
            if 'format' in data:
                if 'duration' in data['format']:
                    media_info['duration'] = float(data['format']['duration'])
                if 'bit_rate' in data['format']:
                    media_info['bit_rate'] = int(data['format']['bit_rate'])
            
            # Extract stream information
            if 'streams' in data and data['streams']:
                stream = data['streams'][0]
                if 'codec_name' in stream:
                    media_info['codec'] = stream['codec_name']
                if 'sample_rate' in stream:
                    media_info['sample_rate'] = int(stream['sample_rate'])
                if 'channels' in stream:
                    media_info['channels'] = int(stream['channels'])
            
            # If duration not in format, try to get it from the stream
            if 'duration' not in media_info and 'streams' in data and data['streams']:
                stream = data['streams'][0]
                if 'duration' in stream:
                    media_info['duration'] = float(stream['duration'])
            
            return media_info
            
        except Exception as e:
            logger.error(f"Error getting media info: {str(e)}")
            return {'duration': 0, 'codec': 'unknown', 'sample_rate': 44100, 'channels': 2} 