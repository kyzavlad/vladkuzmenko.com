"""
BPM Detection and Matching Module

This module provides functionality for detecting the tempo (BPM) of audio tracks
and matching tracks based on target BPM ranges.
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

class BPMDetector:
    """
    Detects and analyzes tempo (BPM) of audio tracks and provides matching capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the BPM detector with configuration options."""
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
        
        # BPM range definitions
        self.bpm_ranges = {
            "very_slow": (0, 70),
            "slow": (70, 90),
            "moderate": (90, 120),
            "fast": (120, 150),
            "very_fast": (150, 300)
        }
        
        # Initialize audio libraries
        self.librosa_available = False
        try:
            import librosa
            self.librosa_available = True
            logger.info("Librosa available for BPM detection")
        except ImportError:
            logger.warning("Librosa not available. BPM detection will use fallback methods.")
    
    def detect_bpm(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect the BPM (tempo) of an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with BPM detection results
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            return {
                "status": "error",
                "error": f"File not found: {audio_path}"
            }
        
        # Check if it's a video file
        is_video = self._is_video_file(audio_path)
        
        # For video files, extract audio first
        temp_audio = None
        try:
            if is_video:
                temp_audio = self._extract_audio_from_video(audio_path)
                if not temp_audio:
                    return {
                        "status": "error",
                        "error": "Failed to extract audio from video file"
                    }
                analysis_path = temp_audio
            else:
                analysis_path = audio_path
            
            # Detect BPM
            bpm_result = self._detect_bpm_from_audio(analysis_path)
            
            # Clean up temporary file if needed
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            # Check for errors
            if "error" in bpm_result:
                return {
                    "status": "error",
                    "error": bpm_result["error"]
                }
            
            # Categorize the BPM
            bpm = bpm_result["bpm"]
            bpm_category = self._categorize_bpm(bpm)
            
            # Format and return the result
            return {
                "status": "success",
                "file_path": audio_path,
                "bpm": bpm,
                "confidence": bpm_result.get("confidence", None),
                "category": bpm_category,
                "range": self.bpm_ranges[bpm_category]
            }
            
        except Exception as e:
            logger.error(f"Error detecting BPM: {str(e)}")
            
            # Clean up any temporary files
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
                
            return {
                "status": "error",
                "error": f"Error detecting BPM: {str(e)}"
            }
    
    def find_matching_tracks(self, 
                            target_bpm: float, 
                            audio_files: List[str],
                            tolerance: float = 5.0,
                            match_style: str = "exact") -> Dict[str, Any]:
        """
        Find tracks that match a target BPM.
        
        Args:
            target_bpm: The target BPM to match
            audio_files: List of paths to audio files
            tolerance: BPM tolerance range (+/-)
            match_style: Matching style: "exact" (within tolerance), 
                        "double" (also match double/half tempo),
                        "harmonic" (match harmonically related tempos)
            
        Returns:
            Dictionary with matching results
        """
        if not audio_files:
            return {
                "status": "error",
                "error": "No audio files provided"
            }
        
        # Process all files
        results = []
        errors = []
        
        for audio_path in audio_files:
            # Skip files that don't exist
            if not os.path.exists(audio_path):
                errors.append({
                    "file_path": audio_path,
                    "error": "File not found"
                })
                continue
                
            # Detect BPM
            detection = self.detect_bpm(audio_path)
            
            # Skip files with errors
            if detection["status"] != "success":
                errors.append({
                    "file_path": audio_path,
                    "error": detection.get("error", "Unknown error")
                })
                continue
            
            # Calculate match score based on chosen matching style
            detected_bpm = detection["bpm"]
            
            if match_style == "exact":
                # Simple exact matching within tolerance
                bpm_difference = abs(detected_bpm - target_bpm)
                match_score = self._calculate_match_score(bpm_difference, tolerance)
                
            elif match_style == "double":
                # Match considering double/half tempo relationships
                bpm_difference = min(
                    abs(detected_bpm - target_bpm),          # Direct match
                    abs(detected_bpm - target_bpm * 2),      # Double tempo
                    abs(detected_bpm - target_bpm / 2)        # Half tempo
                )
                match_score = self._calculate_match_score(bpm_difference, tolerance)
                
            elif match_style == "harmonic":
                # Match considering harmonic tempo relationships (1/2, 2/3, 3/4, etc)
                harmonic_differences = [
                    abs(detected_bpm - target_bpm),          # 1:1
                    abs(detected_bpm - target_bpm * 2),      # 1:2
                    abs(detected_bpm - target_bpm / 2),       # 2:1
                    abs(detected_bpm - target_bpm * 3/2),    # 2:3
                    abs(detected_bpm - target_bpm * 2/3),    # 3:2
                    abs(detected_bpm - target_bpm * 4/3),    # 3:4
                    abs(detected_bpm - target_bpm * 3/4)     # 4:3
                ]
                bpm_difference = min(harmonic_differences)
                match_score = self._calculate_match_score(bpm_difference, tolerance)
            else:
                # Default to exact matching
                bpm_difference = abs(detected_bpm - target_bpm)
                match_score = self._calculate_match_score(bpm_difference, tolerance)
            
            # Only include if there's some level of match
            if match_score > 0:
                results.append({
                    "file_path": audio_path,
                    "bpm": detected_bpm,
                    "category": detection["category"],
                    "match_score": match_score,
                    "bpm_difference": bpm_difference
                })
        
        # Sort by match score (descending)
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "status": "success",
            "target_bpm": target_bpm,
            "tolerance": tolerance,
            "match_style": match_style,
            "matches": results,
            "errors": errors
        }
    
    def match_bpm_range(self, 
                       bpm_range: Tuple[float, float], 
                       audio_files: List[str]) -> Dict[str, Any]:
        """
        Find tracks within a BPM range.
        
        Args:
            bpm_range: Tuple of (min_bpm, max_bpm)
            audio_files: List of paths to audio files
            
        Returns:
            Dictionary with matching results
        """
        if not audio_files:
            return {
                "status": "error",
                "error": "No audio files provided"
            }
        
        min_bpm, max_bpm = bpm_range
        
        # Process all files
        results = []
        errors = []
        
        for audio_path in audio_files:
            # Skip files that don't exist
            if not os.path.exists(audio_path):
                errors.append({
                    "file_path": audio_path,
                    "error": "File not found"
                })
                continue
                
            # Detect BPM
            detection = self.detect_bpm(audio_path)
            
            # Skip files with errors
            if detection["status"] != "success":
                errors.append({
                    "file_path": audio_path,
                    "error": detection.get("error", "Unknown error")
                })
                continue
            
            # Check if BPM is within range
            detected_bpm = detection["bpm"]
            if min_bpm <= detected_bpm <= max_bpm:
                # Calculate position within the range (0-1)
                range_position = (detected_bpm - min_bpm) / (max_bpm - min_bpm) if max_bpm > min_bpm else 0.5
                
                results.append({
                    "file_path": audio_path,
                    "bpm": detected_bpm,
                    "category": detection["category"],
                    "range_position": range_position
                })
        
        # Sort by BPM (ascending)
        results.sort(key=lambda x: x["bpm"])
        
        return {
            "status": "success",
            "bpm_range": bpm_range,
            "matches": results,
            "errors": errors
        }
    
    def suggest_bpm_for_content(self, content_type: str) -> Dict[str, Any]:
        """
        Suggest an appropriate BPM range for different content types.
        
        Args:
            content_type: Type of content (e.g., "interview", "action", "documentary")
            
        Returns:
            Dictionary with BPM suggestions
        """
        # Define BPM recommendations for different content types
        content_bpm_map = {
            # Calm content
            "interview": (70, 90),
            "documentary": (80, 110),
            "tutorial": (85, 105),
            "corporate": (90, 110),
            
            # Medium energy
            "vlog": (90, 120),
            "travel": (90, 125),
            "lifestyle": (100, 120),
            "promotional": (100, 130),
            
            # High energy
            "action": (120, 160),
            "sports": (125, 165),
            "advertisement": (110, 140),
            "gaming": (120, 170),
            
            # Emotional
            "dramatic": (60, 90),
            "inspirational": (70, 100),
            "emotional": (60, 85)
        }
        
        # Get recommendation for content type
        if content_type.lower() in content_bpm_map:
            bpm_range = content_bpm_map[content_type.lower()]
            avg_bpm = sum(bpm_range) / 2
            
            return {
                "status": "success",
                "content_type": content_type,
                "bpm_range": bpm_range,
                "suggested_bpm": avg_bpm,
                "category": self._categorize_bpm(avg_bpm)
            }
        else:
            # Default to moderate range if content type unknown
            return {
                "status": "warning",
                "message": f"Unknown content type: {content_type}. Using default values.",
                "content_type": content_type,
                "bpm_range": (90, 120),
                "suggested_bpm": 105,
                "category": "moderate"
            }
    
    def _detect_bpm_from_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect BPM from audio file using available methods.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with BPM and confidence score
        """
        # Try to use librosa if available (more accurate)
        if self.librosa_available:
            try:
                import librosa
                
                # Load the audio file
                y, sr = librosa.load(audio_path, sr=None)
                
                # Run tempo detection
                # This uses the tempo and beat tracking algorithms in librosa
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                
                # Calculate confidence based on beat strength
                if len(beats) > 0:
                    beat_strengths = onset_env[beats]
                    confidence = min(1.0, float(np.mean(beat_strengths) / np.max(onset_env) if np.max(onset_env) > 0 else 0.5))
                else:
                    confidence = 0.5
                
                return {
                    "bpm": float(tempo),
                    "confidence": confidence
                }
                
            except Exception as e:
                logger.warning(f"Error detecting BPM with librosa: {str(e)}. Falling back to FFmpeg.")
        
        # Fallback to FFmpeg-based tempo detection
        try:
            # Use FFmpeg's ebur128 filter to analyze audio
            cmd = [
                self.ffmpeg_path,
                "-i", audio_path,
                "-filter_complex", "ebur128=peak=true",
                "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse the output to estimate BPM
            # This is a simplified approach using peak levels to estimate rhythm
            # For precise BPM detection, we'd need more advanced analysis
            
            # As a fallback, we'll use a combination of audio duration and estimated beats
            # to make a reasonable guess about tempo
            audio_info = self._get_audio_info(audio_path)
            duration = audio_info.get('duration', 0)
            
            if duration <= 0:
                return {"error": "Could not determine audio duration"}
            
            # Analyze energy peaks to estimate beat count
            # This is a very simplified approach
            peaks = 0
            last_peak_time = 0
            peak_intervals = []
            
            for line in result.stderr.splitlines():
                if "t:" in line and "M:" in line:
                    try:
                        # Extract timestamp and momentary loudness
                        time_match = line.split("t:")[1].split()[0]
                        current_time = float(time_match)
                        
                        # Simple peak detection
                        if current_time - last_peak_time > 0.2:  # Minimum peak interval
                            peaks += 1
                            if last_peak_time > 0:
                                peak_intervals.append(current_time - last_peak_time)
                            last_peak_time = current_time
                    except:
                        pass
            
            # Estimate BPM based on peak intervals
            if peak_intervals:
                avg_interval = sum(peak_intervals) / len(peak_intervals)
                estimated_bpm = 60.0 / avg_interval
                
                # Cap at reasonable values
                estimated_bpm = max(60, min(180, estimated_bpm))
                
                # Low confidence for this method
                return {
                    "bpm": estimated_bpm,
                    "confidence": 0.3  # Low confidence in the FFmpeg method
                }
            
            # If peak detection failed, use a reasonable default
            return {
                "bpm": 120.0,  # Default moderate tempo
                "confidence": 0.1  # Very low confidence
            }
            
        except Exception as e:
            return {"error": f"Error detecting BPM with FFmpeg: {str(e)}"}
    
    def _extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio track from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file, or None if extraction failed
        """
        try:
            # Create a temporary file path for the extracted audio
            audio_path = os.path.join(
                self.temp_dir, 
                f"{os.path.splitext(os.path.basename(video_path))[0]}_audio.wav"
            )
            
            # Use FFmpeg to extract audio
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM format
                "-ar", "44100",  # 44.1kHz sample rate
                "-ac", "2",  # Stereo
                "-y",  # Overwrite output
                audio_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(audio_path):
                return audio_path
            return None
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return None
    
    def _is_video_file(self, file_path: str) -> bool:
        """
        Check if a file is a video file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if it's a video file, False otherwise
        """
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        ext = os.path.splitext(file_path)[1].lower()
        return ext in video_extensions
    
    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """
        Get audio file information using FFprobe.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with audio information
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "format=duration,bit_rate:stream=sample_rate,channels",
            "-of", "json",
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data = json.loads(result.stdout)
            
            audio_info = {}
            
            # Extract format information
            if 'format' in data:
                if 'duration' in data['format']:
                    audio_info['duration'] = float(data['format']['duration'])
                if 'bit_rate' in data['format']:
                    audio_info['bit_rate'] = int(data['format']['bit_rate'])
            
            # Extract stream information
            if 'streams' in data and data['streams']:
                stream = data['streams'][0]
                if 'sample_rate' in stream:
                    audio_info['sample_rate'] = int(stream['sample_rate'])
                if 'channels' in stream:
                    audio_info['channels'] = int(stream['channels'])
            
            return audio_info
            
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return {'duration': 0, 'sample_rate': 44100, 'channels': 2, 'bit_rate': 128000}
    
    def _categorize_bpm(self, bpm: float) -> str:
        """
        Categorize a BPM value into a tempo category.
        
        Args:
            bpm: BPM value
            
        Returns:
            Category name (very_slow, slow, moderate, fast, very_fast)
        """
        for category, (min_bpm, max_bpm) in self.bpm_ranges.items():
            if min_bpm <= bpm < max_bpm:
                return category
        
        # Default if out of all ranges
        if bpm < self.bpm_ranges["very_slow"][0]:
            return "very_slow"
        return "very_fast"
    
    def _calculate_match_score(self, bpm_difference: float, tolerance: float) -> float:
        """
        Calculate a match score based on BPM difference and tolerance.
        
        Args:
            bpm_difference: Difference between target and detected BPM
            tolerance: Tolerance range
            
        Returns:
            Match score (0.0-1.0)
        """
        if bpm_difference <= tolerance:
            # Linear score based on how close to the target BPM
            return 1.0 - (bpm_difference / tolerance)
        return 0.0 