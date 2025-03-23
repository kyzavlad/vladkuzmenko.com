"""
Sound Effects Processor

This module provides functionality for processing sound effects,
including spatial audio positioning, intensity adjustment based on
scene dynamics, and mixing sound effects with video content.
"""

import os
import logging
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import time

logger = logging.getLogger(__name__)

class SoundEffectsProcessor:
    """
    Processor for sound effects that handles spatial audio positioning,
    intensity adjustment, and mixing with video content.
    
    Features:
    - Spatial audio positioning (stereo and 5.1 surround)
    - Intensity adjustment based on scene dynamics
    - Professional mixing and mastering
    - Integration with video content
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sound effects processor.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get(
            'temp_dir', 
            os.path.join(tempfile.gettempdir(), 'sound_effects_processor')
        )
        
        # Create directories if they don't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Default audio settings
        self.default_sample_rate = self.config.get('sample_rate', 44100)
        self.default_channels = self.config.get('channels', 2)  # Stereo by default
        
        # Check for required tools
        self._check_tools()
    
    def apply_spatial_positioning(
        self,
        input_file: str,
        output_file: str,
        position: Dict[str, float],
        channels: int = 2,
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Apply spatial positioning to a sound effect.
        
        Args:
            input_file: Path to the input sound effect file
            output_file: Path to the output file
            position: Dictionary with position data (x, y, z coordinates)
                     x: -1 (left) to 1 (right)
                     y: -1 (back) to 1 (front) - for surround sound
                     z: -1 (below) to 1 (above) - for height channels
            channels: Number of output channels (2 for stereo, 6 for 5.1 surround)
            intensity: Volume intensity (0.0 to 1.0)
            
        Returns:
            Dictionary with the result of the operation
        """
        try:
            # Validate position data
            x = max(-1.0, min(1.0, position.get('x', 0.0)))
            y = max(-1.0, min(1.0, position.get('y', 0.0)))
            z = max(-1.0, min(1.0, position.get('z', 0.0)))
            
            # Ensure intensity is in valid range
            intensity = max(0.0, min(1.0, intensity))
            
            # Generate FFmpeg filter command
            filter_cmd = []
            
            if channels == 2:  # Stereo
                # Calculate left/right balance based on x position
                # x = -1: full left, x = 1: full right, x = 0: center
                pan_left = 1.0 - max(0, x)
                pan_right = 1.0 + min(0, x)
                
                # Apply intensity
                pan_left *= intensity
                pan_right *= intensity
                
                # Create pan filter for stereo
                filter_cmd = [
                    '-filter_complex',
                    f'pan=stereo|c0={pan_left}*c0|c1={pan_right}*c1'
                ]
                
            elif channels == 6:  # 5.1 surround
                # Channel mapping in 5.1: FL, FR, FC, LFE, BL, BR
                # Calculate channel coefficients based on 3D position
                
                # Horizontal position (x) affects FL/FR and BL/BR
                # Vertical position (y) affects front/back balance
                # Height (z) affects overall distribution
                
                # Front vs back blend based on y position
                front_blend = (y + 1) / 2.0  # 0 to 1
                back_blend = 1.0 - front_blend
                
                # Left vs right blend based on x position
                left_blend = (1 - x) / 2.0  # 0 to 1
                right_blend = 1.0 - left_blend
                
                # Apply intensity to all channels
                front_left = left_blend * front_blend * intensity
                front_right = right_blend * front_blend * intensity
                center = (0.5 * front_blend * intensity) if abs(x) < 0.5 else 0.0
                lfe = (0.3 * intensity) if z < 0 else 0.0  # LFE stronger for low sounds
                back_left = left_blend * back_blend * intensity
                back_right = right_blend * back_blend * intensity
                
                # Create pan filter for 5.1
                filter_cmd = [
                    '-filter_complex',
                    (f'pan=5.1|'
                     f'c0={front_left}*c0|'
                     f'c1={front_right}*c0|'
                     f'c2={center}*c0|'
                     f'c3={lfe}*c0|'
                     f'c4={back_left}*c0|'
                     f'c5={back_right}*c0')
                ]
            else:
                # For other channel configurations, just adjust volume
                filter_cmd = [
                    '-filter:a',
                    f'volume={intensity}'
                ]
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', input_file,
                *filter_cmd,
                '-c:a', 'aac',
                '-b:a', '192k',
                output_file
            ]
            
            # Execute FFmpeg command
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return {
                "status": "success",
                "input_file": input_file,
                "output_file": output_file,
                "position": {"x": x, "y": y, "z": z},
                "intensity": intensity,
                "channels": channels
            }
            
        except Exception as e:
            logger.error(f"Error applying spatial positioning: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to apply spatial positioning: {str(e)}"
            }
    
    def adjust_intensity(
        self,
        input_file: str,
        output_file: str,
        intensity: float,
        fade_in: Optional[float] = None,
        fade_out: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Adjust the intensity of a sound effect with optional fade in/out.
        
        Args:
            input_file: Path to the input sound effect file
            output_file: Path to the output file
            intensity: Volume intensity (0.0 to 1.0)
            fade_in: Fade in time in seconds (optional)
            fade_out: Fade out time in seconds (optional)
            
        Returns:
            Dictionary with the result of the operation
        """
        try:
            # Ensure intensity is in valid range
            intensity = max(0.0, min(1.0, intensity))
            
            # Base FFmpeg filter
            filter_str = f'volume={intensity}'
            
            # Add fade effects if provided
            if fade_in is not None or fade_out is not None:
                # Get duration of audio file
                duration = self._get_audio_duration(input_file)
                
                fade_filters = []
                if fade_in is not None and fade_in > 0:
                    fade_filters.append(f'afade=t=in:st=0:d={fade_in}')
                
                if fade_out is not None and fade_out > 0 and duration > fade_out:
                    fade_start = max(0, duration - fade_out)
                    fade_filters.append(f'afade=t=out:st={fade_start}:d={fade_out}')
                
                if fade_filters:
                    filter_str = f'{filter_str},{",".join(fade_filters)}'
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', input_file,
                '-filter:a', filter_str,
                '-c:a', 'aac',
                '-b:a', '192k',
                output_file
            ]
            
            # Execute FFmpeg command
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return {
                "status": "success",
                "input_file": input_file,
                "output_file": output_file,
                "intensity": intensity,
                "fade_in": fade_in,
                "fade_out": fade_out
            }
            
        except Exception as e:
            logger.error(f"Error adjusting intensity: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to adjust intensity: {str(e)}"
            }
    
    def add_sound_effect_to_video(
        self,
        video_file: str,
        sound_file: str,
        output_file: str,
        start_time: float,
        intensity: float = 1.0,
        position: Optional[Dict[str, float]] = None,
        keep_original_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Add a sound effect to a video at a specific time.
        
        Args:
            video_file: Path to the input video file
            sound_file: Path to the sound effect file
            output_file: Path to the output video file
            start_time: Start time in seconds for the sound effect
            intensity: Volume intensity for the sound effect (0.0 to 1.0)
            position: Optional spatial position for the sound effect
            keep_original_audio: Whether to keep the original video audio
            
        Returns:
            Dictionary with the result of the operation
        """
        try:
            # Create a temp file for the processed sound effect
            temp_sound_file = os.path.join(self.temp_dir, f'temp_sound_{int(time.time())}.wav')
            
            # Apply spatial positioning if position is provided
            if position:
                positioning_result = self.apply_spatial_positioning(
                    input_file=sound_file,
                    output_file=temp_sound_file,
                    position=position,
                    intensity=intensity
                )
                
                if positioning_result['status'] != 'success':
                    return positioning_result
                
                processed_sound_file = temp_sound_file
            else:
                # Otherwise just adjust intensity
                intensity_result = self.adjust_intensity(
                    input_file=sound_file,
                    output_file=temp_sound_file,
                    intensity=intensity
                )
                
                if intensity_result['status'] != 'success':
                    return intensity_result
                
                processed_sound_file = temp_sound_file
            
            # Build FFmpeg filter graph
            filter_complex = []
            
            if keep_original_audio:
                # Mix original audio with sound effect
                filter_complex = [
                    '-filter_complex',
                    (f'[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a0];'
                     f'[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,'
                     f'adelay={int(start_time*1000)}|{int(start_time*1000)}[a1];'
                     f'[a0][a1]amix=inputs=2:duration=first[aout]'),
                    '-map', '0:v',
                    '-map', '[aout]'
                ]
            else:
                # Replace original audio with sound effect
                filter_complex = [
                    '-filter_complex',
                    (f'[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,'
                     f'adelay={int(start_time*1000)}|{int(start_time*1000)}[aout]'),
                    '-map', '0:v',
                    '-map', '[aout]'
                ]
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', video_file,
                '-i', processed_sound_file,
                *filter_complex,
                '-c:v', 'copy',  # Copy video stream without re-encoding
                '-c:a', 'aac',
                '-b:a', '192k',
                output_file
            ]
            
            # Execute FFmpeg command
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Clean up temporary file
            if os.path.exists(temp_sound_file):
                os.remove(temp_sound_file)
            
            return {
                "status": "success",
                "video_file": video_file,
                "sound_file": sound_file,
                "output_file": output_file,
                "start_time": start_time,
                "intensity": intensity,
                "position": position
            }
            
        except Exception as e:
            logger.error(f"Error adding sound effect to video: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to add sound effect to video: {str(e)}"
            }
    
    def mix_multiple_sound_effects(
        self,
        sound_files: List[Dict[str, Any]],
        output_file: str,
        target_duration: Optional[float] = None,
        master_volume: float = 1.0
    ) -> Dict[str, Any]:
        """
        Mix multiple sound effects into a single audio file.
        
        Args:
            sound_files: List of dictionaries with sound effect details
                Each dictionary should have:
                - file_path: Path to the sound effect file
                - start_time: Start time in seconds
                - intensity: Volume intensity (0.0 to 1.0)
                - position: Optional spatial position
            output_file: Path to the output audio file
            target_duration: Optional target duration in seconds
            master_volume: Master volume for the final mix (0.0 to 1.0)
            
        Returns:
            Dictionary with the result of the operation
        """
        try:
            if not sound_files:
                return {
                    "status": "error",
                    "error": "No sound files provided"
                }
            
            # Create temporary directory for processed files
            temp_dir = os.path.join(self.temp_dir, f'mix_{int(time.time())}')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Process each sound file
            processed_files = []
            
            for i, sound_data in enumerate(sound_files):
                file_path = sound_data.get('file_path')
                
                if not file_path or not os.path.exists(file_path):
                    continue
                
                start_time = sound_data.get('start_time', 0.0)
                intensity = sound_data.get('intensity', 1.0)
                position = sound_data.get('position')
                
                # Create temp file for processed sound
                temp_file = os.path.join(temp_dir, f'sound_{i}.wav')
                
                # Apply positioning if provided
                if position:
                    result = self.apply_spatial_positioning(
                        input_file=file_path,
                        output_file=temp_file,
                        position=position,
                        intensity=intensity
                    )
                else:
                    result = self.adjust_intensity(
                        input_file=file_path,
                        output_file=temp_file,
                        intensity=intensity
                    )
                
                if result['status'] == 'success':
                    processed_files.append({
                        'file': temp_file,
                        'start_time': start_time
                    })
            
            if not processed_files:
                return {
                    "status": "error",
                    "error": "No valid sound files to process"
                }
            
            # Calculate the max duration if not provided
            if target_duration is None:
                max_end_time = 0
                for sound in processed_files:
                    duration = self._get_audio_duration(sound['file'])
                    end_time = sound['start_time'] + duration
                    max_end_time = max(max_end_time, end_time)
                
                target_duration = max_end_time
            
            # Create silent base track of target duration
            silent_base = os.path.join(temp_dir, 'silent_base.wav')
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'lavfi',
                '-i', f'anullsrc=r={self.default_sample_rate}:cl=stereo',
                '-t', str(target_duration),
                silent_base
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Build filter complex for mixing
            inputs = ['-i', silent_base]
            filter_parts = []
            
            for i, sound in enumerate(processed_files):
                inputs.extend(['-i', sound['file']])
                delay_ms = int(sound['start_time'] * 1000)
                filter_parts.append(
                    f'[{i+1}:a]adelay={delay_ms}|{delay_ms}[s{i}]'
                )
            
            # Mix all sounds
            mix_inputs = ''.join(f'[s{i}]' for i in range(len(processed_files)))
            filter_parts.append(
                f'[0:a]{mix_inputs}amix=inputs={len(processed_files)+1}:duration=longest[aout]'
            )
            
            filter_complex = ';'.join(filter_parts)
            
            # Apply master volume if needed
            if master_volume != 1.0:
                filter_complex += f';[aout]volume={master_volume}[afinal]'
                output_stream = '[afinal]'
            else:
                output_stream = '[aout]'
            
            # Build final FFmpeg command
            cmd = [
                'ffmpeg',
                '-y',
                *inputs,
                '-filter_complex', filter_complex,
                '-map', output_stream,
                '-c:a', 'aac',
                '-b:a', '192k',
                output_file
            ]
            
            # Execute FFmpeg command
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Clean up temporary files
            for sound in processed_files:
                if os.path.exists(sound['file']):
                    os.remove(sound['file'])
            
            if os.path.exists(silent_base):
                os.remove(silent_base)
            
            os.rmdir(temp_dir)
            
            return {
                "status": "success",
                "output_file": output_file,
                "target_duration": target_duration,
                "sound_count": len(processed_files),
                "master_volume": master_volume
            }
            
        except Exception as e:
            logger.error(f"Error mixing sound effects: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to mix sound effects: {str(e)}"
            }
    
    def apply_sound_effects_to_video(
        self,
        video_file: str,
        output_file: str,
        sound_effects: List[Dict[str, Any]],
        keep_original_audio: bool = True,
        master_volume: float = 1.0
    ) -> Dict[str, Any]:
        """
        Apply multiple sound effects to a video.
        
        Args:
            video_file: Path to the input video file
            output_file: Path to the output video file
            sound_effects: List of dictionaries with sound effect details
                Each dictionary should have:
                - file_path: Path to the sound effect file
                - start_time: Start time in seconds
                - intensity: Volume intensity (0.0 to 1.0)
                - position: Optional spatial position
            keep_original_audio: Whether to keep the original video audio
            master_volume: Master volume for the sound effects mix
            
        Returns:
            Dictionary with the result of the operation
        """
        try:
            if not sound_effects:
                return {
                    "status": "error",
                    "error": "No sound effects provided"
                }
            
            # Get video duration
            video_duration = self._get_video_duration(video_file)
            
            # Mix all sound effects into a single audio file
            temp_dir = os.path.join(self.temp_dir, f'video_sfx_{int(time.time())}')
            os.makedirs(temp_dir, exist_ok=True)
            
            mixed_audio = os.path.join(temp_dir, 'mixed_effects.wav')
            
            mix_result = self.mix_multiple_sound_effects(
                sound_files=sound_effects,
                output_file=mixed_audio,
                target_duration=video_duration,
                master_volume=master_volume
            )
            
            if mix_result['status'] != 'success':
                return mix_result
            
            # Build FFmpeg command to combine video with mixed audio
            if keep_original_audio:
                # Mix original audio with sound effects
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', video_file,
                    '-i', mixed_audio,
                    '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first[aout]',
                    '-map', '0:v',
                    '-map', '[aout]',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    output_file
                ]
            else:
                # Replace original audio with sound effects
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', video_file,
                    '-i', mixed_audio,
                    '-map', '0:v',
                    '-map', '1:a',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    output_file
                ]
            
            # Execute FFmpeg command
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Clean up temporary files
            if os.path.exists(mixed_audio):
                os.remove(mixed_audio)
            
            os.rmdir(temp_dir)
            
            return {
                "status": "success",
                "video_file": video_file,
                "output_file": output_file,
                "effects_count": len(sound_effects),
                "keep_original_audio": keep_original_audio,
                "master_volume": master_volume
            }
            
        except Exception as e:
            logger.error(f"Error applying sound effects to video: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to apply sound effects to video: {str(e)}"
            }
    
    def analyze_scene_intensity(
        self,
        video_file: str,
        segment_duration: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze scene dynamics to determine intensity levels over time.
        
        Args:
            video_file: Path to the video file
            segment_duration: Duration of each analysis segment in seconds
            
        Returns:
            Dictionary with intensity analysis results
        """
        try:
            # Get video duration
            video_duration = self._get_video_duration(video_file)
            
            # Create temp file for intensity data
            temp_dir = os.path.join(self.temp_dir, f'intensity_{int(time.time())}')
            os.makedirs(temp_dir, exist_ok=True)
            
            intensity_data = os.path.join(temp_dir, 'intensity.txt')
            
            # Run FFmpeg loudnorm filter to analyze audio
            cmd = [
                'ffmpeg',
                '-y',
                '-i', video_file,
                '-af', 'loudnorm=print_format=json',
                '-f', 'null',
                '-'
            ]
            
            process = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Extract loudness information from stderr
            loudness_info = None
            stderr_lines = process.stderr.splitlines()
            json_start = None
            
            for i, line in enumerate(stderr_lines):
                if '{' in line and '"input_' in line:
                    json_start = i
                    break
            
            if json_start is not None:
                json_str = '\n'.join(stderr_lines[json_start:])
                json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
                try:
                    loudness_info = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Analyze scene changes using FFmpeg
            cmd = [
                'ffmpeg',
                '-i', video_file,
                '-filter:v', 'select=\'gt(scene,0.3)\',showinfo',
                '-f', 'null',
                '-'
            ]
            
            process = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Extract scene change timestamps
            scene_changes = []
            for line in process.stderr.splitlines():
                if 'pts_time' in line:
                    parts = line.split('pts_time:')[1].split()
                    if parts:
                        try:
                            time_val = float(parts[0])
                            scene_changes.append(time_val)
                        except ValueError:
                            pass
            
            # Analyze audio intensity by segments
            segment_count = int(np.ceil(video_duration / segment_duration))
            segments = []
            
            # Extract audio, segment and analyze each segment
            for i in range(segment_count):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, video_duration)
                
                # Skip if segment is too short
                if end_time - start_time < 0.1:
                    continue
                
                segment_file = os.path.join(temp_dir, f'segment_{i}.wav')
                
                # Extract segment
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', video_file,
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-c:a', 'pcm_s16le',
                    '-ar', '44100',
                    segment_file
                ]
                
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Analyze segment loudness
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', segment_file,
                    '-af', 'loudnorm=print_format=json',
                    '-f', 'null',
                    '-'
                ]
                
                process = subprocess.run(
                    cmd, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Extract segment loudness
                segment_loudness = -60  # Default very low value
                
                for line in process.stderr.splitlines():
                    if 'input_i' in line:
                        parts = line.split('input_i:')[1].split()
                        if parts:
                            try:
                                segment_loudness = float(parts[0])
                            except ValueError:
                                pass
                            break
                
                # Calculate normalized intensity (0.0 to 1.0)
                # Typical loudness range: -70 dB (quiet) to 0 dB (loud)
                normalized_intensity = min(1.0, max(0.0, (segment_loudness + 70) / 70))
                
                # Check if there's a scene change in this segment
                segment_scene_changes = [
                    sc for sc in scene_changes 
                    if start_time <= sc < end_time
                ]
                
                # Add segment data
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "intensity": normalized_intensity,
                    "loudness_db": segment_loudness,
                    "scene_changes": segment_scene_changes
                })
                
                # Clean up segment file
                if os.path.exists(segment_file):
                    os.remove(segment_file)
            
            # Clean up temp directory
            os.rmdir(temp_dir)
            
            return {
                "status": "success",
                "video_file": video_file,
                "duration": video_duration,
                "segment_count": len(segments),
                "segment_duration": segment_duration,
                "overall_loudness": loudness_info.get("input_i") if loudness_info else None,
                "scene_changes": scene_changes,
                "intensity_segments": segments
            }
            
        except Exception as e:
            logger.error(f"Error analyzing scene intensity: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to analyze scene intensity: {str(e)}"
            }
    
    def _check_tools(self) -> None:
        """Check if required tools are available."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'], 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Some functionality may be limited.")
    
    def _get_audio_duration(self, file_path: str) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE)
            return float(output.decode('utf-8').strip())
            
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {str(e)}")
            return 0.0
    
    def _get_video_duration(self, file_path: str) -> float:
        """
        Get the duration of a video file in seconds.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE)
            return float(output.decode('utf-8').strip())
            
        except Exception as e:
            logger.warning(f"Failed to get video duration: {str(e)}")
            return 0.0 