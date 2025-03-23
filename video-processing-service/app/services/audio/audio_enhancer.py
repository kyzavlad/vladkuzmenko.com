"""
Audio Enhancer Module

This module provides a comprehensive audio enhancement suite that combines
multiple audio processing techniques to improve audio quality for video content.
"""

import os
import logging
import tempfile
import time
from typing import Optional, Dict, Any, List, Tuple, Union
import asyncio
from pathlib import Path

from app.services.audio.noise_reduction import NoiseReducer
from app.services.audio.voice_enhancement import VoiceEnhancer
from app.services.audio.dynamics_processor import DynamicsProcessor
from app.services.audio.environmental_sound_classifier import EnvironmentalSoundClassifier

logger = logging.getLogger(__name__)

class AudioEnhancer:
    """
    Comprehensive audio enhancement suite for video content.
    
    This class orchestrates various audio processing components to improve
    audio quality, including noise reduction, voice enhancement, dynamics
    processing, de-reverberation, and more.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audio enhancer.
        
        Args:
            config: Configuration options for audio enhancement
        """
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
        
        # Initialize audio processing components
        self.noise_reducer = NoiseReducer(self.config.get('noise_reduction'))
        self.voice_enhancer = VoiceEnhancer(**self.config.get('voice_enhancement', {}))
        self.dynamics_processor = DynamicsProcessor(**self.config.get('dynamics_processing', {}))
        self.sound_classifier = EnvironmentalSoundClassifier(**self.config.get('sound_classification', {}))
        
        # Components to be implemented in future steps
        self.dereverberation = None
        self.deesser = None
        self.voice_isolator = None
        self.audio_normalizer = None
        
        # Initialize temp directory
        os.makedirs(os.path.join(self.temp_dir, "audio_enhanced"), exist_ok=True)
    
    async def enhance_audio(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply audio enhancement processing to an audio file.
        
        Args:
            input_path: Path to the input audio file
            output_path: Path to save the processed audio (if None, auto-generated)
            options: Processing options for each enhancement step
            
        Returns:
            Dictionary with result information including output_path
        """
        options = options or {}
        
        try:
            # Generate output path if not provided
            if output_path is None:
                output_dir = os.path.join(self.temp_dir, "audio_enhanced")
                filename = os.path.basename(input_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
            
            # Process the audio through a pipeline of enhancements
            current_path = input_path
            temp_files = []
            
            # Track processing steps
            processing_steps = []
            start_time = time.time()
            
            # Step 1: Noise Reduction (if enabled)
            if options.get('apply_noise_reduction', True):
                logger.info("Applying noise reduction...")
                
                noise_reduction_options = options.get('noise_reduction', {})
                noise_output = os.path.join(self.temp_dir, "audio_enhanced", 
                                           f"{os.path.basename(input_path)}_noise_reduced.wav")
                
                noise_result = await self.noise_reducer.reduce_noise(
                    audio_path=current_path,
                    output_path=noise_output,
                    noise_profile_id=noise_reduction_options.get('noise_profile_id'),
                    noise_sample=noise_reduction_options.get('noise_sample'),
                    auto_detect=noise_reduction_options.get('auto_detect', True),
                    strength=noise_reduction_options.get('strength')
                )
                
                if "error" in noise_result:
                    logger.error(f"Noise reduction failed: {noise_result['error']}")
                else:
                    current_path = noise_result["output_path"]
                    temp_files.append(current_path)
                    
                    # Record processing step
                    processing_steps.append({
                        "step": "noise_reduction",
                        "input": input_path,
                        "output": current_path,
                        "duration": time.time() - start_time,
                        "details": noise_result
                    })
                    
                    # Update start time for next step
                    start_time = time.time()
            
            # Step 2: Voice Enhancement (if enabled)
            if options.get('apply_voice_enhancement', False):
                logger.info("Applying voice enhancement...")
                
                voice_enhancement_options = options.get('voice_enhancement', {})
                voice_output = os.path.join(self.temp_dir, "audio_enhanced", 
                                           f"{os.path.basename(input_path)}_voice_enhanced.wav")
                
                # Apply voice enhancement
                voice_result = self.voice_enhancer.enhance_voice(
                    audio_path=current_path,
                    output_path=voice_output,
                    apply_eq=voice_enhancement_options.get('apply_eq', True),
                    apply_compression=voice_enhancement_options.get('apply_compression', True),
                    apply_de_essing=voice_enhancement_options.get('apply_de_essing', True),
                    apply_harmonic_enhancement=voice_enhancement_options.get('apply_harmonic_enhancement', True)
                )
                
                if voice_result.get("status") == "error":
                    logger.error(f"Voice enhancement failed: {voice_result.get('error')}")
                else:
                    current_path = voice_result["output_path"]
                    temp_files.append(current_path)
                    
                    # Record processing step
                    processing_steps.append({
                        "step": "voice_enhancement",
                        "input": temp_files[-2] if len(temp_files) >= 2 else input_path,
                        "output": current_path,
                        "duration": time.time() - start_time,
                        "details": voice_result
                    })
                    
                    # Update start time for next step
                    start_time = time.time()
                    
            # Step 3: Dynamics Processing (if enabled)
            if options.get('apply_dynamics_processing', False):
                logger.info("Applying dynamics processing...")
                
                dynamics_options = options.get('dynamics_processing', {})
                dynamics_output = os.path.join(self.temp_dir, "audio_enhanced", 
                                             f"{os.path.basename(input_path)}_dynamics_processed.wav")
                
                # Apply dynamics processing
                dynamics_result = self.dynamics_processor.process_audio(
                    audio_path=current_path,
                    output_path=dynamics_output,
                    apply_compression=dynamics_options.get('apply_compression', True),
                    apply_limiting=dynamics_options.get('apply_limiting', True),
                    apply_expansion=dynamics_options.get('apply_expansion', False),
                    apply_gating=dynamics_options.get('apply_gating', False),
                    target_loudness=dynamics_options.get('target_loudness'),
                    dry_wet_mix=dynamics_options.get('dry_wet_mix', 1.0)
                )
                
                if dynamics_result.get("status") != "success":
                    logger.error(f"Dynamics processing failed: {dynamics_result.get('error')}")
                else:
                    current_path = dynamics_result["output_path"]
                    temp_files.append(current_path)
                    
                    # Record processing step
                    processing_steps.append({
                        "step": "dynamics_processing",
                        "input": temp_files[-2] if len(temp_files) >= 2 else input_path,
                        "output": current_path,
                        "duration": time.time() - start_time,
                        "details": dynamics_result
                    })
                    
                    # Update start time for next step
                    start_time = time.time()
            
            # Final step: Copy the last processed file to the output path (if different)
            if current_path != output_path:
                import shutil
                shutil.copy2(current_path, output_path)
            
            # Clean up temporary files (optional)
            if options.get('cleanup_temp_files', True) and temp_files:
                for temp_file in temp_files:
                    if os.path.exists(temp_file) and temp_file != output_path:
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
            
            # Return result
            return {
                "status": "success",
                "input_path": input_path,
                "output_path": output_path,
                "processing_steps": processing_steps,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {str(e)}")
            return {"error": f"Failed to enhance audio: {str(e)}"}
    
    async def enhance_video_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance the audio track of a video file.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the processed video (if None, auto-generated)
            options: Processing options for each enhancement step
            
        Returns:
            Dictionary with result information including output_path
        """
        options = options or {}
        
        try:
            # Generate output path if not provided
            if output_path is None:
                output_dir = os.path.join(self.temp_dir, "audio_enhanced")
                filename = os.path.basename(video_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
            
            # Extract audio from video
            audio_path = os.path.join(self.temp_dir, "audio_enhanced", 
                                     f"{os.path.basename(video_path)}_audio.wav")
            
            # Run ffmpeg to extract audio
            extract_cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-q:a", "0",
                "-map", "a",
                "-vn",
                audio_path
            ]
            
            extract_process = await asyncio.create_subprocess_exec(
                *extract_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await extract_process.communicate()
            
            if extract_process.returncode != 0:
                logger.error(f"Failed to extract audio: {stderr.decode()}")
                return {"error": "Failed to extract audio from video"}
            
            # Process the extracted audio
            enhanced_audio = os.path.join(self.temp_dir, "audio_enhanced", 
                                         f"{os.path.basename(video_path)}_enhanced_audio.wav")
            
            audio_result = await self.enhance_audio(
                input_path=audio_path,
                output_path=enhanced_audio,
                options=options
            )
            
            if "error" in audio_result:
                return audio_result
            
            # Merge enhanced audio back into video
            merge_cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-i", enhanced_audio,
                "-c:v", "copy",  # Copy video stream without re-encoding
                "-map", "0:v",   # Use video from first input
                "-map", "1:a",   # Use audio from second input
                "-y",            # Overwrite output if exists
                output_path
            ]
            
            merge_process = await asyncio.create_subprocess_exec(
                *merge_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await merge_process.communicate()
            
            if merge_process.returncode != 0:
                logger.error(f"Failed to merge audio with video: {stderr.decode()}")
                return {"error": "Failed to merge enhanced audio with video"}
            
            # Clean up temporary files
            if options.get('cleanup_temp_files', True):
                for temp_file in [audio_path, enhanced_audio]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
            
            # Return result
            return {
                "status": "success",
                "input_path": video_path,
                "output_path": output_path,
                "audio_processing": audio_result,
                "duration": time.time() - audio_result.get('duration', 0)
            }
            
        except Exception as e:
            logger.error(f"Error enhancing video audio: {str(e)}")
            return {"error": f"Failed to enhance video audio: {str(e)}"}
    
    async def analyze_audio(
        self,
        audio_path: str
    ) -> Dict[str, Any]:
        """
        Analyze audio to identify characteristics and potential issues.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Perform noise profile analysis
            noise_profile = await self.noise_reducer.analyze_noise_profile(
                audio_path=audio_path,
                start_time=0.0,
                duration=1.0
            )
            
            # Perform environmental sound classification
            sound_classification = self.sound_classifier.classify_audio(
                audio_path=audio_path,
                return_all_segments=False
            )
            
            # Analyze voice characteristics if available
            voice_analysis = None
            if hasattr(self.voice_enhancer, 'analyze_voice'):
                voice_analysis = self.voice_enhancer.analyze_voice(audio_path)
            
            # Analyze dynamics if available
            dynamics_analysis = None
            if hasattr(self.dynamics_processor, 'analyze_dynamics'):
                dynamics_analysis = self.dynamics_processor.analyze_dynamics(audio_path)
                
            # Combine all analysis results
            return {
                "status": "success",
                "input_path": audio_path,
                "noise_profile": noise_profile,
                "sound_classification": sound_classification,
                "voice_analysis": voice_analysis,
                "dynamics_analysis": dynamics_analysis,
                "analyzed_at": time.time(),
                "recommendations": self._generate_recommendations(
                    noise_profile, 
                    sound_classification,
                    voice_analysis,
                    dynamics_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            return {"error": f"Failed to analyze audio: {str(e)}"}
    
    def _generate_recommendations(
        self,
        noise_profile: Dict[str, Any],
        sound_classification: Dict[str, Any],
        voice_analysis: Optional[Dict[str, Any]],
        dynamics_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            noise_profile: Noise analysis results
            sound_classification: Sound classification results
            voice_analysis: Voice analysis results
            dynamics_analysis: Dynamics analysis results
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "noise_reduction": {
                "apply": False,
                "strength": 0.5,
                "reason": ""
            },
            "voice_enhancement": {
                "apply": False,
                "settings": {},
                "reason": ""
            },
            "dynamics_processing": {
                "apply": False,
                "settings": {},
                "reason": ""
            }
        }
        
        # Check noise profile
        if noise_profile and "snr_db" in noise_profile:
            snr = noise_profile.get("snr_db", 30)
            if snr < 15:
                recommendations["noise_reduction"]["apply"] = True
                recommendations["noise_reduction"]["strength"] = min(1.0, max(0.3, (15 - snr) / 15))
                recommendations["noise_reduction"]["reason"] = f"Low signal-to-noise ratio ({snr:.1f} dB)"
        
        # Check sound classification
        if sound_classification and "dominant_sounds" in sound_classification:
            # Find high probability noise classes
            noise_classes = []
            for sound in sound_classification.get("dominant_sounds", []):
                if sound.get("class") in [
                    "air_conditioner", "engine_idling", "white_noise", 
                    "traffic", "office_noise", "restaurant_chatter"
                ] and sound.get("mean_confidence", 0) > 0.6:
                    noise_classes.append(sound.get("class"))
            
            if noise_classes:
                recommendations["noise_reduction"]["apply"] = True
                recommendations["noise_reduction"]["reason"] += f" Detected noise: {', '.join(noise_classes)}"
                
                # Get noise profile recommendation for the top detected class
                if noise_classes and hasattr(self.sound_classifier, 'get_noise_profile_for_class'):
                    noise_profile_rec = self.sound_classifier.get_noise_profile_for_class(noise_classes[0])
                    # Update noise reduction settings with recommended values
                    if "noise_reduction" in noise_profile_rec:
                        recommendations["noise_reduction"].update(noise_profile_rec["noise_reduction"])
        
        # Check voice analysis
        if voice_analysis and voice_analysis.get("status") == "success":
            # Check if voice enhancement would be beneficial
            if voice_analysis.get("has_sibilance", False) or voice_analysis.get("needs_clarity", False):
                recommendations["voice_enhancement"]["apply"] = True
                recommendations["voice_enhancement"]["settings"] = voice_analysis.get("recommended_settings", {})
                
                reasons = []
                if voice_analysis.get("has_sibilance", False):
                    reasons.append("sibilance detected")
                if voice_analysis.get("needs_clarity", False):
                    reasons.append("clarity improvement needed")
                    
                recommendations["voice_enhancement"]["reason"] = f"Voice quality issues: {', '.join(reasons)}"
        
        # Check dynamics analysis
        if dynamics_analysis and dynamics_analysis.get("status") == "success":
            # Check dynamic range
            dynamic_range = dynamics_analysis.get("levels", {}).get("dynamic_range_db", 0)
            crest_factor = dynamics_analysis.get("levels", {}).get("crest_factor_db", 0)
            suggested_settings = dynamics_analysis.get("suggested_settings", {})
            
            if dynamic_range > 20 or crest_factor > 15 or "preset" in suggested_settings:
                recommendations["dynamics_processing"]["apply"] = True
                
                # Copy any suggested settings
                if suggested_settings:
                    if "compression" in suggested_settings:
                        recommendations["dynamics_processing"]["settings"]["compression"] = suggested_settings["compression"]
                    if "preset" in suggested_settings:
                        recommendations["dynamics_processing"]["settings"]["preset"] = suggested_settings["preset"]
                    if "target_loudness" in suggested_settings:
                        recommendations["dynamics_processing"]["settings"]["target_loudness"] = suggested_settings["target_loudness"]
                
                reasons = []
                if dynamic_range > 20:
                    reasons.append(f"wide dynamic range ({dynamic_range:.1f} dB)")
                if crest_factor > 15:
                    reasons.append(f"high peak-to-average ratio ({crest_factor:.1f} dB)")
                if suggested_settings.get("content_type"):
                    reasons.append(f"content type: {suggested_settings['content_type']}")
                    
                recommendations["dynamics_processing"]["reason"] = f"Dynamic issues: {', '.join(reasons)}"
        
        return recommendations
    
    def get_available_enhancements(self) -> Dict[str, bool]:
        """
        Get a list of available audio enhancement features.
        
        Returns:
            Dictionary with enhancement names and availability status
        """
        return {
            "noise_reduction": self.noise_reducer is not None,
            "voice_enhancement": self.voice_enhancer is not None,
            "dynamics_processing": self.dynamics_processor is not None,
            "dereverberation": self.dereverberation is not None,
            "deessing": self.deesser is not None,
            "sound_classification": self.sound_classifier is not None,
            "voice_isolation": self.voice_isolator is not None,
            "audio_normalization": self.audio_normalizer is not None,
        }
    
    async def batch_process(
        self,
        file_paths: List[str],
        options: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multiple audio or video files in batch.
        
        Args:
            file_paths: List of paths to audio or video files
            options: Processing options for each enhancement step
            output_dir: Directory to save processed files
            
        Returns:
            Dictionary with batch processing results
        """
        options = options or {}
        output_dir = output_dir or os.path.join(self.temp_dir, "audio_enhanced", "batch")
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        failed = []
        
        for file_path in file_paths:
            try:
                # Determine if it's a video or audio file
                is_video = os.path.splitext(file_path)[1].lower() in [
                    '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'
                ]
                
                # Generate output path
                filename = os.path.basename(file_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
                
                # Process the file
                if is_video:
                    result = await self.enhance_video_audio(
                        video_path=file_path,
                        output_path=output_path,
                        options=options
                    )
                else:
                    result = await self.enhance_audio(
                        input_path=file_path,
                        output_path=output_path,
                        options=options
                    )
                
                # Record the result
                if "error" in result:
                    failed.append({
                        "file_path": file_path,
                        "error": result["error"]
                    })
                else:
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                failed.append({
                    "file_path": file_path,
                    "error": str(e)
                })
        
        # Return batch results
        return {
            "status": "completed",
            "successful": results,
            "failed": failed,
            "total": len(file_paths),
            "success_count": len(results),
            "failure_count": len(failed)
        } 