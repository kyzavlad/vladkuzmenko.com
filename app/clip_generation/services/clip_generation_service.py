"""
Clip Generation Service

This module integrates various components of the Clip Generation Microservice,
including video processing, silence detection, and other features.
"""

import os
import time
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from app.clip_generation.clip_generator import ClipGenerator
from app.clip_generation.services.audio_analysis import (
    SilenceDetector, SilenceDetectorConfig,
    SilenceProcessor, SilenceProcessorConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClipGenerationService:
    """
    Main service for clip generation, integrating various components.
    
    This service coordinates the clip generation process, including:
    - Extracting clips from source videos
    - Applying silence detection and removal
    - Managing temporary files and output
    - Providing status updates
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        ffmpeg_path: Optional[str] = None,
        enable_silence_detection: bool = True,
        silence_detection_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the clip generation service.
        
        Args:
            output_dir: Directory for output files
            temp_dir: Directory for temporary files
            ffmpeg_path: Path to ffmpeg binary
            enable_silence_detection: Enable silence detection feature
            silence_detection_config: Configuration for silence detection
        """
        # Set up directories
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "clip_generation"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up ffmpeg path
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        
        # Initialize clip generator
        self.clip_generator = ClipGenerator(
            output_dir=str(self.output_dir),
            temp_dir=str(self.temp_dir),
            ffmpeg_path=self.ffmpeg_path
        )
        
        # Silence detection settings
        self.enable_silence_detection = enable_silence_detection
        self.silence_detection_config = silence_detection_config or {}
        
        # Initialize silence processor if enabled
        self.silence_processor = None
        if self.enable_silence_detection:
            self._init_silence_processor()
        
        logger.info(f"Initialized ClipGenerationService")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Temporary directory: {self.temp_dir}")
        logger.info(f"Silence detection: {'Enabled' if self.enable_silence_detection else 'Disabled'}")
    
    def _init_silence_processor(self):
        """Initialize the silence processor with configuration."""
        # Create silence detector config
        detector_config = SilenceDetectorConfig(
            min_silence_duration=self.silence_detection_config.get("min_silence", 0.3),
            max_silence_duration=self.silence_detection_config.get("max_silence", 2.0),
            silence_threshold=self.silence_detection_config.get("threshold", -35.0),
            adaptive_threshold=self.silence_detection_config.get("adaptive_threshold", True),
            enable_filler_detection=self.silence_detection_config.get("detect_fillers", False),
            language=self.silence_detection_config.get("language", "en"),
            visualize=self.silence_detection_config.get("visualize", False),
            device=self.silence_detection_config.get("device", "cpu")
        )
        
        # Create silence processor config
        processor_config = SilenceProcessorConfig(
            output_dir=str(self.temp_dir / "processed"),
            temp_dir=str(self.temp_dir / "silence_processing"),
            ffmpeg_path=self.ffmpeg_path,
            removal_threshold=self.silence_detection_config.get("removal_threshold", 0.5),
            max_segment_gap=self.silence_detection_config.get("max_segment_gap", 0.1),
            speed_up_silence=self.silence_detection_config.get("speed_up", False),
            speed_factor=self.silence_detection_config.get("speed_factor", 2.0),
            parallel_processing=False,  # Disable parallel processing for integration
            preserve_video_quality=True,
            silence_detector_config=detector_config
        )
        
        # Initialize silence processor
        self.silence_processor = SilenceProcessor(processor_config)
        
        logger.info("Silence processor initialized with custom configuration")
    
    def process_clip_task(self, task: Dict[str, Any], callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process a clip generation task.
        
        Args:
            task: Task parameters
            callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with task results
        """
        start_time = time.time()
        task_id = task.get("task_id", str(int(time.time())))
        
        logger.info(f"Processing clip task: {task_id}")
        
        # Extract required parameters
        source_video = task.get("source_video")
        clip_start = task.get("clip_start")
        clip_end = task.get("clip_end")
        output_file = task.get("output_file")
        
        if not source_video or not os.path.exists(source_video):
            error_msg = f"Source video not found: {source_video}"
            logger.error(error_msg)
            return self._create_error_response(task_id, error_msg)
        
        if clip_start is None or clip_end is None:
            error_msg = "Clip start and end times are required"
            logger.error(error_msg)
            return self._create_error_response(task_id, error_msg)
        
        # Generate output path if not provided
        if not output_file:
            video_name = os.path.splitext(os.path.basename(source_video))[0]
            output_file = str(self.output_dir / f"{video_name}_clip_{task_id}.mp4")
        
        # Create status update function
        def update_status(status: str, progress: float, message: str = "", data: Dict[str, Any] = None):
            if callback:
                status_update = {
                    "task_id": task_id,
                    "status": status,
                    "progress": progress,
                    "message": message,
                    "data": data or {},
                    "timestamp": time.time()
                }
                callback(status_update)
        
        # Update status: started
        update_status("started", 0.0, "Starting clip generation process")
        
        try:
            # Step 1: Extract the clip
            update_status("processing", 10.0, "Extracting clip from source video")
            
            # Generate a temporary file for the extracted clip
            temp_clip_path = str(self.temp_dir / f"temp_clip_{task_id}.mp4")
            
            extract_result = self.clip_generator.extract_clip(
                source_video, 
                temp_clip_path, 
                clip_start, 
                clip_end
            )
            
            if not extract_result or not os.path.exists(temp_clip_path):
                error_msg = "Failed to extract clip from source video"
                logger.error(error_msg)
                update_status("error", 100.0, error_msg)
                return self._create_error_response(task_id, error_msg)
            
            # Update status: clip extracted
            update_status("processing", 40.0, "Clip extracted successfully")
            
            # Step 2: Apply silence detection and processing if enabled
            processed_clip_path = temp_clip_path
            
            if self.enable_silence_detection and task.get("remove_silence", False):
                update_status("processing", 50.0, "Applying silence detection")
                
                # Update silence processor config if provided in task
                if "silence_config" in task:
                    self._update_silence_config(task["silence_config"])
                
                # Process the clip to remove silence
                processed_clip_path = str(self.temp_dir / f"processed_clip_{task_id}.mp4")
                
                silence_result = self.silence_processor.process_video(
                    temp_clip_path,
                    processed_clip_path
                )
                
                if not silence_result or not os.path.exists(processed_clip_path):
                    logger.warning("Silence detection failed, using original clip")
                    processed_clip_path = temp_clip_path
                    update_status("processing", 70.0, "Silence detection failed, using original clip")
                else:
                    # Get silence detection stats
                    stats = self.silence_processor.get_stats()
                    
                    update_status(
                        "processing", 
                        70.0, 
                        f"Silence detection completed, reduced by {stats['reduction_percentage']:.1f}%",
                        {"silence_detection": stats}
                    )
            
            # Step 3: Apply any additional effects
            update_status("processing", 80.0, "Applying additional effects")
            
            # TODO: Implement additional effects (e.g., filters, watermarks)
            
            # Step 4: Generate final output
            update_status("processing", 90.0, "Generating final output")
            
            # Copy/move processed clip to final output path
            import shutil
            shutil.copy2(processed_clip_path, output_file)
            
            # Clean up temporary files
            if os.path.exists(temp_clip_path) and temp_clip_path != processed_clip_path:
                os.unlink(temp_clip_path)
            
            if os.path.exists(processed_clip_path) and processed_clip_path != output_file:
                os.unlink(processed_clip_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create success response
            result = {
                "task_id": task_id,
                "status": "completed",
                "output_file": output_file,
                "processing_time": processing_time,
                "source_video": source_video,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "silence_detection": self.enable_silence_detection and task.get("remove_silence", False)
            }
            
            # Add silence detection stats if available
            if self.enable_silence_detection and task.get("remove_silence", False):
                stats = self.silence_processor.get_stats()
                result["silence_stats"] = stats
            
            # Update status: completed
            update_status("completed", 100.0, "Clip generation completed", result)
            
            logger.info(f"Clip generation completed: {output_file}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing clip task: {str(e)}"
            logger.error(error_msg)
            update_status("error", 100.0, error_msg)
            return self._create_error_response(task_id, error_msg)
    
    def _update_silence_config(self, silence_config: Dict[str, Any]):
        """Update silence processor configuration with task-specific settings."""
        if not self.silence_processor:
            self._init_silence_processor()
            return
        
        detector_config = self.silence_processor.config.silence_detector_config
        
        # Update detector config
        if "min_silence" in silence_config:
            detector_config.min_silence_duration = silence_config["min_silence"]
        
        if "max_silence" in silence_config:
            detector_config.max_silence_duration = silence_config["max_silence"]
        
        if "threshold" in silence_config:
            detector_config.silence_threshold = silence_config["threshold"]
        
        if "adaptive_threshold" in silence_config:
            detector_config.adaptive_threshold = silence_config["adaptive_threshold"]
        
        if "detect_fillers" in silence_config:
            detector_config.enable_filler_detection = silence_config["detect_fillers"]
        
        if "language" in silence_config:
            detector_config.language = silence_config["language"]
        
        # Update processor config
        if "removal_threshold" in silence_config:
            self.silence_processor.config.removal_threshold = silence_config["removal_threshold"]
        
        if "max_segment_gap" in silence_config:
            self.silence_processor.config.max_segment_gap = silence_config["max_segment_gap"]
        
        if "speed_up" in silence_config:
            self.silence_processor.config.speed_up_silence = silence_config["speed_up"]
        
        if "speed_factor" in silence_config:
            self.silence_processor.config.speed_factor = silence_config["speed_factor"]
        
        logger.info("Updated silence processor configuration for task")
    
    def _create_error_response(self, task_id: str, error_msg: str) -> Dict[str, Any]:
        """Create an error response for a failed task."""
        return {
            "task_id": task_id,
            "status": "error",
            "error": error_msg,
            "timestamp": time.time()
        }
    
    def batch_process_clips(self, tasks: List[Dict[str, Any]], callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Process multiple clip generation tasks in batch.
        
        Args:
            tasks: List of task parameters
            callback: Optional callback function for progress updates
            
        Returns:
            List of task results
        """
        results = []
        
        for task in tasks:
            result = self.process_clip_task(task, callback)
            results.append(result)
        
        return results 