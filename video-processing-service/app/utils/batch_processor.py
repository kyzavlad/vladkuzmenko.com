"""
Batch processor for handling multiple video subtitle generation.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
import json
from pathlib import Path
from datetime import datetime

from app.services.subtitles import SubtitleService

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Batch processor for generating subtitles for multiple videos.
    
    This utility handles batch processing of multiple videos, generating
    subtitles in various formats and optionally burning them into the videos.
    """
    
    def __init__(
        self,
        output_dir: str,
        subtitle_service: Optional[SubtitleService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            output_dir: Directory to save output files
            subtitle_service: SubtitleService instance (created if not provided)
            config: Configuration options
        """
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize subtitle service if not provided
        self.subtitle_service = subtitle_service or SubtitleService(config=self.config.get("subtitle_service"))
        
        # Dictionary to track job status
        self.jobs = {}
        
    async def process_batch(
        self,
        video_paths: List[str],
        transcripts: List[Dict[str, Any]],
        job_id: Optional[str] = None,
        subtitle_formats: List[str] = ["srt", "vtt"],
        generate_video: bool = True,
        batch_options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> str:
        """
        Process a batch of videos with their corresponding transcripts.
        
        Args:
            video_paths: List of paths to input videos
            transcripts: List of transcripts, one for each video
            job_id: Optional job ID (generated if not provided)
            subtitle_formats: List of subtitle formats to generate
            generate_video: Whether to generate videos with burnt-in subtitles
            batch_options: Additional options for batch processing
            progress_callback: Optional callback for progress reporting
            
        Returns:
            Job ID
        """
        # Generate job ID if not provided
        if not job_id:
            job_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(video_paths)}_videos"
        
        # Create job directory
        job_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Initialize job status
        self.jobs[job_id] = {
            "status": "running",
            "total_videos": len(video_paths),
            "completed_videos": 0,
            "failed_videos": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "results": []
        }
        
        # Get batch options
        options = batch_options or {}
        
        # Start processing in the background
        asyncio.create_task(
            self._process_batch_async(
                job_id=job_id,
                job_dir=job_dir,
                video_paths=video_paths,
                transcripts=transcripts,
                subtitle_formats=subtitle_formats,
                generate_video=generate_video,
                options=options,
                progress_callback=progress_callback
            )
        )
        
        return job_id
    
    async def _process_batch_async(
        self,
        job_id: str,
        job_dir: str,
        video_paths: List[str],
        transcripts: List[Dict[str, Any]],
        subtitle_formats: List[str],
        generate_video: bool,
        options: Dict[str, Any],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Process the batch asynchronously.
        
        Args:
            job_id: Job ID
            job_dir: Directory for job outputs
            video_paths: List of paths to input videos
            transcripts: List of transcripts
            subtitle_formats: List of subtitle formats
            generate_video: Whether to generate videos
            options: Additional options
            progress_callback: Progress callback
        """
        results = []
        
        # Get options
        video_quality = options.get("video_quality", "medium")
        optimize_positioning = options.get("optimize_positioning", False)
        style_name = options.get("style_name")
        custom_style = options.get("custom_style")
        reading_speed_preset = options.get("reading_speed_preset")
        detect_emphasis = options.get("detect_emphasis", False)
        language = options.get("language")
        auto_detect_language = options.get("auto_detect_language", False)
        background_blur = options.get("background_blur", 0.0)
        
        # Process each video
        for i, (video_path, transcript) in enumerate(zip(video_paths, transcripts)):
            # Update status
            self.jobs[job_id]["current_video"] = os.path.basename(video_path)
            self.jobs[job_id]["current_index"] = i
            
            # Get base filename
            video_basename = os.path.basename(video_path)
            base_filename = os.path.splitext(video_basename)[0]
            
            # Create video-specific output directory
            video_output_dir = os.path.join(job_dir, base_filename)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Report progress
            if progress_callback:
                progress = i / len(video_paths)
                progress_callback(job_id, progress)
            
            try:
                # Process the video
                logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
                
                result_files = await self.subtitle_service.generate_multiple_outputs(
                    video_path=video_path,
                    transcript=transcript,
                    output_dir=video_output_dir,
                    base_filename=base_filename,
                    subtitle_formats=subtitle_formats,
                    generate_video=generate_video,
                    video_quality=video_quality,
                    optimize_positioning=optimize_positioning,
                    style_name=style_name,
                    custom_style=custom_style,
                    reading_speed_preset=reading_speed_preset,
                    detect_emphasis=detect_emphasis,
                    language=language,
                    auto_detect_language=auto_detect_language,
                    background_blur=background_blur
                )
                
                # Record success
                self.jobs[job_id]["completed_videos"] += 1
                results.append({
                    "video_path": video_path,
                    "output_dir": video_output_dir,
                    "files": result_files,
                    "status": "success"
                })
                
            except Exception as e:
                # Record failure
                logger.error(f"Error processing video {video_path}: {str(e)}")
                self.jobs[job_id]["failed_videos"] += 1
                results.append({
                    "video_path": video_path,
                    "output_dir": video_output_dir,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Update job status
        self.jobs[job_id]["status"] = "completed"
        self.jobs[job_id]["end_time"] = datetime.now().isoformat()
        self.jobs[job_id]["results"] = results
        
        # Generate batch summary report
        batch_summary = {
            "job_id": job_id,
            "status": "completed",
            "total_videos": len(video_paths),
            "completed_videos": self.jobs[job_id]["completed_videos"],
            "failed_videos": self.jobs[job_id]["failed_videos"],
            "start_time": self.jobs[job_id]["start_time"],
            "end_time": self.jobs[job_id]["end_time"],
            "results": results,
            "options": {
                "subtitle_formats": subtitle_formats,
                "generate_video": generate_video,
                "video_quality": video_quality,
                "optimize_positioning": optimize_positioning,
                "style_name": style_name,
                "reading_speed_preset": reading_speed_preset,
                "detect_emphasis": detect_emphasis,
                "language": language,
                "auto_detect_language": auto_detect_language
            }
        }
        
        # Save batch summary
        summary_path = os.path.join(job_dir, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        # Report final progress
        if progress_callback:
            progress_callback(job_id, 1.0)
        
        logger.info(f"Batch job {job_id} completed: {self.jobs[job_id]['completed_videos']} succeeded, {self.jobs[job_id]['failed_videos']} failed")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job not found: {job_id}")
        
        return self.jobs[job_id]
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all batch jobs.
        
        Returns:
            List of job summaries
        """
        return [
            {
                "job_id": job_id,
                "status": job_info["status"],
                "total_videos": job_info["total_videos"],
                "completed_videos": job_info["completed_videos"],
                "failed_videos": job_info["failed_videos"],
                "start_time": job_info["start_time"],
                "end_time": job_info["end_time"]
            }
            for job_id, job_info in self.jobs.items()
        ] 