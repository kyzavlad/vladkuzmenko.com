import uuid
import time
from typing import Dict, Optional, List
from datetime import datetime
import os
from .models import JobStatus, ClipStatus, Clip, Platform, CaptionStyle
from .clip_generator import ClipGenerator
from .export_service import ExportService

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.export_service = ExportService()
        
    def create_job(self, video_path: str, request_data: dict) -> str:
        """
        Create a new job.
        
        Args:
            video_path (str): Path to the input video file
            request_data (dict): Job parameters
            
        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": JobStatus(
                status=ClipStatus.PROCESSING,
                progress=0,
                clips_generated=0,
                clips=[]
            ),
            "video_path": video_path,
            "request_data": request_data,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "type": "generation" if video_path else "export"
        }
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get the status of a job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            Optional[JobStatus]: Job status if found, None otherwise
        """
        if job_id not in self.jobs:
            return None
        return self.jobs[job_id]["status"]
    
    def update_job_progress(self, job_id: str, progress: float, clips: Optional[list] = None):
        """
        Update job progress and generated clips.
        
        Args:
            job_id (str): Job ID
            progress (float): Progress percentage (0-100)
            clips (Optional[list]): List of generated clips
        """
        if job_id in self.jobs:
            self.jobs[job_id]["status"].progress = progress
            if clips is not None:
                self.jobs[job_id]["status"].clips = clips
                self.jobs[job_id]["status"].clips_generated = len(clips)
            self.jobs[job_id]["updated_at"] = datetime.now()
    
    def complete_job(self, job_id: str, clips: list):
        """
        Mark a job as completed.
        
        Args:
            job_id (str): Job ID
            clips (list): List of generated clips
        """
        if job_id in self.jobs:
            self.jobs[job_id]["status"].status = ClipStatus.COMPLETED
            self.jobs[job_id]["status"].progress = 100
            self.jobs[job_id]["status"].clips = clips
            self.jobs[job_id]["status"].clips_generated = len(clips)
            self.jobs[job_id]["updated_at"] = datetime.now()
    
    def fail_job(self, job_id: str, error_message: str):
        """
        Mark a job as failed.
        
        Args:
            job_id (str): Job ID
            error_message (str): Error message
        """
        if job_id in self.jobs:
            self.jobs[job_id]["status"].status = ClipStatus.FAILED
            self.jobs[job_id]["status"].error_message = error_message
            self.jobs[job_id]["updated_at"] = datetime.now()
    
    def process_job(self, job_id: str):
        """
        Process a job.
        
        Args:
            job_id (str): Job ID
        """
        if job_id not in self.jobs:
            return
            
        job = self.jobs[job_id]
        
        try:
            if job["type"] == "generation":
                self._process_generation_job(job_id)
            else:
                self._process_export_job(job_id)
                
        except Exception as e:
            self.fail_job(job_id, str(e))
    
    def _process_generation_job(self, job_id: str):
        """Process a clip generation job."""
        job = self.jobs[job_id]
        video_path = job["video_path"]
        request_data = job["request_data"]
        
        # Initialize clip generator
        generator = ClipGenerator(video_path)
        
        # Generate clips based on request parameters
        clips = []
        total_duration = generator.video.duration
        current_time = 0
        
        while current_time < total_duration and len(clips) < request_data["max_clips"]:
            # Calculate target duration for this clip
            target_duration = min(
                request_data["max_clip_duration"],
                total_duration - current_time
            )
            
            # Process clip
            output_path, duration = generator.process_clip(target_duration)
            
            # Create clip object
            clip = Clip(
                clip_id=str(uuid.uuid4()),
                start_time=current_time,
                end_time=current_time + duration,
                duration=duration,
                preview_url=f"/preview/{os.path.basename(output_path)}",
                download_url=f"/download/{os.path.basename(output_path)}",
                thumbnail_url=f"/thumbnail/{os.path.basename(output_path)}",
                engagement_score=0.0  # TODO: Implement engagement scoring
            )
            
            clips.append(clip)
            current_time += duration
            
            # Update progress
            progress = min(100, (current_time / total_duration) * 100)
            self.update_job_progress(job_id, progress, clips)
        
        # Mark job as completed
        self.complete_job(job_id, clips)
    
    def _process_export_job(self, job_id: str):
        """Process a batch export job."""
        job = self.jobs[job_id]
        request_data = job["request_data"]
        
        try:
            # Extract parameters
            clips = [Clip(**clip_data) for clip_data in request_data["clips"]]
            platform = Platform(request_data["platform"])
            include_captions = request_data["include_captions"]
            caption_style = CaptionStyle(request_data["caption_style"])
            add_watermark = request_data["add_watermark"]
            watermark_path = request_data["watermark_image"]
            
            # Create output directory
            output_dir = os.path.join("output", job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process clips
            total_clips = len(clips)
            exported_paths = []
            
            for i, clip in enumerate(clips):
                try:
                    path = self.export_service.export_clip(
                        clip,
                        platform,
                        output_dir,
                        include_captions,
                        caption_style,
                        add_watermark,
                        watermark_path
                    )
                    exported_paths.append(path)
                    
                    # Update progress
                    progress = ((i + 1) / total_clips) * 100
                    self.update_job_progress(job_id, progress)
                    
                except Exception as e:
                    print(f"Error exporting clip {clip.clip_id}: {str(e)}")
                    continue
            
            # Create result clips
            result_clips = []
            for clip, path in zip(clips, exported_paths):
                result_clip = clip.copy()
                result_clip.download_url = f"/download/{os.path.basename(path)}"
                result_clips.append(result_clip)
            
            # Mark job as completed
            self.complete_job(job_id, result_clips)
            
        except Exception as e:
            self.fail_job(job_id, str(e)) 