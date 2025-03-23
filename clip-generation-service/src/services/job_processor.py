from typing import Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
from ..models.database import Job, JobLog, JobStatus, JobType
from ..database.repositories import JobRepository, JobLogRepository
from ..models.job import JobCreate, JobUpdate
from ..models.config import AppConfig
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)

class JobProcessor:
    def __init__(
        self,
        job_repository: JobRepository,
        job_log_repository: JobLogRepository,
        config: AppConfig
    ):
        self.job_repository = job_repository
        self.job_log_repository = job_log_repository
        self.config = config
        self.video_processor = VideoProcessor(config)

    async def create_job(self, job_data: JobCreate) -> Job:
        """Create a new job."""
        try:
            # Create job in database
            job = await self.job_repository.create({
                "job_id": job_data.job_id,
                "user_id": job_data.user_id,
                "job_type": job_data.job_type,
                "status": JobStatus.PENDING,
                "progress": 0.0,
                "input_data": job_data.input_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })

            # Log job creation
            await self.job_log_repository.create({
                "job_id": job.id,
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "message": f"Job {job.job_id} created",
                "details": {"job_type": job.job_type}
            })

            return job
        except Exception as e:
            logger.error(f"Error creating job: {str(e)}")
            raise

    async def process_job(self, job_id: str) -> Job:
        """Process a job based on its type."""
        try:
            # Get job from database
            job = await self.job_repository.get_by_id(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")

            # Update job status to processing
            job = await self.job_repository.update_status(
                job_id=job.job_id,
                status=JobStatus.PROCESSING,
                progress=0.0
            )

            # Log job start
            await self.job_log_repository.create({
                "job_id": job.id,
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "message": f"Job {job.job_id} processing started",
                "details": {"job_type": job.job_type}
            })

            # Process job based on type
            if job.job_type == JobType.VIDEO_EDIT:
                await self._process_video_edit(job)
            elif job.job_type == JobType.AVATAR_CREATE:
                await self._process_avatar_create(job)
            elif job.job_type == JobType.AVATAR_GENERATE:
                await self._process_avatar_generate(job)
            elif job.job_type == JobType.VIDEO_TRANSLATE:
                await self._process_video_translate(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

            # Update job status to completed
            job = await self.job_repository.update_status(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                progress=100.0
            )

            # Log job completion
            await self.job_log_repository.create({
                "job_id": job.id,
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "message": f"Job {job.job_id} completed successfully",
                "details": {"job_type": job.job_type}
            })

            return job
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            
            # Update job status to failed
            if job:
                await self.job_repository.update_status(
                    job_id=job.job_id,
                    status=JobStatus.FAILED,
                    error_message=str(e)
                )

                # Log job failure
                await self.job_log_repository.create({
                    "job_id": job.id,
                    "timestamp": datetime.utcnow(),
                    "level": "ERROR",
                    "message": f"Job {job.job_id} failed",
                    "details": {"error": str(e)}
                })
            
            raise

    async def _process_video_edit(self, job: Job) -> None:
        """Process video edit job."""
        try:
            # Get input parameters
            input_data = job.input_data
            input_path = input_data["input_path"]
            target_duration = input_data.get("target_duration", 30.0)
            target_width = input_data.get("target_width", 1080)
            target_height = input_data.get("target_height", 1920)
            target_lufs = input_data.get("target_lufs", -14.0)

            # Generate output path
            output_filename = f"edited_{Path(input_path).stem}.mp4"
            output_path = str(Path(self.config.storage.output_dir) / output_filename)

            # Process video
            output_path, duration = await self.video_processor.process_video_edit(
                input_path=input_path,
                output_path=output_path,
                target_duration=target_duration,
                target_width=target_width,
                target_height=target_height,
                target_lufs=target_lufs
            )

            # Update job with output data
            await self.job_repository.update(
                job_id=job.job_id,
                output_data={
                    "output_path": output_path,
                    "duration": duration
                }
            )

        except Exception as e:
            logger.error(f"Error processing video edit job: {str(e)}")
            raise

    async def _process_avatar_create(self, job: Job) -> None:
        """Process avatar creation job."""
        try:
            # Get input parameters
            input_data = job.input_data
            input_path = input_data["input_path"]
            avatar_type = input_data.get("avatar_type", "realistic")
            style = input_data.get("style")

            # Generate output path
            output_filename = f"avatar_{Path(input_path).stem}.glb"
            output_path = str(Path(self.config.storage.output_dir) / output_filename)

            # Process avatar creation
            output_path, duration = await self.video_processor.process_avatar_create(
                input_path=input_path,
                output_path=output_path,
                avatar_type=avatar_type,
                style=style
            )

            # Update job with output data
            await self.job_repository.update(
                job_id=job.job_id,
                output_data={
                    "output_path": output_path,
                    "duration": duration
                }
            )

        except Exception as e:
            logger.error(f"Error processing avatar creation job: {str(e)}")
            raise

    async def _process_avatar_generate(self, job: Job) -> None:
        """Process avatar generation job."""
        try:
            # Get input parameters
            input_data = job.input_data
            avatar_id = input_data["avatar_id"]
            script = input_data["script"]
            voice_id = input_data.get("voice_id")

            # Generate output path
            output_filename = f"generated_{avatar_id}.mp4"
            output_path = str(Path(self.config.storage.output_dir) / output_filename)

            # Process avatar generation
            output_path, duration = await self.video_processor.process_avatar_generate(
                avatar_id=avatar_id,
                output_path=output_path,
                script=script,
                voice_id=voice_id
            )

            # Update job with output data
            await self.job_repository.update(
                job_id=job.job_id,
                output_data={
                    "output_path": output_path,
                    "duration": duration
                }
            )

        except Exception as e:
            logger.error(f"Error processing avatar generation job: {str(e)}")
            raise

    async def _process_video_translate(self, job: Job) -> None:
        """Process video translation job."""
        try:
            # Get input parameters
            input_data = job.input_data
            input_path = input_data["input_path"]
            target_language = input_data["target_language"]
            voice_id = input_data.get("voice_id")

            # Generate output path
            output_filename = f"translated_{Path(input_path).stem}.mp4"
            output_path = str(Path(self.config.storage.output_dir) / output_filename)

            # Process video translation
            output_path, duration = await self.video_processor.process_video_translate(
                input_path=input_path,
                output_path=output_path,
                target_language=target_language,
                voice_id=voice_id
            )

            # Update job with output data
            await self.job_repository.update(
                job_id=job.job_id,
                output_data={
                    "output_path": output_path,
                    "duration": duration
                }
            )

        except Exception as e:
            logger.error(f"Error processing video translation job: {str(e)}")
            raise

    async def update_job_progress(self, job_id: str, progress: float) -> Job:
        """Update job progress."""
        try:
            job = await self.job_repository.update_status(
                job_id=job_id,
                progress=progress
            )

            # Log progress update
            await self.job_log_repository.create({
                "job_id": job.id,
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "message": f"Job {job.job_id} progress updated",
                "details": {"progress": progress}
            })

            return job
        except Exception as e:
            logger.error(f"Error updating job progress: {str(e)}")
            raise

    async def cancel_job(self, job_id: str) -> Job:
        """Cancel a job."""
        try:
            job = await self.job_repository.update_status(
                job_id=job_id,
                status=JobStatus.CANCELLED
            )

            # Log job cancellation
            await self.job_log_repository.create({
                "job_id": job.id,
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "message": f"Job {job.job_id} cancelled",
                "details": {"job_type": job.job_type}
            })

            return job
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            raise 