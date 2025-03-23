import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from ..models.database import Job, JobStatus
from ..database.repositories import JobRepository, JobLogRepository
from ..models.config import AppConfig
from .job_processor import JobProcessor

logger = logging.getLogger(__name__)

class JobQueueManager:
    def __init__(
        self,
        job_repository: JobRepository,
        job_log_repository: JobLogRepository,
        config: AppConfig,
        max_concurrent_jobs: int = 5
    ):
        self.job_repository = job_repository
        self.job_log_repository = job_log_repository
        self.config = config
        self.job_processor = JobProcessor(job_repository, job_log_repository, config)
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the job queue manager."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_queue())
        logger.info("Job queue manager started")

    async def stop(self):
        """Stop the job queue manager."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Cancel all active jobs
        for job_id, task in self.active_jobs.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await self.job_processor.cancel_job(job_id)

        self.active_jobs.clear()
        logger.info("Job queue manager stopped")

    async def _process_queue(self):
        """Process jobs from the queue."""
        while self._running:
            try:
                # Get pending jobs
                jobs = await self.job_repository.get_pending_jobs(
                    limit=self.max_concurrent_jobs - len(self.active_jobs)
                )

                # Process each job
                for job in jobs:
                    if job.job_id not in self.active_jobs:
                        task = asyncio.create_task(
                            self._process_job(job.job_id)
                        )
                        self.active_jobs[job.job_id] = task

                # Clean up completed jobs
                completed_jobs = [
                    job_id for job_id, task in self.active_jobs.items()
                    if task.done()
                ]
                for job_id in completed_jobs:
                    del self.active_jobs[job_id]

                # Wait before checking for new jobs
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error processing job queue: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _process_job(self, job_id: str):
        """Process a single job."""
        try:
            await self.job_processor.process_job(job_id)
        except asyncio.CancelledError:
            logger.info(f"Job {job_id} processing cancelled")
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
        finally:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

    async def submit_job(self, job_data: Dict) -> Job:
        """Submit a new job to the queue."""
        try:
            # Create job
            job = await self.job_processor.create_job(job_data)

            # Log job submission
            await self.job_log_repository.create({
                "job_id": job.id,
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "message": f"Job {job.job_id} submitted to queue",
                "details": {"job_type": job.job_type}
            })

            return job
        except Exception as e:
            logger.error(f"Error submitting job: {str(e)}")
            raise

    async def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get the status of a job."""
        try:
            return await self.job_repository.get_by_id(job_id)
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's running."""
        try:
            if job_id in self.active_jobs:
                task = self.active_jobs[job_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.active_jobs[job_id]

            await self.job_processor.cancel_job(job_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            return False 