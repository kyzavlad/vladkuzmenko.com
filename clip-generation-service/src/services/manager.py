import logging
from typing import Optional
from fastapi import FastAPI
from ..database.repositories import JobRepository, JobLogRepository
from ..database.connection import init_db
from ..models.config import AppConfig
from .job_queue import JobQueueManager
from .websocket import WebSocketManager

logger = logging.getLogger(__name__)

class ServiceManager:
    def __init__(self, app: FastAPI, config: AppConfig):
        self.app = app
        self.config = config
        self.job_queue: Optional[JobQueueManager] = None
        self.websocket_manager: Optional[WebSocketManager] = None
        self._running = False

    async def start(self):
        """Start all services."""
        if self._running:
            return

        try:
            # Initialize database
            init_db(self.config.database)

            # Initialize repositories
            job_repository = JobRepository()
            job_log_repository = JobLogRepository()

            # Initialize job queue
            self.job_queue = JobQueueManager(
                job_repository=job_repository,
                job_log_repository=job_log_repository,
                config=self.config,
                max_concurrent_jobs=self.config.processing.max_concurrent_jobs
            )
            await self.job_queue.start()

            # Initialize WebSocket manager
            self.websocket_manager = WebSocketManager()

            # Add WebSocket endpoint
            @self.app.websocket("/ws/{user_id}")
            async def websocket_endpoint(websocket, user_id: str):
                await self.websocket_manager.handle_client(websocket, user_id)

            self._running = True
            logger.info("All services started successfully")

        except Exception as e:
            logger.error(f"Error starting services: {str(e)}")
            await self.stop()
            raise

    async def stop(self):
        """Stop all services."""
        if not self._running:
            return

        try:
            # Stop job queue
            if self.job_queue:
                await self.job_queue.stop()

            # Clean up WebSocket connections
            if self.websocket_manager:
                # WebSocket connections will be cleaned up automatically
                # when clients disconnect
                pass

            self._running = False
            logger.info("All services stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping services: {str(e)}")
            raise

    async def submit_job(self, job_data: dict) -> dict:
        """Submit a new job."""
        if not self.job_queue:
            raise RuntimeError("Job queue not initialized")

        try:
            job = await self.job_queue.submit_job(job_data)
            return job.dict()
        except Exception as e:
            logger.error(f"Error submitting job: {str(e)}")
            raise

    async def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get job status."""
        if not self.job_queue:
            raise RuntimeError("Job queue not initialized")

        try:
            job = await self.job_queue.get_job_status(job_id)
            return job.dict() if job else None
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        if not self.job_queue:
            raise RuntimeError("Job queue not initialized")

        try:
            return await self.job_queue.cancel_job(job_id)
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            raise

    async def broadcast_job_update(self, job: dict):
        """Broadcast job update to connected clients."""
        if not self.websocket_manager:
            raise RuntimeError("WebSocket manager not initialized")

        try:
            await self.websocket_manager.broadcast_job_update(job)
        except Exception as e:
            logger.error(f"Error broadcasting job update: {str(e)}")
            raise

    async def broadcast_job_completion(self, job: dict):
        """Broadcast job completion to connected clients."""
        if not self.websocket_manager:
            raise RuntimeError("WebSocket manager not initialized")

        try:
            await self.websocket_manager.broadcast_job_completion(job)
        except Exception as e:
            logger.error(f"Error broadcasting job completion: {str(e)}")
            raise

    async def broadcast_job_error(self, job: dict, error: str):
        """Broadcast job error to connected clients."""
        if not self.websocket_manager:
            raise RuntimeError("WebSocket manager not initialized")

        try:
            await self.websocket_manager.broadcast_job_error(job, error)
        except Exception as e:
            logger.error(f"Error broadcasting job error: {str(e)}")
            raise 