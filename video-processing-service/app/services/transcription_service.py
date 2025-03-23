import logging
import httpx
import json
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.core.config import settings
from app.models.video import Video, ProcessingTask, ProcessingStatus
from app.services.video_service import VideoService

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.transcription_url = settings.TRANSCRIPTION_SERVICE_URL
        self.service_api_key = settings.SERVICE_API_KEY
        self.video_service = VideoService(db)
    
    async def request_transcription(self, video_id: str, user_id: str) -> Dict[str, Any]:
        """
        Request transcription for a video from the Transcription Service.
        """
        # Get video
        video = await self.video_service.get_video(video_id)
        if not video:
            logger.error(f"Video not found: {video_id}")
            raise ValueError(f"Video not found: {video_id}")
        
        # Create or update transcription task
        task = await self._get_or_create_task(video_id, "transcription")
        
        try:
            # Update task status
            task.status = ProcessingStatus.PROCESSING.value
            task.started_at = datetime.utcnow()
            await self.db.commit()
            await self.db.refresh(task)
            
            # Update video status
            video.status = ProcessingStatus.TRANSCRIBING.value
            await self.db.commit()
            await self.db.refresh(video)
            
            # Prepare request data
            callback_url = f"{settings.VIDEO_PROCESSING_SERVICE_URL}/api/v1/transcriptions/callback"
            
            request_data = {
                "video_id": video_id,
                "user_id": user_id,
                "callback_url": callback_url,
                "video_url": await self._get_video_download_url(video),
                "options": {
                    "generate_subtitles": video.subtitles_enabled
                }
            }
            
            # Send request to Transcription Service
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.transcription_url}/api/v1/transcriptions",
                    json=request_data,
                    headers={"Authorization": f"Bearer {self.service_api_key}"}
                )
                
                if response.status_code != 200 and response.status_code != 201:
                    logger.error(f"Transcription service error: {response.status_code} - {response.text}")
                    raise Exception(f"Transcription service error: {response.status_code}")
                
                result = response.json()
                
                # Update video with transcription ID
                video.transcription_id = result.get("transcription_id")
                await self.db.commit()
                
                return result
        except Exception as e:
            logger.error(f"Error requesting transcription: {str(e)}")
            
            # Update task status to failed
            task.status = ProcessingStatus.FAILED.value
            task.error_message = str(e)
            await self.db.commit()
            
            # If this was the only active task, update video status
            if not await self._has_active_tasks(video_id, exclude_task_id=task.id):
                video.status = ProcessingStatus.FAILED.value
                video.error_message = f"Transcription failed: {str(e)}"
                await self.db.commit()
            
            raise
    
    async def get_transcription(self, transcription_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transcription details from the Transcription Service.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.transcription_url}/api/v1/transcriptions/{transcription_id}",
                    headers={"Authorization": f"Bearer {self.service_api_key}"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Error fetching transcription: {response.status_code} - {response.text}")
                    return None
                
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching transcription: {str(e)}")
            return None
    
    async def update_transcription_status(
        self,
        transcription_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update the status of a transcription task based on callback from Transcription Service.
        """
        # Find video by transcription_id
        stmt = (
            self.db.query(Video)
            .filter(Video.transcription_id == transcription_id)
            .first()
        )
        video = await self.db.execute(stmt)
        video = video.scalar_one_or_none()
        
        if not video:
            logger.error(f"Video with transcription_id {transcription_id} not found")
            return
        
        # Find associated task
        stmt = (
            self.db.query(ProcessingTask)
            .filter(
                ProcessingTask.video_id == video.id,
                ProcessingTask.task_type == "transcription"
            )
            .first()
        )
        task = await self.db.execute(stmt)
        task = task.scalar_one_or_none()
        
        if not task:
            logger.error(f"Transcription task for video {video.id} not found")
            return
        
        # Update task status
        if status.lower() == "completed":
            task.status = ProcessingStatus.COMPLETED.value
            task.completed_at = datetime.utcnow()
            task.progress = 100.0
            task.result = result
            
            # If subtitles were generated, update video record
            if result and "subtitle_url" in result:
                video.subtitle_path = result["subtitle_url"]
            
            # If there are no other active tasks, update video status
            if not await self._has_active_tasks(video.id, exclude_task_id=task.id):
                video.status = ProcessingStatus.COMPLETED.value
                video.processed_at = datetime.utcnow()
        
        elif status.lower() == "failed":
            task.status = ProcessingStatus.FAILED.value
            task.error_message = error or "Transcription failed"
            
            # If there are no other active tasks, update video status
            if not await self._has_active_tasks(video.id, exclude_task_id=task.id):
                video.status = ProcessingStatus.FAILED.value
                video.error_message = error or "Transcription failed"
        
        elif status.lower() == "processing":
            task.status = ProcessingStatus.PROCESSING.value
            if result and "progress" in result:
                task.progress = result["progress"]
        
        # Commit changes
        await self.db.commit()
    
    async def _get_or_create_task(self, video_id: str, task_type: str) -> ProcessingTask:
        """
        Get existing task or create a new one.
        """
        stmt = (
            self.db.query(ProcessingTask)
            .filter(
                ProcessingTask.video_id == video_id,
                ProcessingTask.task_type == task_type
            )
            .first()
        )
        task = await self.db.execute(stmt)
        task = task.scalar_one_or_none()
        
        if task:
            return task
        
        # Create new task
        task = ProcessingTask(
            video_id=video_id,
            task_type=task_type,
            status=ProcessingStatus.PENDING.value
        )
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        
        return task
    
    async def _has_active_tasks(self, video_id: str, exclude_task_id: Optional[str] = None) -> bool:
        """
        Check if video has any active tasks (pending or processing).
        """
        query = (
            self.db.query(ProcessingTask)
            .filter(
                ProcessingTask.video_id == video_id,
                ProcessingTask.status.in_([
                    ProcessingStatus.PENDING.value,
                    ProcessingStatus.PROCESSING.value
                ])
            )
        )
        
        if exclude_task_id:
            query = query.filter(ProcessingTask.id != exclude_task_id)
        
        result = await self.db.execute(query)
        return bool(result.first())
    
    async def _get_video_download_url(self, video: Video) -> str:
        """
        Get a pre-signed URL for downloading the video for processing.
        """
        from app.services.storage_service import StorageService
        
        storage = StorageService()
        return await storage.get_presigned_url(video.storage_path, expires_in=3600) 