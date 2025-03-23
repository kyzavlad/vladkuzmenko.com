import logging
import os
import uuid
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime
import asyncio
from fastapi import UploadFile
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.video import Video, ProcessingTask, ProcessingStatus
from app.schemas.video import VideoCreate, ProcessingOptions
from app.services.storage_service import StorageService
from app.core.config import settings

logger = logging.getLogger(__name__)


class VideoService:
    """
    Service for managing video entities and processing operations.
    Coordinates between database, storage, and processing tasks.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.storage = StorageService()
    
    async def create_video(self, video_data: VideoCreate, storage_path: str) -> Video:
        """
        Create a new video record in the database.
        
        Args:
            video_data: Video data to create
            storage_path: Path where the video will be stored
            
        Returns:
            Created video object
        """
        # Create video object
        video = Video(
            user_id=video_data.user_id,
            title=video_data.title,
            description=video_data.description,
            original_filename=video_data.original_filename,
            storage_path=storage_path,
            content_type=video_data.content_type,
            file_size=video_data.file_size,
            status=ProcessingStatus.PENDING.value,
            transcription_enabled=video_data.transcription_enabled,
            subtitles_enabled=video_data.subtitles_enabled,
            enhancement_enabled=video_data.enhancement_enabled,
            avatar_enabled=video_data.avatar_enabled,
        )
        
        # Save to database
        self.db.add(video)
        await self.db.commit()
        await self.db.refresh(video)
        
        logger.info(f"Created video record: {video.id}")
        return video
    
    async def get_video(self, video_id: str) -> Optional[Video]:
        """
        Get a video by ID, including all related tasks.
        
        Args:
            video_id: ID of the video to get
            
        Returns:
            Video object if found, None otherwise
        """
        stmt = select(Video).where(Video.id == video_id)
        result = await self.db.execute(stmt)
        video = result.scalars().first()
        
        return video
    
    async def get_videos_by_user(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 10,
        status: Optional[str] = None
    ) -> Tuple[List[Video], int]:
        """
        Get a paginated list of videos for a user.
        
        Args:
            user_id: ID of the user to get videos for
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            status: Optional filter by video status
            
        Returns:
            Tuple of (list of videos, total count)
        """
        # Construct base query
        query = select(Video).where(Video.user_id == user_id)
        
        # Add status filter if provided
        if status:
            query = query.where(Video.status == status)
        
        # Order by creation date, newest first
        query = query.order_by(Video.created_at.desc())
        
        # Execute count query
        count_query = select(func.count()).select_from(query.subquery())
        result = await self.db.execute(count_query)
        total = result.scalar()
        
        # Execute paginated query
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        videos = result.scalars().all()
        
        return videos, total
    
    async def update_video_status(
        self, 
        video_id: str, 
        status: str, 
        progress: float = None,
        error_message: str = None
    ) -> Optional[Video]:
        """
        Update the status of a video.
        
        Args:
            video_id: ID of the video to update
            status: New status value
            progress: Optional progress value (0-100)
            error_message: Optional error message if status is failed
            
        Returns:
            Updated video object if found, None otherwise
        """
        # Get video
        video = await self.get_video(video_id)
        if not video:
            logger.warning(f"Video not found for status update: {video_id}")
            return None
        
        # Update status
        video.status = status
        if progress is not None:
            video.progress = progress
        if error_message:
            video.error_message = error_message
        
        # If completed, set processed_at timestamp
        if status == ProcessingStatus.COMPLETED.value:
            video.processed_at = datetime.utcnow()
        
        # Save to database
        await self.db.commit()
        await self.db.refresh(video)
        
        logger.info(f"Updated video status: {video_id} -> {status}")
        return video
    
    async def delete_video(self, video_id: str) -> bool:
        """
        Delete a video and all associated files and tasks.
        
        Args:
            video_id: ID of the video to delete
            
        Returns:
            True if video was deleted, False otherwise
        """
        # Get video
        video = await self.get_video(video_id)
        if not video:
            logger.warning(f"Video not found for deletion: {video_id}")
            return False
        
        try:
            # Delete stored files
            storage_paths = [video.storage_path]
            
            # Add additional paths if they exist
            if video.enhanced_video_path:
                storage_paths.append(video.enhanced_video_path)
            if video.avatar_video_path:
                storage_paths.append(video.avatar_video_path)
            if video.subtitle_path:
                storage_paths.append(video.subtitle_path)
            
            # Delete all files
            for path in storage_paths:
                try:
                    await self.storage.delete_file(path)
                except Exception as e:
                    logger.warning(f"Error deleting file {path}: {str(e)}")
            
            # Delete from database
            await self.db.delete(video)
            await self.db.commit()
            
            logger.info(f"Deleted video: {video_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting video {video_id}: {str(e)}")
            await self.db.rollback()
            return False
    
    async def process_uploaded_video(
        self, 
        video_id: str, 
        file: UploadFile, 
        options: ProcessingOptions
    ) -> None:
        """
        Process an uploaded video. This is intended to be run as a background task.
        
        Args:
            video_id: ID of the video to process
            file: Uploaded file object
            options: Processing options
        """
        logger.info(f"Starting processing for video {video_id}")
        
        # Get video
        video = await self.get_video(video_id)
        if not video:
            logger.error(f"Video not found for processing: {video_id}")
            return
        
        try:
            # Update status to uploading
            await self.update_video_status(
                video_id, 
                ProcessingStatus.UPLOADING.value,
                progress=10
            )
            
            # Save file to storage
            await self.storage.save_upload(file, video.storage_path)
            
            # Update status to processing
            await self.update_video_status(
                video_id, 
                ProcessingStatus.PROCESSING.value,
                progress=20
            )
            
            # Extract basic video metadata if possible
            await self._extract_video_metadata(video)
            
            # Process based on enabled options
            processing_tasks = []
            
            # Create processing tasks based on options
            if options.transcription or options.subtitles:
                # Add transcription task - will be handled by transcription_service.request_transcription
                # which gets called separately
                await self._create_task(video_id, "transcription")
            
            if options.enhancement:
                # Add enhancement task
                enhance_task = await self._create_task(video_id, "enhancement")
                processing_tasks.append(
                    self._process_enhancement(video_id, enhance_task.id)
                )
            
            if options.avatar:
                # Add avatar task
                avatar_task = await self._create_task(video_id, "avatar")
                processing_tasks.append(
                    self._process_avatar(video_id, avatar_task.id)
                )
            
            # If no additional processing tasks (beyond transcription), mark as completed
            if not processing_tasks and not (options.transcription or options.subtitles):
                await self.update_video_status(
                    video_id,
                    ProcessingStatus.COMPLETED.value,
                    progress=100
                )
                logger.info(f"Video {video_id} processing completed - no processing tasks")
                return
            
            # Execute any other processing tasks in parallel
            if processing_tasks:
                await asyncio.gather(*processing_tasks)
                
            # Check if all tasks are complete
            await self._update_overall_status(video_id)
            
            logger.info(f"Video {video_id} processing workflow completed")
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")
            await self.update_video_status(
                video_id,
                ProcessingStatus.FAILED.value,
                error_message=f"Processing error: {str(e)}"
            )
    
    async def _extract_video_metadata(self, video: Video) -> None:
        """
        Extract metadata from the video file, like duration, resolution, etc.
        
        Args:
            video: Video object to extract metadata for
        """
        # This would typically use FFmpeg to extract metadata
        # For now, we'll just update with a placeholder duration
        # In a real implementation, we would call ffprobe to get the actual metadata
        
        try:
            # Update video with placeholder metadata
            video.duration = 60.0  # Placeholder: 1 minute
            video.width = 1920
            video.height = 1080
            video.fps = 30.0
            
            # Save to database
            await self.db.commit()
            logger.info(f"Updated video metadata for {video.id}")
        except Exception as e:
            logger.error(f"Error extracting metadata for video {video.id}: {str(e)}")
    
    async def _create_task(self, video_id: str, task_type: str) -> ProcessingTask:
        """
        Create a processing task for a video.
        
        Args:
            video_id: ID of the video to create task for
            task_type: Type of task to create
            
        Returns:
            Created task object
        """
        task = ProcessingTask(
            video_id=video_id,
            task_type=task_type,
            status=ProcessingStatus.PENDING.value,
            progress=0.0
        )
        
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        
        logger.info(f"Created processing task: {task.id} ({task_type}) for video {video_id}")
        return task
    
    async def _process_enhancement(self, video_id: str, task_id: str) -> None:
        """
        Process video enhancement task.
        
        Args:
            video_id: ID of the video to enhance
            task_id: ID of the task
        """
        logger.info(f"Starting enhancement for video {video_id}")
        
        # Get video and task
        video = await self.get_video(video_id)
        if not video:
            logger.error(f"Video not found for enhancement: {video_id}")
            return
        
        # Update task status
        task = await self._get_task(task_id)
        if not task:
            logger.error(f"Task not found for enhancement: {task_id}")
            return
        
        try:
            # Update task status to processing
            await self._update_task_status(
                task_id, 
                ProcessingStatus.ENHANCING.value,
                progress=10
            )
            
            # In a real implementation, we would call an enhancement service
            # For now, we'll simulate enhancement by creating a copy of the original video
            
            # Generate path for enhanced video
            file_ext = os.path.splitext(video.storage_path)[1]
            enhanced_path = f"videos/{video.user_id}/enhanced_{uuid.uuid4()}{file_ext}"
            
            # Copy original video to enhanced path
            success = await self.storage.copy_file(video.storage_path, enhanced_path)
            
            if not success:
                raise Exception("Failed to create enhanced video")
            
            # Update video record with enhanced video path
            video.enhanced_video_path = enhanced_path
            video.has_enhancement = True
            await self.db.commit()
            
            # Mark task as completed
            await self._update_task_status(
                task_id,
                ProcessingStatus.COMPLETED.value,
                progress=100
            )
            
            logger.info(f"Enhancement completed for video {video_id}")
        except Exception as e:
            logger.error(f"Error enhancing video {video_id}: {str(e)}")
            await self._update_task_status(
                task_id,
                ProcessingStatus.FAILED.value,
                error_message=f"Enhancement error: {str(e)}"
            )
    
    async def _process_avatar(self, video_id: str, task_id: str) -> None:
        """
        Process avatar generation task.
        
        Args:
            video_id: ID of the video for avatar generation
            task_id: ID of the task
        """
        logger.info(f"Starting avatar generation for video {video_id}")
        
        # Get video and task
        video = await self.get_video(video_id)
        if not video:
            logger.error(f"Video not found for avatar generation: {video_id}")
            return
        
        # Update task status
        task = await self._get_task(task_id)
        if not task:
            logger.error(f"Task not found for avatar generation: {task_id}")
            return
        
        try:
            # Update task status to processing
            await self._update_task_status(
                task_id, 
                ProcessingStatus.GENERATING_AVATAR.value,
                progress=10
            )
            
            # Check if avatar service is enabled
            if not settings.AVATAR_GENERATION_ENABLED:
                raise Exception("Avatar generation is not enabled")
            
            # In a real implementation, we would call the avatar service
            # For now, we'll simulate avatar generation with a placeholder
            
            # Simulate processing time
            await asyncio.sleep(2)
            
            # Generate path for avatar video
            file_ext = os.path.splitext(video.storage_path)[1]
            avatar_path = f"videos/{video.user_id}/avatar_{uuid.uuid4()}{file_ext}"
            
            # In a real implementation, we would generate the avatar video
            # For now, we'll copy the original as a placeholder
            success = await self.storage.copy_file(video.storage_path, avatar_path)
            
            if not success:
                raise Exception("Failed to create avatar video")
            
            # Update video record with avatar video path
            video.avatar_video_path = avatar_path
            video.has_avatar = True
            await self.db.commit()
            
            # Mark task as completed
            await self._update_task_status(
                task_id,
                ProcessingStatus.COMPLETED.value,
                progress=100
            )
            
            logger.info(f"Avatar generation completed for video {video_id}")
        except Exception as e:
            logger.error(f"Error generating avatar for video {video_id}: {str(e)}")
            await self._update_task_status(
                task_id,
                ProcessingStatus.FAILED.value,
                error_message=f"Avatar generation error: {str(e)}"
            )
    
    async def _get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """
        Get a processing task by ID.
        
        Args:
            task_id: ID of the task to get
            
        Returns:
            Task object if found, None otherwise
        """
        stmt = select(ProcessingTask).where(ProcessingTask.id == task_id)
        result = await self.db.execute(stmt)
        task = result.scalars().first()
        
        return task
    
    async def _update_task_status(
        self, 
        task_id: str, 
        status: str, 
        progress: float = None,
        error_message: str = None
    ) -> Optional[ProcessingTask]:
        """
        Update the status of a processing task.
        
        Args:
            task_id: ID of the task to update
            status: New status value
            progress: Optional progress value (0-100)
            error_message: Optional error message if status is failed
            
        Returns:
            Updated task object if found, None otherwise
        """
        # Get task
        task = await self._get_task(task_id)
        if not task:
            logger.warning(f"Task not found for status update: {task_id}")
            return None
        
        # Update status
        task.status = status
        if progress is not None:
            task.progress = progress
        if error_message:
            task.error_message = error_message
        
        # Update timestamps
        if status == ProcessingStatus.PROCESSING.value and not task.started_at:
            task.started_at = datetime.utcnow()
        elif status in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value] and not task.completed_at:
            task.completed_at = datetime.utcnow()
        
        # Save to database
        await self.db.commit()
        await self.db.refresh(task)
        
        logger.info(f"Updated task status: {task_id} -> {status}")
        
        # Also update the video's overall status
        await self._update_overall_status(task.video_id)
        
        return task
    
    async def _update_overall_status(self, video_id: str) -> None:
        """
        Update the overall status of a video based on its tasks.
        
        Args:
            video_id: ID of the video to update
        """
        # Get video
        video = await self.get_video(video_id)
        if not video:
            logger.warning(f"Video not found for status update: {video_id}")
            return
        
        # Get all tasks
        stmt = select(ProcessingTask).where(ProcessingTask.video_id == video_id)
        result = await self.db.execute(stmt)
        tasks = result.scalars().all()
        
        if not tasks:
            return
        
        # Count tasks by status
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == ProcessingStatus.COMPLETED.value)
        failed = sum(1 for t in tasks if t.status == ProcessingStatus.FAILED.value)
        pending_processing = total - completed - failed
        
        # Calculate overall progress
        if total > 0:
            overall_progress = sum(t.progress for t in tasks) / total
        else:
            overall_progress = 0
        
        # Update video status based on tasks
        if failed == total:
            # All tasks failed
            await self.update_video_status(
                video_id,
                ProcessingStatus.FAILED.value,
                progress=overall_progress,
                error_message="All processing tasks failed"
            )
        elif completed == total:
            # All tasks completed
            await self.update_video_status(
                video_id,
                ProcessingStatus.COMPLETED.value,
                progress=100
            )
        elif failed > 0:
            # Some tasks failed
            await self.update_video_status(
                video_id,
                ProcessingStatus.PROCESSING.value,
                progress=overall_progress,
                error_message=f"{failed} task(s) failed, {pending_processing} in progress"
            )
        else:
            # Some tasks still processing
            current_status = ProcessingStatus.PROCESSING.value
            
            # Get the specific processing state from the current task if possible
            for task in tasks:
                if task.status not in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value, ProcessingStatus.PENDING.value]:
                    current_status = task.status
                    break
            
            await self.update_video_status(
                video_id,
                current_status,
                progress=overall_progress
            ) 