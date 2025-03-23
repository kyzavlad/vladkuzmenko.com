"""
Clip Generation Service

This module provides the core service for generating video clips from source videos
using various parameters and configurations.
"""

import os
import time
import uuid
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClipGenerationService:
    """
    Service for generating video clips from source videos.
    
    This service handles the validation, queuing, and management of clip generation
    tasks. It integrates with the video processing service for actual video manipulation.
    """
    
    def __init__(
        self,
        output_dir: str = "output/clips",
        temp_dir: str = "temp/clips",
        max_concurrent_tasks: int = 5,
        default_format: str = "mp4",
        default_quality: str = "high"
    ):
        """
        Initialize the clip generation service.
        
        Args:
            output_dir: Directory for storing generated clips
            temp_dir: Directory for temporary files during processing
            max_concurrent_tasks: Maximum number of tasks to process concurrently
            default_format: Default output format for generated clips
            default_quality: Default quality setting for generated clips
        """
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_format = default_format
        self.default_quality = default_quality
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Task tracking
        self.tasks = {}  # Dictionary to track task status
        self.active_tasks = 0
        
        logger.info(f"ClipGenerationService initialized")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Max concurrent tasks: {max_concurrent_tasks}")
    
    def create_clip_task(
        self,
        source_video: str,
        start_time: float,
        end_time: float,
        output_name: Optional[str] = None,
        format: Optional[str] = None,
        quality: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new clip generation task.
        
        Args:
            source_video: Path to the source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_name: Name for the output clip (without extension)
            format: Output format (e.g., mp4, avi, mov)
            quality: Quality setting (low, medium, high)
            params: Additional parameters for clip generation
            
        Returns:
            Task information dictionary
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Set default values
        format = format or self.default_format
        quality = quality or self.default_quality
        
        if output_name is None:
            # Generate a default output name based on source file and timestamps
            base_name = os.path.splitext(os.path.basename(source_video))[0]
            output_name = f"{base_name}_clip_{start_time:.1f}_{end_time:.1f}"
        
        # Create task info
        task_info = {
            "task_id": task_id,
            "source_video": source_video,
            "start_time": start_time,
            "end_time": end_time,
            "output_name": output_name,
            "output_path": os.path.join(self.output_dir, f"{output_name}.{format}"),
            "format": format,
            "quality": quality,
            "params": params or {},
            "status": "created",
            "created_at": time.time(),
            "updated_at": time.time(),
            "completed_at": None,
            "error": None
        }
        
        # Store task information
        self.tasks[task_id] = task_info
        
        logger.info(f"Created clip generation task {task_id} for {source_video}")
        
        # Queue the task for processing
        self._queue_task(task_id)
        
        return task_info
    
    def _queue_task(self, task_id: str) -> None:
        """
        Queue a task for processing.
        
        Args:
            task_id: The ID of the task to queue
        """
        # Update task status
        self.tasks[task_id]["status"] = "queued"
        self.tasks[task_id]["updated_at"] = time.time()
        
        logger.info(f"Queued task {task_id} for processing")
        
        # In a real implementation, this would add the task to a message queue
        # For now, we'll simulate immediate processing if capacity is available
        if self.active_tasks < self.max_concurrent_tasks:
            self._process_task(task_id)
    
    def _process_task(self, task_id: str) -> None:
        """
        Process a clip generation task.
        
        Args:
            task_id: The ID of the task to process
        """
        # Update task status
        self.tasks[task_id]["status"] = "processing"
        self.tasks[task_id]["updated_at"] = time.time()
        self.active_tasks += 1
        
        logger.info(f"Processing task {task_id}")
        
        # In a real implementation, this would delegate to a worker process
        # For now, we'll simulate task processing
        try:
            # Simulated processing time
            time.sleep(2)
            
            # Mark task as completed
            self._complete_task(task_id)
        except Exception as e:
            # Handle task failure
            self._fail_task(task_id, str(e))
    
    def _complete_task(self, task_id: str) -> None:
        """
        Mark a task as completed.
        
        Args:
            task_id: The ID of the completed task
        """
        # Update task status
        self.tasks[task_id]["status"] = "completed"
        self.tasks[task_id]["updated_at"] = time.time()
        self.tasks[task_id]["completed_at"] = time.time()
        self.active_tasks -= 1
        
        logger.info(f"Completed task {task_id}")
        
        # Process next task if available
        self._process_next_task()
    
    def _fail_task(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed.
        
        Args:
            task_id: The ID of the failed task
            error: Error message
        """
        # Update task status
        self.tasks[task_id]["status"] = "failed"
        self.tasks[task_id]["updated_at"] = time.time()
        self.tasks[task_id]["error"] = error
        self.active_tasks -= 1
        
        logger.error(f"Task {task_id} failed: {error}")
        
        # Process next task if available
        self._process_next_task()
    
    def _process_next_task(self) -> None:
        """
        Process the next queued task if capacity is available.
        """
        if self.active_tasks >= self.max_concurrent_tasks:
            return
        
        # Find the next queued task
        for task_id, task_info in self.tasks.items():
            if task_info["status"] == "queued":
                self._process_task(task_id)
                break
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Task information dictionary
            
        Raises:
            KeyError: If the task ID is not found
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        
        return self.tasks[task_id]
    
    def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List tasks with optional filtering.
        
        Args:
            status: Filter by task status
            limit: Maximum number of tasks to return
            offset: Offset for pagination
            
        Returns:
            List of task information dictionaries
        """
        tasks = list(self.tasks.values())
        
        # Filter by status if provided
        if status:
            tasks = [task for task in tasks if task["status"] == status]
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        tasks = tasks[offset:offset + limit]
        
        return tasks
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a pending or processing task.
        
        Args:
            task_id: The ID of the task to cancel
            
        Returns:
            Updated task information dictionary
            
        Raises:
            KeyError: If the task ID is not found
            ValueError: If the task cannot be cancelled
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Only queued or processing tasks can be cancelled
        if task["status"] not in ["created", "queued", "processing"]:
            raise ValueError(f"Task {task_id} cannot be cancelled (status: {task['status']})")
        
        # Update task status
        task["status"] = "cancelled"
        task["updated_at"] = time.time()
        
        # If the task was processing, decrement active task count
        if task["status"] == "processing":
            self.active_tasks -= 1
            self._process_next_task()
        
        logger.info(f"Cancelled task {task_id}")
        
        return task 