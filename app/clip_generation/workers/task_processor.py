"""
Task Processor Module

This module provides the worker implementation for processing clip generation tasks
from the task queue.
"""

import os
import time
import json
import logging
import threading
import queue
from typing import Dict, List, Optional, Union, Any

from app.clip_generation.services.clip_generator import ClipGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskProcessor:
    """
    Worker process that processes clip generation tasks from a queue.
    
    This class handles the execution of clip generation tasks, managing
    worker threads, and reporting task status.
    """
    
    def __init__(
        self,
        output_dir: str = "output/clips",
        temp_dir: str = "temp/clips",
        num_workers: int = 2,
        ffmpeg_path: str = "ffmpeg"
    ):
        """
        Initialize the task processor.
        
        Args:
            output_dir: Directory for storing output clips
            temp_dir: Directory for temporary files
            num_workers: Number of worker threads to spawn
            ffmpeg_path: Path to the ffmpeg executable
        """
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.num_workers = num_workers
        
        # Create the clip generator
        self.clip_generator = ClipGenerator(
            ffmpeg_path=ffmpeg_path,
            temp_dir=temp_dir
        )
        
        # Create task queue
        self.task_queue = queue.Queue()
        
        # Track active workers
        self.workers = []
        self.running = False
        
        # Task callbacks for reporting status
        self.task_callbacks = {}
        
        logger.info(f"TaskProcessor initialized with {num_workers} workers")
    
    def start(self) -> None:
        """
        Start the task processor and worker threads.
        """
        if self.running:
            logger.warning("TaskProcessor is already running")
            return
        
        logger.info(f"Starting TaskProcessor with {self.num_workers} workers")
        self.running = True
        
        # Create and start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"ClipWorker-{i}",
                daemon=True
            )
            self.workers.append(worker)
            worker.start()
            logger.info(f"Started worker thread {worker.name}")
    
    def stop(self) -> None:
        """
        Stop the task processor and worker threads.
        """
        if not self.running:
            logger.warning("TaskProcessor is not running")
            return
        
        logger.info("Stopping TaskProcessor")
        self.running = False
        
        # Wait for worker threads to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
                logger.info(f"Stopped worker thread {worker.name}")
        
        self.workers = []
        logger.info("TaskProcessor stopped")
    
    def add_task(self, task: Dict[str, Any], callback: Optional[callable] = None) -> None:
        """
        Add a task to the processing queue.
        
        Args:
            task: Task information dictionary
            callback: Optional callback function to call when the task completes
        """
        task_id = task.get("task_id")
        if not task_id:
            logger.error("Task has no task_id, cannot add to queue")
            return
        
        logger.info(f"Adding task {task_id} to queue")
        
        # Store callback if provided
        if callback:
            self.task_callbacks[task_id] = callback
        
        # Add to queue
        self.task_queue.put(task)
    
    def _worker_loop(self) -> None:
        """
        Worker thread function that processes tasks from the queue.
        """
        thread_name = threading.current_thread().name
        logger.info(f"Worker {thread_name} started")
        
        while self.running:
            try:
                # Get a task from the queue (block with timeout)
                try:
                    task = self.task_queue.get(block=True, timeout=1.0)
                except queue.Empty:
                    continue
                
                task_id = task.get("task_id")
                source_video = task.get("source_video")
                
                logger.info(f"Worker {thread_name} processing task {task_id}")
                
                # Process the task
                try:
                    self._process_task(task)
                    self._report_task_completed(task)
                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {str(e)}")
                    self._report_task_failed(task, str(e))
                
                # Mark task as done in queue
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {thread_name} encountered an error: {str(e)}")
                time.sleep(1.0)  # Prevent tight error loop
        
        logger.info(f"Worker {thread_name} stopped")
    
    def _process_task(self, task: Dict[str, Any]) -> None:
        """
        Process a clip generation task.
        
        Args:
            task: Task information dictionary
            
        Raises:
            Exception: If the task processing fails
        """
        task_id = task.get("task_id")
        source_video = task.get("source_video")
        start_time = task.get("start_time")
        end_time = task.get("end_time")
        output_path = task.get("output_path")
        format = task.get("format")
        quality = task.get("quality")
        params = task.get("params", {})
        
        # Validate required parameters
        if not all([task_id, source_video, start_time is not None, end_time is not None, output_path]):
            raise ValueError("Missing required task parameters")
        
        # Extract additional parameters
        resize = params.get("resize")
        audio = params.get("audio", True)
        codec = params.get("codec")
        
        # Generate the clip
        self.clip_generator.generate_clip(
            source_video=source_video,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
            quality=quality,
            format=format,
            codec=codec,
            resize=resize,
            audio=audio,
            additional_params=params
        )
    
    def _report_task_completed(self, task: Dict[str, Any]) -> None:
        """
        Report that a task has completed successfully.
        
        Args:
            task: Task information dictionary
        """
        task_id = task.get("task_id")
        
        # Update task status
        task["status"] = "completed"
        task["completed_at"] = time.time()
        
        logger.info(f"Task {task_id} completed successfully")
        
        # Call callback if registered
        callback = self.task_callbacks.get(task_id)
        if callback:
            try:
                callback(task)
                del self.task_callbacks[task_id]
            except Exception as e:
                logger.error(f"Error calling callback for task {task_id}: {str(e)}")
    
    def _report_task_failed(self, task: Dict[str, Any], error: str) -> None:
        """
        Report that a task has failed.
        
        Args:
            task: Task information dictionary
            error: Error message
        """
        task_id = task.get("task_id")
        
        # Update task status
        task["status"] = "failed"
        task["error"] = error
        task["completed_at"] = time.time()
        
        logger.error(f"Task {task_id} failed: {error}")
        
        # Call callback if registered
        callback = self.task_callbacks.get(task_id)
        if callback:
            try:
                callback(task)
                del self.task_callbacks[task_id]
            except Exception as e:
                logger.error(f"Error calling callback for task {task_id}: {str(e)}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the task queue.
        
        Returns:
            Dictionary containing queue statistics
        """
        return {
            "queue_size": self.task_queue.qsize(),
            "active_workers": len([w for w in self.workers if w.is_alive()]),
            "total_workers": len(self.workers),
            "running": self.running
        } 