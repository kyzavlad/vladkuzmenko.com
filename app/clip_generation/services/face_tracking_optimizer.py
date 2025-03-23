"""
Face Tracking Optimizer Module

This module provides performance optimization for the Face Tracking System,
including frame sampling, GPU acceleration, and batch processing.
"""

import os
import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
import threading
from queue import Queue
import torch

from app.clip_generation.services.face_tracking import FaceBox, FaceDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Enumeration of frame sampling strategies."""
    UNIFORM = "uniform"  # Sample every N frames
    ADAPTIVE = "adaptive"  # Adjust sampling rate based on movement
    KEYFRAME = "keyframe"  # Sample on scene changes
    MOTION = "motion"  # Sample based on motion detection


class FaceTrackingOptimizer:
    """
    Performance optimizer for face tracking.
    
    This class provides optimization techniques to improve the performance
    of face tracking, including frame sampling, GPU acceleration, and batch processing.
    """
    
    def __init__(
        self,
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        sampling_rate: int = 5,
        use_gpu: bool = True,
        batch_size: int = 4,
        worker_threads: int = 2,
        motion_threshold: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the face tracking optimizer.
        
        Args:
            sampling_strategy: Strategy for frame sampling
            sampling_rate: Base sampling rate (process 1 in N frames)
            use_gpu: Whether to use GPU acceleration
            batch_size: Size of batches for batch processing
            worker_threads: Number of worker threads for parallel processing
            motion_threshold: Threshold for motion-based sampling
            device: Computation device ('cuda' or 'cpu')
        """
        self.sampling_strategy = sampling_strategy
        self.sampling_rate = sampling_rate
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.worker_threads = worker_threads
        self.motion_threshold = motion_threshold
        self.device = device
        
        # Internal state
        self.last_processed_frame = None
        self.frame_index = 0
        self.last_keyframe_index = 0
        self.adaptive_rate = sampling_rate
        self.batch_queue = Queue(maxsize=batch_size * 2)
        self.result_queue = Queue()
        self.processing = False
        self.workers = []
        
        # Initialize frame differencing for motion detection
        self.prev_gray = None
        
        logger.info(f"Initialized FaceTrackingOptimizer with {sampling_strategy} "
                   f"sampling, GPU={self.use_gpu}, batch_size={batch_size}")
        
        # Initialize GPU context if available
        if self.use_gpu:
            self._init_gpu()
    
    def _init_gpu(self) -> None:
        """Initialize GPU acceleration."""
        try:
            if torch.cuda.is_available():
                # Set up CUDA device
                torch.cuda.set_device(0)  # Use first GPU
                
                # Initialize CUDA context for OpenCV if available
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    cv2.cuda.setDevice(0)
                    
                logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
                
                # Warm up GPU
                dummy_tensor = torch.zeros((1, 3, 640, 640)).to(self.device)
                _ = dummy_tensor + dummy_tensor  # Simple operation to initialize CUDA context
            else:
                logger.warning("GPU requested but not available. Falling back to CPU.")
                self.use_gpu = False
                self.device = "cpu"
        except Exception as e:
            logger.error(f"Error initializing GPU: {str(e)}. Falling back to CPU.")
            self.use_gpu = False
            self.device = "cpu"
    
    def start_batch_processing(self, detector: FaceDetector) -> None:
        """
        Start batch processing workers.
        
        Args:
            detector: Face detector to use for processing
        """
        if self.processing:
            return
            
        self.processing = True
        self.batch_queue = Queue(maxsize=self.batch_size * 2)
        self.result_queue = Queue()
        
        for i in range(self.worker_threads):
            thread = threading.Thread(
                target=self._batch_worker,
                args=(detector, i),
                daemon=True
            )
            thread.start()
            self.workers.append(thread)
            
        logger.info(f"Started {self.worker_threads} batch processing workers")
    
    def stop_batch_processing(self) -> None:
        """Stop batch processing workers."""
        self.processing = False
        
        # Wait for workers to finish
        for thread in self.workers:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        self.workers = []
        logger.info("Stopped batch processing workers")
    
    def _batch_worker(self, detector: FaceDetector, worker_id: int) -> None:
        """
        Worker thread for batch processing.
        
        Args:
            detector: Face detector to use
            worker_id: Worker thread ID
        """
        logger.info(f"Batch worker {worker_id} started")
        
        while self.processing:
            try:
                # Get a batch of frames
                batch = []
                batch_indices = []
                
                # Try to fill the batch
                for _ in range(self.batch_size):
                    if not self.processing:
                        break
                        
                    try:
                        item = self.batch_queue.get(timeout=0.1)
                        batch.append(item[0])  # Frame
                        batch_indices.append(item[1])  # Frame index
                    except:
                        # Queue empty or timeout
                        break
                
                if not batch:
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
                    continue
                
                # Process the batch
                start_time = time.time()
                
                if hasattr(detector, 'detect_batch'):
                    # Use batch detection if available
                    results = detector.detect_batch(batch)
                else:
                    # Fall back to sequential processing
                    results = []
                    for frame in batch:
                        results.append(detector.detect(frame))
                
                process_time = time.time() - start_time
                
                # Put results in result queue
                for idx, (frame_idx, face_boxes) in enumerate(zip(batch_indices, results)):
                    self.result_queue.put((frame_idx, face_boxes))
                    
                logger.debug(f"Worker {worker_id} processed batch of {len(batch)} "
                           f"frames in {process_time:.3f}s")
                    
            except Exception as e:
                logger.error(f"Error in batch worker {worker_id}: {str(e)}")
    
    def should_process_frame(self, frame: np.ndarray) -> bool:
        """
        Determine if the current frame should be processed based on sampling strategy.
        
        Args:
            frame: Current video frame
            
        Returns:
            Whether to process this frame
        """
        self.frame_index += 1
        
        if self.sampling_strategy == SamplingStrategy.UNIFORM:
            # Simple uniform sampling
            return self.frame_index % self.sampling_rate == 0
            
        elif self.sampling_strategy == SamplingStrategy.ADAPTIVE:
            # Adaptive sampling based on previous detections
            if self.frame_index % self.adaptive_rate == 0:
                return True
            return False
            
        elif self.sampling_strategy == SamplingStrategy.KEYFRAME:
            # Detect scene changes
            process = self._is_keyframe(frame)
            if process:
                self.last_keyframe_index = self.frame_index
            # Always process if it's been too long since the last keyframe
            if self.frame_index - self.last_keyframe_index >= self.sampling_rate * 2:
                return True
            return process
            
        elif self.sampling_strategy == SamplingStrategy.MOTION:
            # Process based on motion detection
            return self._detect_significant_motion(frame)
            
        # Default: process every frame
        return True
    
    def _is_keyframe(self, frame: np.ndarray) -> bool:
        """
        Detect if the current frame is a keyframe (scene change).
        
        Args:
            frame: Current video frame
            
        Returns:
            Whether this frame is a keyframe
        """
        if self.last_processed_frame is None:
            self.last_processed_frame = frame.copy()
            return True
            
        # Convert frames to grayscale
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_gray = cv2.cvtColor(self.last_processed_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(current_gray, last_gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        change_percent = np.count_nonzero(thresh) / thresh.size
        
        # Update last processed frame
        self.last_processed_frame = frame.copy()
        
        # Consider it a keyframe if change is significant
        return change_percent > 0.2  # 20% change threshold
    
    def _detect_significant_motion(self, frame: np.ndarray) -> bool:
        """
        Detect if there is significant motion in the frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            Whether significant motion is detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame case
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return True
        
        # Calculate optical flow using Farneback method
        if self.use_gpu and hasattr(cv2, 'cuda'):
            # GPU version
            prev_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(self.prev_gray)
            
            curr_gpu = cv2.cuda_GpuMat()
            curr_gpu.upload(gray)
            
            flow_gpu = cv2.cuda.FarnebackOpticalFlow.create(
                5, 0.5, False, 15, 3, 5, 1.2, 0
            )
            flow = flow_gpu.calc(prev_gpu, curr_gpu, None)
            flow_cpu = flow.download()
            
            # Compute magnitude and angle
            mag, _ = cv2.cartToPolar(flow_cpu[..., 0], flow_cpu[..., 1])
        else:
            # CPU version
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Update previous frame
        self.prev_gray = gray.copy()
        
        # Calculate average motion
        mean_motion = np.mean(mag)
        
        return mean_motion > self.motion_threshold
    
    def update_adaptive_rate(self, face_count: int, process_time: float) -> None:
        """
        Update the adaptive sampling rate based on processing results.
        
        Args:
            face_count: Number of faces detected
            process_time: Time taken to process the frame
        """
        if self.sampling_strategy != SamplingStrategy.ADAPTIVE:
            return
            
        # Target 30ms processing time per frame for 30fps video
        target_time = 0.03
        
        if process_time > target_time * 2:
            # Too slow, increase sampling rate (skip more frames)
            self.adaptive_rate = min(20, self.adaptive_rate + 1)
        elif process_time < target_time / 2 and face_count > 0:
            # Fast enough with faces, decrease sampling rate (process more frames)
            self.adaptive_rate = max(1, self.adaptive_rate - 1)
        
        logger.debug(f"Adaptive sampling rate updated to {self.adaptive_rate}")
    
    def optimize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimize a frame for face detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            Optimized frame for detection
        """
        if not self.use_gpu:
            # Just resize for CPU processing
            return cv2.resize(frame, (640, 480))
        
        try:
            # Upload to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Resize on GPU
            gpu_resized = cv2.cuda.resize(gpu_frame, (640, 480))
            
            # Optional: Apply GPU-accelerated preprocessing
            # gpu_processed = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
            # gpu_processed = cv2.cuda.normalize(gpu_processed, None, 0, 1, cv2.NORM_MINMAX)
            
            # Download back to CPU
            optimized = gpu_resized.download()
            return optimized
            
        except Exception as e:
            logger.error(f"GPU frame optimization failed: {str(e)}")
            # Fallback to CPU
            return cv2.resize(frame, (640, 480))
    
    def add_to_batch_queue(self, frame: np.ndarray, frame_idx: int) -> None:
        """
        Add a frame to the batch processing queue.
        
        Args:
            frame: Video frame to process
            frame_idx: Frame index
        """
        try:
            # Optimize frame first
            optimized_frame = self.optimize_frame(frame)
            
            # Add to queue with timeout
            self.batch_queue.put((optimized_frame, frame_idx), timeout=0.1)
            
        except Exception as e:
            logger.error(f"Error adding frame to batch queue: {str(e)}")
    
    def get_batch_results(self) -> List[Tuple[int, List[FaceBox]]]:
        """
        Get available results from the batch processing queue.
        
        Returns:
            List of (frame_idx, face_boxes) tuples
        """
        results = []
        
        # Get all available results without blocking
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except:
                break
                
        # Sort by frame index
        results.sort(key=lambda x: x[0])
        
        return results 