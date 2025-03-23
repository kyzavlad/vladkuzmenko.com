import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import cv2
from scipy.signal import find_peaks

@dataclass
class DownsamplingConfig:
    """Configuration for intelligent downsampling."""
    target_fps: int
    min_frames: int
    max_frames: int
    motion_threshold: float = 0.1
    content_threshold: float = 0.2
    temporal_smoothing: int = 5

class IntelligentDownsampler:
    """Performs intelligent downsampling of video frames."""
    
    def __init__(
        self,
        config: DownsamplingConfig,
        device: torch.device
    ):
        """
        Initialize downsampler.
        
        Args:
            config (DownsamplingConfig): Downsampling configuration
            device (torch.device): Device to use
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def compute_motion_score(
        self,
        frames: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute motion score between consecutive frames.
        
        Args:
            frames (torch.Tensor): Video frames
            
        Returns:
            torch.Tensor: Motion scores
        """
        # Convert to grayscale
        if frames.shape[1] == 3:  # RGB
            frames = frames.mean(dim=1, keepdim=True)
        
        # Compute optical flow
        flow_scores = []
        for i in range(len(frames) - 1):
            prev_frame = frames[i].cpu().numpy()
            curr_frame = frames[i + 1].cpu().numpy()
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame,
                curr_frame,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Compute flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_scores.append(magnitude.mean())
        
        return torch.tensor(flow_scores, device=self.device)
    
    def compute_content_score(
        self,
        frames: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute content importance score for each frame.
        
        Args:
            frames (torch.Tensor): Video frames
            model (Optional[nn.Module]): Feature extraction model
            
        Returns:
            torch.Tensor: Content scores
        """
        if model is None:
            # Use simple edge detection as fallback
            scores = []
            for frame in frames:
                frame_np = frame.cpu().numpy()
                edges = cv2.Canny(frame_np, 100, 200)
                scores.append(edges.mean())
        else:
            # Extract features using the model
            with torch.no_grad():
                features = model(frames)
                # Compute feature variance as importance score
                scores = features.var(dim=1).mean(dim=1)
        
        return torch.tensor(scores, device=self.device)
    
    def smooth_scores(
        self,
        scores: torch.Tensor,
        window_size: int
    ) -> torch.Tensor:
        """
        Apply temporal smoothing to scores.
        
        Args:
            scores (torch.Tensor): Input scores
            window_size (int): Smoothing window size
            
        Returns:
            torch.Tensor: Smoothed scores
        """
        kernel = torch.ones(window_size) / window_size
        kernel = kernel.to(self.device)
        
        # Pad scores for convolution
        padded = torch.nn.functional.pad(
            scores,
            (window_size // 2, window_size // 2),
            mode="replicate"
        )
        
        # Apply convolution
        smoothed = torch.nn.functional.conv1d(
            padded.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0)
        ).squeeze()
        
        return smoothed
    
    def select_keyframes(
        self,
        frames: torch.Tensor,
        motion_scores: torch.Tensor,
        content_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select keyframes based on motion and content scores.
        
        Args:
            frames (torch.Tensor): Video frames
            motion_scores (torch.Tensor): Motion scores
            content_scores (torch.Tensor): Content scores
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Selected frames and indices
        """
        # Combine scores
        combined_scores = (
            self.config.motion_threshold * motion_scores +
            self.config.content_threshold * content_scores
        )
        
        # Smooth scores
        smoothed_scores = self.smooth_scores(
            combined_scores,
            self.config.temporal_smoothing
        )
        
        # Find peaks in scores
        peaks, _ = find_peaks(
            smoothed_scores.cpu().numpy(),
            distance=self.config.target_fps // 2
        )
        
        # Ensure minimum and maximum number of frames
        if len(peaks) < self.config.min_frames:
            # Add frames with highest scores
            remaining = self.config.min_frames - len(peaks)
            non_peak_scores = np.delete(smoothed_scores.cpu().numpy(), peaks)
            non_peak_indices = np.delete(np.arange(len(frames)), peaks)
            additional_indices = non_peak_indices[
                np.argsort(non_peak_scores)[-remaining:]
            ]
            peaks = np.concatenate([peaks, additional_indices])
        elif len(peaks) > self.config.max_frames:
            # Keep frames with highest scores
            peak_scores = smoothed_scores[peaks]
            top_indices = np.argsort(peak_scores)[-self.config.max_frames:]
            peaks = peaks[top_indices]
        
        # Sort indices
        peaks = np.sort(peaks)
        
        return frames[peaks], torch.tensor(peaks, device=self.device)
    
    def downsample(
        self,
        frames: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform intelligent downsampling.
        
        Args:
            frames (torch.Tensor): Video frames
            model (Optional[nn.Module]): Feature extraction model
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Downsampled frames and indices
        """
        # Compute motion scores
        motion_scores = self.compute_motion_score(frames)
        
        # Compute content scores
        content_scores = self.compute_content_score(frames, model)
        
        # Select keyframes
        keyframes, indices = self.select_keyframes(
            frames,
            motion_scores,
            content_scores
        )
        
        self.logger.info(
            f"Downsampled from {len(frames)} to {len(keyframes)} frames"
        )
        
        return keyframes, indices 