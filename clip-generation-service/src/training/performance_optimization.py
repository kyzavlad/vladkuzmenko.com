import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import gc
import psutil
import os

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int
    max_sequence_length: int
    num_workers: int
    pin_memory: bool = True
    prefetch_factor: int = 2

class MemoryManager:
    """Manages memory usage during training."""
    
    def __init__(self, device: torch.device):
        """
        Initialize memory manager.
        
        Args:
            device (torch.device): Device to monitor
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Dict[str, float]: Memory usage statistics
        """
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        else:
            process = psutil.Process(os.getpid())
            return {
                "rss": process.memory_info().rss / 1024**3,  # GB
                "vms": process.memory_info().vms / 1024**3  # GB
            }
    
    def clear_cache(self):
        """Clear memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def log_memory_usage(self, stage: str):
        """
        Log current memory usage.
        
        Args:
            stage (str): Current processing stage
        """
        usage = self.get_memory_usage()
        self.logger.info(f"Memory usage at {stage}:")
        for key, value in usage.items():
            self.logger.info(f"  {key}: {value:.2f} GB")

class BatchProcessor:
    """Optimizes batch processing for GPU utilization."""
    
    def __init__(
        self,
        model: nn.Module,
        batch_config: BatchConfig,
        memory_manager: MemoryManager
    ):
        """
        Initialize batch processor.
        
        Args:
            model (nn.Module): Model to process batches
            batch_config (BatchConfig): Batch processing configuration
            memory_manager (MemoryManager): Memory manager instance
        """
        self.model = model
        self.batch_config = batch_config
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
    
    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process a single batch.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for training
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Loss and metrics
        """
        # Move batch to device
        batch = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Forward pass
        outputs = self.model(
            batch["visual_features"],
            batch["audio_features"],
            batch["text_features"],
            batch["mask"]
        )
        
        # Compute loss if optimizer is provided (training mode)
        if optimizer is not None:
            loss, metrics = self.model.criterion(
                outputs[0],
                outputs[1],
                batch["targets"],
                batch["engagement_targets"]
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss, metrics
        
        return outputs
    
    def process_batches(
        self,
        data_loader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> List[Dict[str, float]]:
        """
        Process multiple batches.
        
        Args:
            data_loader (DataLoader): Data loader
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for training
            
        Returns:
            List[Dict[str, float]]: List of metrics for each batch
        """
        self.memory_manager.clear_cache()
        self.memory_manager.log_memory_usage("start")
        
        batch_metrics = []
        
        for batch_idx, batch in enumerate(data_loader):
            # Process batch
            if optimizer is not None:
                loss, metrics = self.process_batch(batch, optimizer)
            else:
                outputs = self.process_batch(batch)
                metrics = {"output": outputs}
            
            batch_metrics.append(metrics)
            
            # Log memory usage periodically
            if (batch_idx + 1) % 10 == 0:
                self.memory_manager.log_memory_usage(f"batch_{batch_idx + 1}")
        
        self.memory_manager.clear_cache()
        self.memory_manager.log_memory_usage("end")
        
        return batch_metrics

def create_optimized_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_config: BatchConfig
) -> torch.utils.data.DataLoader:
    """
    Create an optimized data loader.
    
    Args:
        dataset (Dataset): Dataset to load
        batch_config (BatchConfig): Batch configuration
        
    Returns:
        DataLoader: Optimized data loader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_config.batch_size,
        num_workers=batch_config.num_workers,
        pin_memory=batch_config.pin_memory,
        prefetch_factor=batch_config.prefetch_factor,
        persistent_workers=True
    ) 