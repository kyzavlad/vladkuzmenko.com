import asyncio
import torch
import os
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    save_dir: str
    save_interval: int  # Save every N batches
    max_checkpoints: int = 5
    save_best_only: bool = True

class AsyncProcessor:
    """Handles asynchronous processing with checkpointing."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_config: CheckpointConfig,
        device: torch.device
    ):
        """
        Initialize async processor.
        
        Args:
            model (torch.nn.Module): Model to process
            checkpoint_config (CheckpointConfig): Checkpoint configuration
            device (torch.device): Device to use
        """
        self.model = model
        self.checkpoint_config = checkpoint_config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_config.save_dir, exist_ok=True)
        
        # Initialize checkpoint tracking
        self.best_metric = float("-inf")
        self.checkpoint_history: List[str] = []
    
    async def save_checkpoint(
        self,
        epoch: int,
        batch_idx: int,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ):
        """
        Save a checkpoint.
        
        Args:
            epoch (int): Current epoch
            batch_idx (int): Current batch index
            optimizer_state (Optional[Dict]): Optimizer state
            scheduler_state (Optional[Dict]): Scheduler state
            metrics (Optional[Dict]): Current metrics
        """
        checkpoint = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine if this is the best checkpoint
        is_best = False
        if metrics and self.checkpoint_config.save_best_only:
            current_metric = metrics.get("val_loss", float("inf"))
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                is_best = True
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_config.save_dir,
            f"checkpoint_epoch{epoch}_batch{batch_idx}.pth"
        )
        
        # Save in background
        await asyncio.to_thread(torch.save, checkpoint, checkpoint_path)
        
        # Update checkpoint history
        self.checkpoint_history.append(checkpoint_path)
        
        # Remove old checkpoints if needed
        if len(self.checkpoint_history) > self.checkpoint_config.max_checkpoints:
            oldest_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(oldest_checkpoint):
                await asyncio.to_thread(os.remove, oldest_checkpoint)
        
        # Save best checkpoint separately
        if is_best:
            best_path = os.path.join(
                self.checkpoint_config.save_dir,
                "best_model.pth"
            )
            await asyncio.to_thread(torch.save, checkpoint, best_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    async def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to restore
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Scheduler to restore
            
        Returns:
            Dict: Checkpoint data
        """
        checkpoint = await asyncio.to_thread(torch.load, checkpoint_path)
        
        # Restore model state
        self.model.load_state_dict(checkpoint["model_state"])
        
        # Restore optimizer state if provided
        if optimizer and checkpoint["optimizer_state"]:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Restore scheduler state if provided
        if scheduler and checkpoint["scheduler_state"]:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
    
    async def process_batch_async(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Process a batch asynchronously.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for training
            
        Returns:
            Dict: Processing results
        """
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
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
            
            return {"loss": loss.item(), "metrics": metrics}
        
        return {"outputs": outputs}
    
    async def process_epoch_async(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> List[Dict]:
        """
        Process an epoch asynchronously.
        
        Args:
            data_loader (DataLoader): Data loader
            epoch (int): Current epoch
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for training
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler
            
        Returns:
            List[Dict]: List of batch results
        """
        batch_results = []
        
        for batch_idx, batch in enumerate(data_loader):
            # Process batch
            result = await self.process_batch_async(batch, optimizer)
            batch_results.append(result)
            
            # Save checkpoint if needed
            if (batch_idx + 1) % self.checkpoint_config.save_interval == 0:
                await self.save_checkpoint(
                    epoch,
                    batch_idx,
                    optimizer.state_dict() if optimizer else None,
                    scheduler.state_dict() if scheduler else None,
                    result.get("metrics")
                )
        
        return batch_results 