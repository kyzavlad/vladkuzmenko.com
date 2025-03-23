import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import wandb
import json

from ..models.interesting_moment import (
    InterestingMomentModel,
    InterestingMomentLoss
)

class InterestingMomentDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 32,
        platform: Optional[str] = None,
        category: Optional[str] = None
    ):
        """
        Dataset for interesting moment detection.
        
        Args:
            data_dir (str): Data directory
            sequence_length (int): Length of video sequences
            platform (Optional[str]): Platform filter (TikTok, Instagram, YouTube)
            category (Optional[str]): Content category filter
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Filter by platform and category if specified
        self.sequences = []
        for seq in self.metadata["sequences"]:
            if platform and seq["platform"] != platform:
                continue
            if category and seq["category"] != category:
                continue
            self.sequences.append(seq)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Load features
        features = {
            "visual": torch.load(
                os.path.join(self.data_dir, sequence["visual_features"])
            ),
            "audio": torch.load(
                os.path.join(self.data_dir, sequence["audio_features"])
            ),
            "text": torch.load(
                os.path.join(self.data_dir, sequence["text_features"])
            )
        }
        
        # Load annotations
        annotations = torch.load(
            os.path.join(self.data_dir, sequence["annotations"])
        )
        
        # Create attention mask
        mask = torch.ones(self.sequence_length)
        if len(features["visual"]) < self.sequence_length:
            mask[len(features["visual"]):] = 0
        
        return {
            "visual_features": features["visual"],
            "audio_features": features["audio"],
            "text_features": features["text"],
            "targets": annotations["labels"],
            "engagement_targets": annotations["engagement"],
            "mask": mask,
            "metadata": {
                "platform": sequence["platform"],
                "category": sequence["category"],
                "video_id": sequence["video_id"]
            }
        }

def train_interesting_moment(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    sequence_length: int = 32,
    platform: Optional[str] = None,
    category: Optional[str] = None,
    val_split: float = 0.2,
    log_interval: int = 10,
    checkpoint_interval: int = 5,
    use_wandb: bool = True
):
    """
    Train interesting moment detection model.
    
    Args:
        data_dir (str): Data directory
        output_dir (str): Output directory for checkpoints
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        sequence_length (int): Length of video sequences
        platform (Optional[str]): Platform filter
        category (Optional[str]): Content category filter
        val_split (float): Validation split ratio
        log_interval (int): Logging interval
        checkpoint_interval (int): Checkpoint saving interval
        use_wandb (bool): Whether to use Weights & Biases
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="interesting-moment-detection",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "platform": platform,
                "category": category,
                "num_epochs": num_epochs
            }
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = InterestingMomentDataset(
        data_dir,
        sequence_length,
        platform,
        category
    )
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = InterestingMomentModel(
        sequence_length=sequence_length
    ).to(device)
    
    criterion = InterestingMomentLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # Move data to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            logits, engagement = model(
                batch["visual_features"],
                batch["audio_features"],
                batch["text_features"],
                batch["mask"]
            )
            
            # Compute loss
            loss, loss_dict = criterion(
                logits,
                engagement,
                batch["targets"],
                batch["engagement_targets"]
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss_dict)
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"Train Batch {batch_idx + 1}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )
                
                if use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        **{f"train_{k}": v for k, v in loss_dict.items()}
                    })
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                # Move data to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                logits, engagement = model(
                    batch["visual_features"],
                    batch["audio_features"],
                    batch["text_features"],
                    batch["mask"]
                )
                
                # Compute loss
                loss, loss_dict = criterion(
                    logits,
                    engagement,
                    batch["targets"],
                    batch["engagement_targets"]
                )
                
                val_losses.append(loss_dict)
        
        # Compute average losses
        avg_train_loss = np.mean([l["total"] for l in train_losses])
        avg_val_loss = np.mean([l["total"] for l in val_losses])
        
        logger.info(
            f"Epoch {epoch + 1} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss
            })
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                output_dir,
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model: {best_model_path}")
    
    logger.info("Training completed!")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--platform", type=str, choices=["TikTok", "Instagram", "YouTube"])
    parser.add_argument("--category", type=str)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    train_interesting_moment(**vars(args)) 