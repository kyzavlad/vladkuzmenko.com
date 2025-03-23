import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import wandb

from ..models.face_tracking import FaceTrackingModel, FaceTrackingLoss

class FaceTrackingDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 16,
        transform: transforms.Compose = None
    ):
        """
        Dataset for face tracking.
        
        Args:
            data_dir (str): Data directory
            sequence_length (int): Length of video sequences
            transform (transforms.Compose): Image transformations
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load video sequences and annotations
        self.sequences = []
        self.annotations = []
        
        for video_dir in os.listdir(data_dir):
            video_path = os.path.join(data_dir, video_dir)
            if os.path.isdir(video_path):
                frames = sorted([
                    f for f in os.listdir(video_path)
                    if f.endswith((".jpg", ".png"))
                ])
                
                # Load annotations
                anno_path = os.path.join(video_path, "annotations.txt")
                if os.path.exists(anno_path):
                    with open(anno_path, "r") as f:
                        annotations = []
                        for line in f:
                            # Format: frame_idx x1 y1 x2 y2 identity_id
                            data = list(map(float, line.strip().split()))
                            annotations.append(data)
                            
                    # Group frames into sequences
                    for i in range(0, len(frames) - sequence_length + 1):
                        self.sequences.append(frames[i:i + sequence_length])
                        self.annotations.append(annotations[i:i + sequence_length])
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Load frames
        frames = []
        for frame_name in self.sequences[idx]:
            frame_path = os.path.join(self.data_dir, frame_name)
            frame = Image.open(frame_path).convert("RGB")
            frame = self.transform(frame)
            frames.append(frame)
        
        # Stack frames
        frames = torch.stack(frames)  # (T, C, H, W)
        
        # Process annotations
        annotations = self.annotations[idx]
        targets = {
            "boxes": torch.tensor([anno[1:5] for anno in annotations]),
            "labels": torch.tensor([1] * len(annotations)),  # 1 for face
            "identities": torch.tensor([anno[5] for anno in annotations])
        }
        
        return frames, targets

def train_face_tracking(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    sequence_length: int = 16,
    val_split: float = 0.2,
    log_interval: int = 10,
    checkpoint_interval: int = 5,
    use_wandb: bool = True
):
    """
    Train face tracking model.
    
    Args:
        data_dir (str): Data directory
        output_dir (str): Output directory for checkpoints
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        sequence_length (int): Length of video sequences
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
            project="face-tracking",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "num_epochs": num_epochs
            }
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = FaceTrackingDataset(data_dir, sequence_length)
    
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
    model = FaceTrackingModel().to(device)
    criterion = FaceTrackingLoss().to(device)
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
        
        for batch_idx, (frames, targets) in enumerate(tqdm(train_loader)):
            # Move data to device
            frames = frames.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            detection_logits, embeddings, identity_embeddings = model(
                frames,
                sequence_length=sequence_length
            )
            
            # Compute loss
            loss, loss_dict = criterion(
                detection_logits,
                embeddings,
                identity_embeddings,
                targets["labels"],
                targets["boxes"],
                targets["identities"]
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
            for frames, targets in tqdm(val_loader):
                # Move data to device
                frames = frames.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                # Forward pass
                detection_logits, embeddings, identity_embeddings = model(
                    frames,
                    sequence_length=sequence_length
                )
                
                # Compute loss
                loss, loss_dict = criterion(
                    detection_logits,
                    embeddings,
                    identity_embeddings,
                    targets["labels"],
                    targets["boxes"],
                    targets["identities"]
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
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    train_face_tracking(**vars(args)) 