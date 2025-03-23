import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple, Optional
import numpy as np

class FaceTrackingModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,  # face/no-face for detection
        embedding_dim: int = 256
    ):
        """
        Initialize face tracking model.
        
        Args:
            backbone (str): Backbone architecture
            pretrained (bool): Whether to use pretrained weights
            num_classes (int): Number of output classes
            embedding_dim (int): Dimension of face embeddings
        """
        super().__init__()
        
        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add face detection head
        self.detection_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Add face embedding head for tracking
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Add temporal consistency module
        self.temporal_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Add identity persistence module
        self.identity_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features."""
        return self.backbone(x).squeeze(-1).squeeze(-1)
    
    def detect_faces(self, x: torch.Tensor) -> torch.Tensor:
        """Detect faces in images."""
        features = self.extract_features(x)
        return self.detection_head(features)
    
    def compute_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Compute face embeddings."""
        features = self.extract_features(x)
        return self.embedding_head(features)
    
    def forward(
        self,
        x: torch.Tensor,
        sequence_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (B, T, C, H, W)
            sequence_length (Optional[int]): Length of temporal sequence
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Detection logits
                - Face embeddings
                - Identity-persistent embeddings
        """
        batch_size = x.size(0)
        
        if sequence_length is not None:
            # Handle temporal sequence
            x = x.view(-1, *x.shape[2:])  # (B*T, C, H, W)
            
        # Extract features
        features = self.extract_features(x)  # (B*T, backbone_dim)
        
        # Face detection
        detection_logits = self.detection_head(features)
        
        # Face embeddings
        embeddings = self.embedding_head(features)
        
        if sequence_length is not None:
            # Reshape back to temporal sequence
            embeddings = embeddings.view(batch_size, sequence_length, -1)
            
            # Apply temporal consistency
            temporal_embeddings, _ = self.temporal_lstm(embeddings)
            
            # Apply identity persistence
            identity_embeddings = self.identity_head(temporal_embeddings)
            
            # Reshape detection logits
            detection_logits = detection_logits.view(batch_size, sequence_length, -1)
        else:
            identity_embeddings = self.identity_head(embeddings.unsqueeze(1)).squeeze(1)
        
        return detection_logits, embeddings, identity_embeddings
    
    def track_faces(
        self,
        frames: List[np.ndarray],
        threshold: float = 0.5,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Track faces across frames.
        
        Args:
            frames (List[np.ndarray]): List of frames
            threshold (float): Detection threshold
            min_size (Tuple[int, int]): Minimum face size
            
        Returns:
            List[List[Tuple[int, int, int, int]]]: Face bounding boxes for each frame
        """
        device = next(self.parameters()).device
        
        # Convert frames to tensor
        x = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            for frame in frames
        ]).to(device)
        
        # Process sequence
        detection_logits, embeddings, identity_embeddings = self(
            x,
            sequence_length=len(frames)
        )
        
        # Get face detections
        face_probs = torch.softmax(detection_logits, dim=-1)[..., 1]
        face_mask = face_probs > threshold
        
        # Track faces using identity embeddings
        tracked_boxes = []
        prev_embeddings = None
        
        for t in range(len(frames)):
            frame_boxes = []
            
            if face_mask[t]:
                curr_embeddings = identity_embeddings[t]
                
                if prev_embeddings is not None:
                    # Match faces with previous frame
                    similarity = torch.matmul(
                        curr_embeddings,
                        prev_embeddings.transpose(0, 1)
                    )
                    matches = torch.argmax(similarity, dim=1)
                    
                    # Update boxes based on matches
                    for i, match_idx in enumerate(matches):
                        if similarity[i, match_idx] > 0.8:  # Identity threshold
                            frame_boxes.append(tracked_boxes[-1][match_idx])
                
                prev_embeddings = curr_embeddings
            
            tracked_boxes.append(frame_boxes)
        
        return tracked_boxes
    
class FaceTrackingLoss(nn.Module):
    def __init__(
        self,
        detection_weight: float = 1.0,
        embedding_weight: float = 1.0,
        identity_weight: float = 1.0
    ):
        """
        Loss function for face tracking model.
        
        Args:
            detection_weight (float): Weight for detection loss
            embedding_weight (float): Weight for embedding loss
            identity_weight (float): Weight for identity persistence loss
        """
        super().__init__()
        self.detection_weight = detection_weight
        self.embedding_weight = embedding_weight
        self.identity_weight = identity_weight
        
        self.detection_loss = nn.CrossEntropyLoss()
        self.embedding_loss = nn.TripletMarginLoss(margin=0.3)
        self.identity_loss = nn.MSELoss()
    
    def forward(
        self,
        detection_logits: torch.Tensor,
        embeddings: torch.Tensor,
        identity_embeddings: torch.Tensor,
        targets: torch.Tensor,
        positive_pairs: torch.Tensor,
        negative_pairs: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss.
        
        Args:
            detection_logits (torch.Tensor): Detection predictions
            embeddings (torch.Tensor): Face embeddings
            identity_embeddings (torch.Tensor): Identity-persistent embeddings
            targets (torch.Tensor): Detection targets
            positive_pairs (torch.Tensor): Positive pair indices
            negative_pairs (torch.Tensor): Negative pair indices
            
        Returns:
            Tuple[torch.Tensor, dict]: Total loss and loss components
        """
        # Detection loss
        det_loss = self.detection_loss(detection_logits, targets)
        
        # Embedding loss (triplet)
        emb_loss = self.embedding_loss(
            embeddings,
            embeddings[positive_pairs],
            embeddings[negative_pairs]
        )
        
        # Identity persistence loss
        id_loss = self.identity_loss(
            identity_embeddings[:-1],
            identity_embeddings[1:]
        )
        
        # Total loss
        total_loss = (
            self.detection_weight * det_loss +
            self.embedding_weight * emb_loss +
            self.identity_weight * id_loss
        )
        
        return total_loss, {
            "detection": det_loss.item(),
            "embedding": emb_loss.item(),
            "identity": id_loss.item(),
            "total": total_loss.item()
        } 