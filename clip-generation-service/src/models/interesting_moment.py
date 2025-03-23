import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple, Optional, Dict
import numpy as np

class MultimodalAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8):
        """
        Multi-head attention for multimodal features.
        
        Args:
            input_dim (int): Input feature dimension
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            torch.Tensor: Attention output
        """
        batch_size = query.size(0)
        
        # Project inputs
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        
        return out

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        visual_dim: int = 2048,
        audio_dim: int = 512,
        text_dim: int = 768,
        hidden_dim: int = 512
    ):
        """
        Feature extractor for multimodal inputs.
        
        Args:
            visual_dim (int): Visual feature dimension
            audio_dim (int): Audio feature dimension
            text_dim (int): Text feature dimension
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        
        # Visual feature processing
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Audio feature processing
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Text feature processing
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cross-modal attention
        self.cross_attention = MultimodalAttention(hidden_dim)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            visual_features (torch.Tensor): Visual features
            audio_features (torch.Tensor): Audio features
            text_features (torch.Tensor): Text features
            mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            torch.Tensor: Fused features
        """
        # Project features
        v = self.visual_proj(visual_features)
        a = self.audio_proj(audio_features)
        t = self.text_proj(text_features)
        
        # Cross-modal attention
        v_attended = self.cross_attention(v, a, a, mask)
        a_attended = self.cross_attention(a, v, v, mask)
        t_attended = self.cross_attention(t, v, v, mask)
        
        # Concatenate and fuse
        fused = torch.cat([v_attended, a_attended, t_attended], dim=-1)
        fused = self.fusion(fused)
        
        return fused

class InterestingMomentModel(nn.Module):
    def __init__(
        self,
        visual_dim: int = 2048,
        audio_dim: int = 512,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 2,  # interesting/not interesting
        sequence_length: int = 32
    ):
        """
        Model for interesting moment detection.
        
        Args:
            visual_dim (int): Visual feature dimension
            audio_dim (int): Audio feature dimension
            text_dim (int): Text feature dimension
            hidden_dim (int): Hidden layer dimension
            num_classes (int): Number of output classes
            sequence_length (int): Length of temporal sequence
        """
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            visual_dim,
            audio_dim,
            text_dim,
            hidden_dim
        )
        
        # Temporal processing
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Temporal attention
        self.temporal_attention = MultimodalAttention(hidden_dim * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Engagement prediction head
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.sequence_length = sequence_length
    
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            visual_features (torch.Tensor): Visual features
            audio_features (torch.Tensor): Audio features
            text_features (torch.Tensor): Text features
            mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Classification logits
                - Engagement scores
        """
        # Extract and fuse features
        fused_features = self.feature_extractor(
            visual_features,
            audio_features,
            text_features,
            mask
        )
        
        # Process temporal sequence
        lstm_out, _ = self.temporal_lstm(fused_features)
        
        # Apply temporal attention
        attended_features = self.temporal_attention(
            lstm_out,
            lstm_out,
            lstm_out,
            mask
        )
        
        # Classification
        logits = self.classifier(attended_features)
        
        # Engagement prediction
        engagement = self.engagement_head(attended_features)
        
        return logits, engagement
    
    def predict_interesting_moments(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Predict interesting moments in a video.
        
        Args:
            visual_features (torch.Tensor): Visual features
            audio_features (torch.Tensor): Audio features
            text_features (torch.Tensor): Text features
            threshold (float): Classification threshold
            
        Returns:
            List[Dict]: List of interesting moments with timestamps and scores
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            logits, engagement = self(
                visual_features,
                audio_features,
                text_features
            )
            
            # Get predictions
            probs = torch.softmax(logits, dim=-1)[..., 1]
            moments = []
            
            # Find interesting moments
            for t in range(len(probs)):
                if probs[t] > threshold:
                    moments.append({
                        "timestamp": t,
                        "probability": probs[t].item(),
                        "engagement_score": engagement[t].item()
                    })
            
            # Merge nearby moments
            merged_moments = []
            if moments:
                current = moments[0]
                
                for moment in moments[1:]:
                    if moment["timestamp"] - current["timestamp"] <= 2:
                        # Merge moments
                        current["timestamp"] = (
                            current["timestamp"] + moment["timestamp"]
                        ) / 2
                        current["probability"] = max(
                            current["probability"],
                            moment["probability"]
                        )
                        current["engagement_score"] = max(
                            current["engagement_score"],
                            moment["engagement_score"]
                        )
                    else:
                        merged_moments.append(current)
                        current = moment
                
                merged_moments.append(current)
            
            return merged_moments

class InterestingMomentLoss(nn.Module):
    def __init__(
        self,
        classification_weight: float = 1.0,
        engagement_weight: float = 0.5
    ):
        """
        Loss function for interesting moment detection.
        
        Args:
            classification_weight (float): Weight for classification loss
            engagement_weight (float): Weight for engagement loss
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.engagement_weight = engagement_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.engagement_loss = nn.MSELoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        engagement: torch.Tensor,
        targets: torch.Tensor,
        engagement_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss.
        
        Args:
            logits (torch.Tensor): Classification logits
            engagement (torch.Tensor): Engagement predictions
            targets (torch.Tensor): Classification targets
            engagement_targets (torch.Tensor): Engagement targets
            
        Returns:
            Tuple[torch.Tensor, Dict]: Total loss and loss components
        """
        # Classification loss
        cls_loss = self.classification_loss(logits, targets)
        
        # Engagement loss
        eng_loss = self.engagement_loss(engagement, engagement_targets)
        
        # Total loss
        total_loss = (
            self.classification_weight * cls_loss +
            self.engagement_weight * eng_loss
        )
        
        return total_loss, {
            "classification": cls_loss.item(),
            "engagement": eng_loss.item(),
            "total": total_loss.item()
        } 