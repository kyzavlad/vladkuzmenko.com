import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error
)
import json
import os
from datetime import datetime

@dataclass
class TestConfig:
    """Configuration for quality assurance testing."""
    output_dir: str
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_predictions: bool = True
    save_visualizations: bool = True

class QualityAssuranceTester:
    """Framework for quality assurance testing."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TestConfig
    ):
        """
        Initialize quality assurance tester.
        
        Args:
            model (nn.Module): Model to test
            config (TestConfig): Test configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def evaluate_face_tracking(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate face tracking performance.
        
        Args:
            data_loader (DataLoader): Data loader
            save_results (bool): Whether to save results
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        tracking_consistency = []
        
        with torch.no_grad():
            for batch in data_loader:
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
                
                # Collect predictions and targets
                all_predictions.extend(outputs[0].cpu().numpy())
                all_targets.extend(batch["targets"].cpu().numpy())
                
                # Compute tracking consistency
                tracking_consistency.extend(
                    self._compute_tracking_consistency(outputs[0])
                )
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets,
            np.argmax(all_predictions, axis=1),
            average="binary"
        )
        
        ap = average_precision_score(
            all_targets,
            all_predictions[:, 1]
        )
        
        auc = roc_auc_score(
            all_targets,
            all_predictions[:, 1]
        )
        
        # Calculate tracking consistency metrics
        consistency_mean = np.mean(tracking_consistency)
        consistency_std = np.std(tracking_consistency)
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_precision": ap,
            "auc_roc": auc,
            "tracking_consistency_mean": consistency_mean,
            "tracking_consistency_std": consistency_std
        }
        
        if save_results:
            self._save_results(metrics, "face_tracking")
        
        return metrics
    
    def evaluate_engagement_prediction(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate engagement prediction performance.
        
        Args:
            data_loader (DataLoader): Data loader
            save_results (bool): Whether to save results
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
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
                
                # Collect predictions and targets
                all_predictions.extend(outputs[1].cpu().numpy())
                all_targets.extend(batch["engagement_targets"].cpu().numpy())
        
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }
        
        if save_results:
            self._save_results(metrics, "engagement_prediction")
        
        return metrics
    
    def evaluate_silent_removal(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate silent segment removal performance.
        
        Args:
            data_loader (DataLoader): Data loader
            save_results (bool): Whether to save results
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
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
                
                # Collect predictions and targets
                all_predictions.extend(outputs[2].cpu().numpy())
                all_targets.extend(batch["silent_targets"].cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets,
            np.argmax(all_predictions, axis=1),
            average="binary"
        )
        
        ap = average_precision_score(
            all_targets,
            all_predictions[:, 1]
        )
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_precision": ap
        }
        
        if save_results:
            self._save_results(metrics, "silent_removal")
        
        return metrics
    
    def _compute_tracking_consistency(
        self,
        predictions: torch.Tensor
    ) -> List[float]:
        """
        Compute tracking consistency between consecutive frames.
        
        Args:
            predictions (torch.Tensor): Model predictions
            
        Returns:
            List[float]: Consistency scores
        """
        consistency_scores = []
        
        # Convert predictions to numpy
        pred_np = predictions.cpu().numpy()
        
        # Compute consistency between consecutive frames
        for i in range(len(pred_np) - 1):
            # Compare predictions
            consistency = np.mean(
                np.abs(pred_np[i + 1] - pred_np[i])
            )
            consistency_scores.append(1 - consistency)
        
        return consistency_scores
    
    def _save_results(
        self,
        metrics: Dict[str, float],
        test_type: str
    ):
        """
        Save test results.
        
        Args:
            metrics (Dict[str, float]): Test metrics
            test_type (str): Type of test
        """
        # Create results dictionary
        results = {
            "test_type": test_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        # Save results
        results_path = os.path.join(
            self.config.output_dir,
            f"{test_type}_results.json"
        )
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to: {results_path}")
        
        # Log metrics
        self.logger.info(f"\n{test_type} Results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}") 