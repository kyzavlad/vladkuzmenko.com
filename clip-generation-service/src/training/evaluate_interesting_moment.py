import os
import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.interesting_moment import InterestingMomentModel
from .train_interesting_moment import InterestingMomentDataset

def evaluate_model(
    model: InterestingMomentModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model (InterestingMomentModel): Model to evaluate
        data_loader (DataLoader): Data loader
        device (torch.device): Device to use
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    model.eval()
    all_logits = []
    all_engagement = []
    all_targets = []
    all_engagement_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
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
            
            # Collect predictions and targets
            all_logits.extend(logits.cpu().numpy())
            all_engagement.extend(engagement.cpu().numpy())
            all_targets.extend(batch["targets"].cpu().numpy())
            all_engagement_targets.extend(
                batch["engagement_targets"].cpu().numpy()
            )
    
    # Convert to numpy arrays
    all_logits = np.array(all_logits)
    all_engagement = np.array(all_engagement)
    all_targets = np.array(all_targets)
    all_engagement_targets = np.array(all_engagement_targets)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        np.argmax(all_logits, axis=1),
        average="binary"
    )
    
    ap = average_precision_score(
        all_targets,
        all_logits[:, 1]
    )
    
    auc = roc_auc_score(
        all_targets,
        all_logits[:, 1]
    )
    
    # Calculate engagement metrics
    engagement_mae = np.mean(np.abs(
        all_engagement - all_engagement_targets
    ))
    
    engagement_rmse = np.sqrt(np.mean(
        (all_engagement - all_engagement_targets) ** 2
    ))
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_precision": ap,
        "auc_roc": auc,
        "engagement_mae": engagement_mae,
        "engagement_rmse": engagement_rmse
    }

def plot_results(
    metrics: Dict[str, float],
    output_dir: str
):
    """
    Plot evaluation results.
    
    Args:
        metrics (Dict[str, float]): Evaluation metrics
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot classification metrics
    plt.figure(figsize=(10, 6))
    classification_metrics = {
        k: v for k, v in metrics.items()
        if k not in ["engagement_mae", "engagement_rmse"]
    }
    
    plt.bar(classification_metrics.keys(), classification_metrics.values())
    plt.title("Classification Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_metrics.png"))
    plt.close()
    
    # Plot engagement metrics
    plt.figure(figsize=(10, 6))
    engagement_metrics = {
        k: v for k, v in metrics.items()
        if k in ["engagement_mae", "engagement_rmse"]
    }
    
    plt.bar(engagement_metrics.keys(), engagement_metrics.values())
    plt.title("Engagement Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "engagement_metrics.png"))
    plt.close()

def evaluate_interesting_moment(
    model_path: str,
    data_dir: str,
    output_dir: str,
    sequence_length: int = 32,
    batch_size: int = 8,
    platform: str = None,
    category: str = None
):
    """
    Evaluate interesting moment detection model.
    
    Args:
        model_path (str): Path to trained model
        data_dir (str): Data directory
        output_dir (str): Output directory for results
        sequence_length (int): Length of video sequences
        batch_size (int): Batch size
        platform (str): Platform filter
        category (str): Content category filter
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = InterestingMomentModel(
        sequence_length=sequence_length
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Create dataset and loader
    dataset = InterestingMomentDataset(
        data_dir,
        sequence_length,
        platform,
        category
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate model
    metrics = evaluate_model(model, loader, device)
    
    # Log results
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Plot results
    plot_results(metrics, output_dir)
    
    # Save results
    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info(f"Results saved to: {results_path}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--platform", type=str, choices=["TikTok", "Instagram", "YouTube"])
    parser.add_argument("--category", type=str)
    
    args = parser.parse_args()
    evaluate_interesting_moment(**vars(args)) 