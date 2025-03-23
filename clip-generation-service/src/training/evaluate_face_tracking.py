import os
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm
import cv2

from .quality_assurance import QualityAssuranceTester, TestConfig
from ..models.interesting_moment import InterestingMomentModel
from .train_interesting_moment import InterestingMomentDataset

@dataclass
class FaceTrackingConfig:
    """Configuration for face tracking evaluation."""
    iou_threshold: float = 0.5
    consistency_threshold: float = 0.8
    min_face_size: int = 20
    max_face_size: int = 200
    save_visualizations: bool = True
    visualization_dir: Optional[str] = None

class FaceTrackingEvaluator:
    """Evaluates face tracking consistency."""
    
    def __init__(
        self,
        model: InterestingMomentModel,
        test_config: TestConfig,
        tracking_config: FaceTrackingConfig
    ):
        """
        Initialize face tracking evaluator.
        
        Args:
            model (InterestingMomentModel): Model to evaluate
            test_config (TestConfig): Test configuration
            tracking_config (FaceTrackingConfig): Tracking configuration
        """
        self.model = model
        self.test_config = test_config
        self.tracking_config = tracking_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize quality assurance tester
        self.qa_tester = QualityAssuranceTester(model, test_config)
        
        # Create visualization directory if needed
        if tracking_config.save_visualizations and tracking_config.visualization_dir:
            os.makedirs(tracking_config.visualization_dir, exist_ok=True)
    
    def evaluate_tracking(
        self,
        data_dir: str,
        sequence_length: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate face tracking consistency.
        
        Args:
            data_dir (str): Data directory
            sequence_length (int): Length of video sequences
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Create dataset
        dataset = InterestingMomentDataset(
            data_dir,
            sequence_length
        )
        
        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.test_config.batch_size,
            num_workers=self.test_config.num_workers,
            pin_memory=True
        )
        
        # Initialize metrics
        total_frames = 0
        total_faces = 0
        consistent_tracks = 0
        total_tracks = 0
        iou_scores = []
        
        # Process batches
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating tracking")):
            # Get predictions
            with torch.no_grad():
                predictions = self.model(batch)
            
            # Process each sequence in batch
            for seq_idx in range(len(batch)):
                # Get face detections
                faces = predictions["faces"][seq_idx]
                
                # Process each frame
                for frame_idx in range(len(faces)):
                    frame_faces = faces[frame_idx]
                    
                    # Skip if no faces detected
                    if len(frame_faces) == 0:
                        continue
                    
                    total_frames += 1
                    total_faces += len(frame_faces)
                    
                    # Process each face
                    for face_idx, face in enumerate(frame_faces):
                        # Check face size
                        face_size = max(face[2] - face[0], face[3] - face[1])
                        if not (self.tracking_config.min_face_size <= face_size <= self.tracking_config.max_face_size):
                            continue
                        
                        # Get face in next frame
                        if frame_idx < len(faces) - 1:
                            next_frame_faces = faces[frame_idx + 1]
                            
                            # Find matching face
                            best_iou = 0
                            best_match = None
                            
                            for next_face in next_frame_faces:
                                iou = self._compute_iou(face, next_face)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match = next_face
                            
                            # Update metrics
                            if best_match is not None:
                                iou_scores.append(best_iou)
                                
                                if best_iou >= self.tracking_config.iou_threshold:
                                    consistent_tracks += 1
                                total_tracks += 1
                        
                        # Save visualization if needed
                        if (self.tracking_config.save_visualizations and
                            self.tracking_config.visualization_dir and
                            batch_idx == 0 and seq_idx == 0):
                            self._save_face_visualization(
                                batch["frames"][seq_idx][frame_idx],
                                face,
                                frame_idx,
                                face_idx
                            )
        
        # Compute metrics
        metrics = {
            "total_frames": total_frames,
            "total_faces": total_faces,
            "faces_per_frame": total_faces / total_frames if total_frames > 0 else 0,
            "consistent_tracks": consistent_tracks,
            "total_tracks": total_tracks,
            "tracking_consistency": consistent_tracks / total_tracks if total_tracks > 0 else 0,
            "mean_iou": np.mean(iou_scores) if iou_scores else 0,
            "std_iou": np.std(iou_scores) if iou_scores else 0
        }
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _compute_iou(
        self,
        box1: torch.Tensor,
        box2: torch.Tensor
    ) -> float:
        """
        Compute IoU between two bounding boxes.
        
        Args:
            box1 (torch.Tensor): First bounding box [x1, y1, x2, y2]
            box2 (torch.Tensor): Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU score
        """
        # Convert to numpy
        box1 = box1.cpu().numpy()
        box2 = box2.cpu().numpy()
        
        # Get intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Compute intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Compute union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Compute IoU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def _save_face_visualization(
        self,
        frame: torch.Tensor,
        face: torch.Tensor,
        frame_idx: int,
        face_idx: int
    ):
        """
        Save face visualization.
        
        Args:
            frame (torch.Tensor): Frame image
            face (torch.Tensor): Face bounding box
            frame_idx (int): Frame index
            face_idx (int): Face index
        """
        # Convert frame to numpy
        frame = frame.cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        
        # Convert face to numpy
        face = face.cpu().numpy()
        
        # Draw bounding box
        cv2.rectangle(
            frame,
            (int(face[0]), int(face[1])),
            (int(face[2]), int(face[3])),
            (0, 255, 0),
            2
        )
        
        # Save image
        image_path = os.path.join(
            self.tracking_config.visualization_dir,
            f"frame_{frame_idx}_face_{face_idx}.jpg"
        )
        cv2.imwrite(image_path, frame)
    
    def _save_results(self, metrics: Dict[str, float]):
        """
        Save evaluation results.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics
        """
        # Create results dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "iou_threshold": self.tracking_config.iou_threshold,
                "consistency_threshold": self.tracking_config.consistency_threshold,
                "min_face_size": self.tracking_config.min_face_size,
                "max_face_size": self.tracking_config.max_face_size
            },
            "metrics": metrics
        }
        
        # Save results
        results_path = os.path.join(
            self.test_config.output_dir,
            "face_tracking_results.json"
        )
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to: {results_path}")
        
        # Log metrics
        self.logger.info("\nFace Tracking Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")

def main():
    """Main function for evaluating face tracking."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--consistency_threshold", type=float, default=0.8)
    parser.add_argument("--min_face_size", type=int, default=20)
    parser.add_argument("--max_face_size", type=int, default=200)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--visualization_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model = InterestingMomentModel().to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    
    # Create configurations
    test_config = TestConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    tracking_config = FaceTrackingConfig(
        iou_threshold=args.iou_threshold,
        consistency_threshold=args.consistency_threshold,
        min_face_size=args.min_face_size,
        max_face_size=args.max_face_size,
        save_visualizations=args.save_visualizations,
        visualization_dir=args.visualization_dir
    )
    
    # Create evaluator
    evaluator = FaceTrackingEvaluator(model, test_config, tracking_config)
    
    # Run evaluation
    metrics = evaluator.evaluate_tracking(
        args.data_dir,
        args.sequence_length
    )

if __name__ == "__main__":
    main() 