import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm
import librosa

from .quality_assurance import QualityAssuranceTester, TestConfig
from ..models.interesting_moment import InterestingMomentModel
from .train_interesting_moment import InterestingMomentDataset

@dataclass
class SilentRemovalConfig:
    """Configuration for silent segment removal evaluation."""
    energy_threshold: float = 0.01
    min_silence_duration: float = 0.5
    max_silence_duration: float = 5.0
    save_visualizations: bool = True
    visualization_dir: Optional[str] = None

class SilentRemovalEvaluator:
    """Evaluates silent segment removal performance."""
    
    def __init__(
        self,
        model: InterestingMomentModel,
        test_config: TestConfig,
        removal_config: SilentRemovalConfig
    ):
        """
        Initialize silent removal evaluator.
        
        Args:
            model (InterestingMomentModel): Model to evaluate
            test_config (TestConfig): Test configuration
            removal_config (SilentRemovalConfig): Removal configuration
        """
        self.model = model
        self.test_config = test_config
        self.removal_config = removal_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize quality assurance tester
        self.qa_tester = QualityAssuranceTester(model, test_config)
        
        # Create visualization directory if needed
        if removal_config.save_visualizations and removal_config.visualization_dir:
            os.makedirs(removal_config.visualization_dir, exist_ok=True)
    
    def evaluate_removal(
        self,
        data_dir: str,
        sequence_length: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate silent segment removal performance.
        
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
        total_segments = 0
        total_silent_segments = 0
        correctly_removed = 0
        incorrectly_removed = 0
        missed_silent_segments = 0
        
        # Process batches
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating silent removal")):
            # Get predictions
            with torch.no_grad():
                predictions = self.model(batch)
            
            # Process each sequence in batch
            for seq_idx in range(len(batch)):
                # Get audio segments
                audio = batch["audio"][seq_idx]
                segment_predictions = predictions["silent_segments"][seq_idx]
                
                # Get ground truth silent segments
                ground_truth = self._detect_silent_segments(audio)
                
                # Update metrics
                total_segments += len(ground_truth)
                total_silent_segments += sum(1 for seg in ground_truth if seg["is_silent"])
                
                # Compare predictions with ground truth
                for pred in segment_predictions:
                    # Find matching ground truth segment
                    matched = False
                    for gt in ground_truth:
                        if self._segments_overlap(pred, gt):
                            matched = True
                            if gt["is_silent"]:
                                correctly_removed += 1
                            else:
                                incorrectly_removed += 1
                            break
                    
                    if not matched:
                        missed_silent_segments += 1
                
                # Save visualization if needed
                if (self.removal_config.save_visualizations and
                    self.removal_config.visualization_dir and
                    batch_idx == 0 and seq_idx == 0):
                    self._save_visualization(
                        audio,
                        ground_truth,
                        segment_predictions,
                        seq_idx
                    )
        
        # Compute metrics
        precision = correctly_removed / (correctly_removed + incorrectly_removed) if (correctly_removed + incorrectly_removed) > 0 else 0
        recall = correctly_removed / total_silent_segments if total_silent_segments > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "total_segments": total_segments,
            "total_silent_segments": total_silent_segments,
            "correctly_removed": correctly_removed,
            "incorrectly_removed": incorrectly_removed,
            "missed_silent_segments": missed_silent_segments,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _detect_silent_segments(
        self,
        audio: torch.Tensor
    ) -> List[Dict]:
        """
        Detect silent segments in audio.
        
        Args:
            audio (torch.Tensor): Audio waveform
            
        Returns:
            List[Dict]: List of segments with silent flag
        """
        # Convert to numpy
        audio = audio.cpu().numpy()
        
        # Compute energy
        energy = librosa.feature.rms(y=audio)[0]
        
        # Find silent segments
        silent_frames = energy < self.removal_config.energy_threshold
        
        # Group consecutive silent frames
        segments = []
        start_frame = None
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and start_frame is None:
                start_frame = i
            elif not is_silent and start_frame is not None:
                duration = (i - start_frame) / librosa.get_samplerate()
                if (self.removal_config.min_silence_duration <= duration <=
                    self.removal_config.max_silence_duration):
                    segments.append({
                        "start": start_frame,
                        "end": i,
                        "duration": duration,
                        "is_silent": True
                    })
                start_frame = None
        
        # Handle last segment
        if start_frame is not None:
            duration = (len(silent_frames) - start_frame) / librosa.get_samplerate()
            if (self.removal_config.min_silence_duration <= duration <=
                self.removal_config.max_silence_duration):
                segments.append({
                    "start": start_frame,
                    "end": len(silent_frames),
                    "duration": duration,
                    "is_silent": True
                })
        
        return segments
    
    def _segments_overlap(
        self,
        seg1: Dict,
        seg2: Dict
    ) -> bool:
        """
        Check if two segments overlap.
        
        Args:
            seg1 (Dict): First segment
            seg2 (Dict): Second segment
            
        Returns:
            bool: Whether segments overlap
        """
        return (seg1["start"] <= seg2["end"] and seg2["start"] <= seg1["end"])
    
    def _save_visualization(
        self,
        audio: torch.Tensor,
        ground_truth: List[Dict],
        predictions: List[Dict],
        seq_idx: int
    ):
        """
        Save visualization of silent segment removal.
        
        Args:
            audio (torch.Tensor): Audio waveform
            ground_truth (List[Dict]): Ground truth segments
            predictions (List[Dict]): Predicted segments
            seq_idx (int): Sequence index
        """
        # Convert audio to numpy
        audio = audio.cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot audio waveform
        plt.plot(audio, label="Audio")
        
        # Plot ground truth segments
        for seg in ground_truth:
            if seg["is_silent"]:
                plt.axvspan(
                    seg["start"],
                    seg["end"],
                    color="red",
                    alpha=0.3,
                    label="Ground Truth Silent"
                )
        
        # Plot predicted segments
        for seg in predictions:
            plt.axvspan(
                seg["start"],
                seg["end"],
                color="blue",
                alpha=0.3,
                label="Predicted Silent"
            )
        
        plt.title(f"Silent Segment Removal - Sequence {seq_idx}")
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")
        plt.legend()
        
        # Save figure
        plt.savefig(os.path.join(
            self.removal_config.visualization_dir,
            f"silent_removal_seq_{seq_idx}.png"
        ))
        plt.close()
    
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
                "energy_threshold": self.removal_config.energy_threshold,
                "min_silence_duration": self.removal_config.min_silence_duration,
                "max_silence_duration": self.removal_config.max_silence_duration
            },
            "metrics": metrics
        }
        
        # Save results
        results_path = os.path.join(
            self.test_config.output_dir,
            "silent_removal_results.json"
        )
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to: {results_path}")
        
        # Log metrics
        self.logger.info("\nSilent Removal Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")

def main():
    """Main function for evaluating silent segment removal."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--energy_threshold", type=float, default=0.01)
    parser.add_argument("--min_silence_duration", type=float, default=0.5)
    parser.add_argument("--max_silence_duration", type=float, default=5.0)
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
    
    removal_config = SilentRemovalConfig(
        energy_threshold=args.energy_threshold,
        min_silence_duration=args.min_silence_duration,
        max_silence_duration=args.max_silence_duration,
        save_visualizations=args.save_visualizations,
        visualization_dir=args.visualization_dir
    )
    
    # Create evaluator
    evaluator = SilentRemovalEvaluator(model, test_config, removal_config)
    
    # Run evaluation
    metrics = evaluator.evaluate_removal(
        args.data_dir,
        args.sequence_length
    )

if __name__ == "__main__":
    main() 