import os
import torch
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm

from .quality_assurance import QualityAssuranceTester, TestConfig
from ..models.interesting_moment import InterestingMomentModel
from .train_interesting_moment import InterestingMomentDataset

@dataclass
class ContentTypeConfig:
    """Configuration for content type testing."""
    name: str
    platform: Optional[str] = None
    category: Optional[str] = None
    min_samples: int = 100
    max_samples: int = 1000

class DiverseContentTester:
    """Tests model performance on diverse content types."""
    
    def __init__(
        self,
        model: InterestingMomentModel,
        test_config: TestConfig,
        content_types: List[ContentTypeConfig]
    ):
        """
        Initialize diverse content tester.
        
        Args:
            model (InterestingMomentModel): Model to test
            test_config (TestConfig): Test configuration
            content_types (List[ContentTypeConfig]): Content types to test
        """
        self.model = model
        self.test_config = test_config
        self.content_types = content_types
        self.logger = logging.getLogger(__name__)
        
        # Initialize quality assurance tester
        self.qa_tester = QualityAssuranceTester(model, test_config)
    
    def test_content_types(
        self,
        data_dir: str,
        sequence_length: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        Test model on different content types.
        
        Args:
            data_dir (str): Data directory
            sequence_length (int): Length of video sequences
            
        Returns:
            Dict[str, Dict[str, float]]: Results for each content type
        """
        results = {}
        
        for content_type in tqdm(self.content_types, desc="Testing content types"):
            self.logger.info(f"\nTesting {content_type.name}")
            
            # Create dataset for content type
            dataset = InterestingMomentDataset(
                data_dir,
                sequence_length,
                content_type.platform,
                content_type.category
            )
            
            # Filter dataset size if needed
            if len(dataset) > content_type.max_samples:
                dataset = torch.utils.data.Subset(
                    dataset,
                    range(content_type.max_samples)
                )
            elif len(dataset) < content_type.min_samples:
                self.logger.warning(
                    f"Insufficient samples for {content_type.name}: "
                    f"{len(dataset)} < {content_type.min_samples}"
                )
                continue
            
            # Create data loader
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.test_config.batch_size,
                num_workers=self.test_config.num_workers,
                pin_memory=True
            )
            
            # Run tests
            face_tracking_metrics = self.qa_tester.evaluate_face_tracking(loader)
            engagement_metrics = self.qa_tester.evaluate_engagement_prediction(loader)
            silent_removal_metrics = self.qa_tester.evaluate_silent_removal(loader)
            
            # Combine results
            results[content_type.name] = {
                "face_tracking": face_tracking_metrics,
                "engagement_prediction": engagement_metrics,
                "silent_removal": silent_removal_metrics,
                "num_samples": len(dataset)
            }
        
        # Save overall results
        self._save_overall_results(results)
        
        return results
    
    def _save_overall_results(
        self,
        results: Dict[str, Dict[str, float]]
    ):
        """
        Save overall test results.
        
        Args:
            results (Dict[str, Dict[str, float]]): Test results
        """
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "content_types": results,
            "averages": self._compute_averages(results)
        }
        
        # Save summary
        summary_path = os.path.join(
            self.test_config.output_dir,
            "diverse_content_summary.json"
        )
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved summary to: {summary_path}")
        
        # Log summary
        self.logger.info("\nOverall Results:")
        for metric, value in summary["averages"].items():
            self.logger.info(f"{metric}: {value:.4f}")
    
    def _compute_averages(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute average metrics across content types.
        
        Args:
            results (Dict[str, Dict[str, float]]): Test results
            
        Returns:
            Dict[str, float]: Average metrics
        """
        averages = {}
        
        # Collect metrics
        metrics_by_type = {}
        for content_type, content_results in results.items():
            for test_type, metrics in content_results.items():
                if test_type not in metrics_by_type:
                    metrics_by_type[test_type] = {}
                for metric, value in metrics.items():
                    if metric not in metrics_by_type[test_type]:
                        metrics_by_type[test_type][metric] = []
                    metrics_by_type[test_type][metric].append(value)
        
        # Compute averages
        for test_type, metrics in metrics_by_type.items():
            for metric, values in metrics.items():
                averages[f"{test_type}_{metric}"] = np.mean(values)
        
        return averages

def main():
    """Main function for testing diverse content."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model = InterestingMomentModel().to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    
    # Define content types to test
    content_types = [
        ContentTypeConfig(
            name="TikTok_Entertainment",
            platform="TikTok",
            category="entertainment"
        ),
        ContentTypeConfig(
            name="TikTok_Education",
            platform="TikTok",
            category="education"
        ),
        ContentTypeConfig(
            name="Instagram_Reels",
            platform="Instagram",
            category="reels"
        ),
        ContentTypeConfig(
            name="YouTube_Shorts",
            platform="YouTube",
            category="shorts"
        ),
        ContentTypeConfig(
            name="YouTube_LongForm",
            platform="YouTube",
            category="long_form"
        )
    ]
    
    # Create test configuration
    test_config = TestConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create tester
    tester = DiverseContentTester(model, test_config, content_types)
    
    # Run tests
    results = tester.test_content_types(
        args.data_dir,
        args.sequence_length
    )

if __name__ == "__main__":
    main() 