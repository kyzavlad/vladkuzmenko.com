import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.interesting_moment import InterestingMomentModel
from .train_interesting_moment import train_interesting_moment
from .evaluate_interesting_moment import evaluate_interesting_moment

class ABTestConfig:
    def __init__(
        self,
        name: str,
        model_config: Dict,
        training_config: Dict
    ):
        """
        Configuration for A/B test.
        
        Args:
            name (str): Configuration name
            model_config (Dict): Model configuration
            training_config (Dict): Training configuration
        """
        self.name = name
        self.model_config = model_config
        self.training_config = training_config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "model_config": self.model_config,
            "training_config": self.training_config
        }

def run_ab_test(
    configs: List[ABTestConfig],
    data_dir: str,
    output_dir: str,
    num_runs: int = 5,
    seed: int = 42
):
    """
    Run A/B test comparing different configurations.
    
    Args:
        configs (List[ABTestConfig]): List of configurations to test
        data_dir (str): Data directory
        output_dir (str): Output directory
        num_runs (int): Number of runs per configuration
        seed (int): Random seed
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Run experiments
    results = {}
    
    for config in configs:
        logger.info(f"Testing configuration: {config.name}")
        config_results = []
        
        for run in range(num_runs):
            logger.info(f"Run {run + 1}/{num_runs}")
            
            # Create run directory
            run_dir = os.path.join(
                output_dir,
                f"{config.name}_run_{run + 1}"
            )
            os.makedirs(run_dir, exist_ok=True)
            
            # Train model
            train_interesting_moment(
                data_dir=data_dir,
                output_dir=run_dir,
                **config.training_config
            )
            
            # Evaluate model
            metrics = evaluate_interesting_moment(
                model_path=os.path.join(run_dir, "best_model.pth"),
                data_dir=data_dir,
                output_dir=run_dir,
                **config.training_config
            )
            
            config_results.append(metrics)
        
        # Calculate statistics
        stats_dict = {}
        for metric in config_results[0].keys():
            values = [r[metric] for r in config_results]
            stats_dict[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        results[config.name] = {
            "config": config.to_dict(),
            "results": config_results,
            "statistics": stats_dict
        }
    
    # Save results
    results_path = os.path.join(output_dir, "ab_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Perform statistical tests
    statistical_tests = {}
    
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            config1, config2 = configs[i], configs[j]
            
            for metric in results[config1.name]["results"][0].keys():
                values1 = [
                    r[metric]
                    for r in results[config1.name]["results"]
                ]
                values2 = [
                    r[metric]
                    for r in results[config2.name]["results"]
                ]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(values1, values2)
                
                statistical_tests[f"{config1.name}_vs_{config2.name}_{metric}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
    
    # Save statistical tests
    tests_path = os.path.join(output_dir, "statistical_tests.json")
    with open(tests_path, "w") as f:
        json.dump(statistical_tests, f, indent=2)
    
    logger.info(f"Statistical tests saved to: {tests_path}")
    
    # Plot results
    plot_ab_test_results(results, output_dir)
    
    return results, statistical_tests

def plot_ab_test_results(
    results: Dict,
    output_dir: str
):
    """
    Plot A/B test results.
    
    Args:
        results (Dict): A/B test results
        output_dir (str): Output directory
    """
    # Plot mean metrics
    plt.figure(figsize=(12, 6))
    
    configs = list(results.keys())
    metrics = list(results[configs[0]]["results"][0].keys())
    
    x = np.arange(len(configs))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        means = [
            results[config]["statistics"][metric]["mean"]
            for config in configs
        ]
        stds = [
            results[config]["statistics"][metric]["std"]
            for config in configs
        ]
        
        plt.bar(
            x + i * width,
            means,
            width,
            label=metric,
            yerr=stds,
            capsize=5
        )
    
    plt.xlabel("Configuration")
    plt.ylabel("Metric Value")
    plt.title("A/B Test Results")
    plt.xticks(x + width * (len(metrics) - 1) / 2, configs)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "ab_test_results.png"))
    plt.close()
    
    # Plot individual runs
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for config in configs:
            values = [r[metric] for r in results[config]["results"]]
            plt.plot(values, label=config, marker="o")
        
        plt.xlabel("Run")
        plt.ylabel(metric)
        plt.title(f"{metric} Across Runs")
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{metric}_across_runs.png"))
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Define configurations to test
    configs = [
        ABTestConfig(
            name="baseline",
            model_config={
                "visual_dim": 2048,
                "audio_dim": 512,
                "text_dim": 768,
                "hidden_dim": 512,
                "sequence_length": 32
            },
            training_config={
                "num_epochs": 100,
                "batch_size": 8,
                "learning_rate": 1e-4
            }
        ),
        ABTestConfig(
            name="larger_model",
            model_config={
                "visual_dim": 2048,
                "audio_dim": 512,
                "text_dim": 768,
                "hidden_dim": 1024,
                "sequence_length": 32
            },
            training_config={
                "num_epochs": 100,
                "batch_size": 8,
                "learning_rate": 1e-4
            }
        ),
        ABTestConfig(
            name="longer_sequence",
            model_config={
                "visual_dim": 2048,
                "audio_dim": 512,
                "text_dim": 768,
                "hidden_dim": 512,
                "sequence_length": 64
            },
            training_config={
                "num_epochs": 100,
                "batch_size": 8,
                "learning_rate": 1e-4
            }
        )
    ]
    
    run_ab_test(
        configs,
        args.data_dir,
        args.output_dir,
        args.num_runs,
        args.seed
    ) 