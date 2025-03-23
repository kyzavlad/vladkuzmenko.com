#!/usr/bin/env python3
"""
Face Tracking Benchmark

This script benchmarks various performance optimization strategies for face tracking.
It compares different sampling strategies, batch processing configurations, and GPU usage.

Usage:
    python benchmark_face_tracking.py --video_path VIDEO_PATH
"""

import os
import cv2
import time
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from tabulate import tabulate

from app.clip_generation.services.face_tracking_manager import FaceTrackingManager
from app.clip_generation.services.face_tracking_optimizer import SamplingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Tracking Benchmark")
    
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory for benchmark results")
    parser.add_argument("--num_frames", type=int, default=300,
                        help="Number of frames to process (0 for all)")
    parser.add_argument("--skip_gpu", action="store_true",
                        help="Skip GPU tests")
    parser.add_argument("--plot_results", action="store_true",
                        help="Generate performance plots")
    
    return parser.parse_args()


def benchmark_configuration(
    video_path: str,
    sampling_strategy: str,
    batch_size: int = 1,
    worker_threads: int = 0,
    use_gpu: bool = False,
    num_frames: int = 300
) -> Dict[str, Any]:
    """
    Benchmark a specific configuration.
    
    Args:
        video_path: Path to video file
        sampling_strategy: Sampling strategy to use
        batch_size: Batch size for processing
        worker_threads: Number of worker threads
        use_gpu: Whether to use GPU
        num_frames: Number of frames to process
        
    Returns:
        Dictionary of benchmark results
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {}
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Limit number of frames if specified
    if num_frames > 0:
        frames_to_process = min(num_frames, total_frames)
    else:
        frames_to_process = total_frames
    
    # Initialize face tracking manager
    tracker = FaceTrackingManager(
        detection_interval=5,
        recognition_interval=15,
        sampling_strategy=sampling_strategy,
        batch_size=batch_size,
        worker_threads=worker_threads,
        use_gpu=use_gpu
    )
    
    # Collect statistics
    processing_times = []
    face_counts = []
    frames_processed = 0
    frames_skipped = 0
    
    try:
        # Process frames
        frame_count = 0
        progress_bar = tqdm(total=frames_to_process, 
                          desc=f"Strategy: {sampling_strategy}, Batch: {batch_size}x{worker_threads}, GPU: {use_gpu}")
        
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame and time it
            start_time = time.time()
            tracked_faces = tracker.process_frame(frame)
            processing_time = time.time() - start_time
            
            # Collect statistics
            processing_times.append(processing_time)
            face_counts.append(len(tracked_faces))
            
            # Update progress
            frame_count += 1
            progress_bar.update(1)
            
        progress_bar.close()
        
        # Get tracker statistics
        if hasattr(tracker, 'stats'):
            frames_processed = tracker.stats.get("frames_processed", 0)
            frames_skipped = tracker.stats.get("frames_skipped", 0)
            detection_times = tracker.stats.get("detection_time", [])
        else:
            detection_times = []
    
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cap.release()
        tracker.reset()
    
    # Calculate statistics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        fps_avg = 1.0 / avg_time if avg_time > 0 else 0
        
        if detection_times:
            avg_detection_time = sum(detection_times) / len(detection_times)
        else:
            avg_detection_time = 0
            
        avg_faces = sum(face_counts) / len(face_counts) if face_counts else 0
        processed_pct = frames_processed / frame_count * 100 if frame_count > 0 else 0
    else:
        avg_time = max_time = min_time = fps_avg = avg_detection_time = avg_faces = processed_pct = 0
    
    # Return benchmark results
    return {
        "config": {
            "sampling_strategy": sampling_strategy,
            "batch_size": batch_size,
            "worker_threads": worker_threads,
            "use_gpu": use_gpu
        },
        "stats": {
            "frames_total": frame_count,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "processed_pct": processed_pct,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "fps": fps_avg,
            "avg_detection_time": avg_detection_time,
            "avg_faces": avg_faces
        },
        "times": processing_times,
        "face_counts": face_counts
    }


def run_benchmarks(args):
    """Run all benchmark configurations."""
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define configurations to benchmark
    configurations = []
    
    # Sampling strategies (CPU)
    for strategy in ["uniform", "adaptive", "keyframe", "motion"]:
        configurations.append({
            "sampling_strategy": strategy,
            "batch_size": 1,
            "worker_threads": 0,
            "use_gpu": False
        })
    
    # Batch processing (CPU)
    for batch_size, workers in [(2, 2), (4, 2), (4, 4)]:
        configurations.append({
            "sampling_strategy": "uniform",
            "batch_size": batch_size,
            "worker_threads": workers,
            "use_gpu": False
        })
    
    # GPU configurations (if available and not skipped)
    if not args.skip_gpu:
        # Check if GPU is available
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except:
            has_gpu = False
        
        if has_gpu:
            # GPU with different strategies
            for strategy in ["uniform", "adaptive"]:
                configurations.append({
                    "sampling_strategy": strategy,
                    "batch_size": 1,
                    "worker_threads": 0,
                    "use_gpu": True
                })
            
            # GPU with batch processing
            configurations.append({
                "sampling_strategy": "uniform",
                "batch_size": 4,
                "worker_threads": 2,
                "use_gpu": True
            })
    
    # Run benchmarks
    results = []
    for config in configurations:
        logger.info(f"Benchmarking: {config}")
        result = benchmark_configuration(
            video_path=args.video_path,
            sampling_strategy=config["sampling_strategy"],
            batch_size=config["batch_size"],
            worker_threads=config["worker_threads"],
            use_gpu=config["use_gpu"],
            num_frames=args.num_frames
        )
        results.append(result)
    
    # Summarize results
    summarize_results(results, args.output_dir)
    
    # Plot results if requested
    if args.plot_results:
        plot_results(results, args.output_dir)


def summarize_results(results: List[Dict[str, Any]], output_dir: str):
    """Summarize benchmark results."""
    if not results:
        logger.error("No benchmark results to summarize")
        return
    
    # Prepare results table
    headers = [
        "Strategy", "Batch", "Workers", "GPU", 
        "FPS", "Avg Time (ms)", "Processed %", "Avg Faces"
    ]
    
    rows = []
    for result in results:
        config = result["config"]
        stats = result["stats"]
        
        rows.append([
            config["sampling_strategy"],
            config["batch_size"],
            config["worker_threads"],
            "Yes" if config["use_gpu"] else "No",
            f"{stats['fps']:.1f}",
            f"{stats['avg_time'] * 1000:.1f}",
            f"{stats['processed_pct']:.1f}%",
            f"{stats['avg_faces']:.1f}"
        ])
    
    # Sort by FPS (descending)
    rows.sort(key=lambda x: float(x[4]), reverse=True)
    
    # Print table
    table = tabulate(rows, headers=headers, tablefmt="grid")
    print("\nBenchmark Results:\n")
    print(table)
    
    # Save results to file
    with open(os.path.join(output_dir, "benchmark_results.txt"), "w") as f:
        f.write("Face Tracking Performance Benchmark\n")
        f.write("=================================\n\n")
        f.write(f"Video: {args.video_path}\n")
        f.write(f"Frames: {args.num_frames}\n\n")
        f.write(table)


def plot_results(results: List[Dict[str, Any]], output_dir: str):
    """Generate plots of benchmark results."""
    if not results:
        logger.error("No benchmark results to plot")
        return
    
    # Plot FPS comparison
    plt.figure(figsize=(12, 6))
    
    # Extract strategies and FPS values
    strategies = []
    fps_values = []
    colors = []
    
    for result in results:
        config = result["config"]
        stats = result["stats"]
        
        # Create label
        if config["worker_threads"] > 0:
            label = f"{config['sampling_strategy']}\nBatch: {config['batch_size']}x{config['worker_threads']}"
        else:
            label = config["sampling_strategy"]
        
        if config["use_gpu"]:
            label += "\n(GPU)"
            color = 'green'
        else:
            color = 'blue'
        
        strategies.append(label)
        fps_values.append(stats["fps"])
        colors.append(color)
    
    # Sort by FPS
    sorted_data = sorted(zip(strategies, fps_values, colors), key=lambda x: x[1], reverse=True)
    strategies, fps_values, colors = zip(*sorted_data)
    
    # Create bar chart
    bars = plt.bar(strategies, fps_values, color=colors)
    
    # Add FPS values on top of bars
    for bar, fps in zip(bars, fps_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{fps:.1f}",
            ha='center',
            fontweight='bold'
        )
    
    plt.title("Face Tracking Performance Comparison (FPS)")
    plt.ylabel("Frames Per Second")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "fps_comparison.png"))
    
    # Plot processing percentage for adaptive strategies
    adaptive_results = [r for r in results if r["config"]["sampling_strategy"] in ["adaptive", "keyframe", "motion"]]
    
    if adaptive_results:
        plt.figure(figsize=(10, 5))
        
        labels = []
        processed_pct = []
        
        for result in adaptive_results:
            config = result["config"]
            stats = result["stats"]
            
            label = f"{config['sampling_strategy']}"
            if config["use_gpu"]:
                label += "\n(GPU)"
            
            labels.append(label)
            processed_pct.append(stats["processed_pct"])
        
        plt.bar(labels, processed_pct, color='orange')
        plt.title("Frame Processing Percentage by Sampling Strategy")
        plt.ylabel("Processed Frames (%)")
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "processing_percentage.png"))
    
    logger.info(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmarks(args) 