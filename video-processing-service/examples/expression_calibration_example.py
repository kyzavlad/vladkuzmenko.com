#!/usr/bin/env python3
"""
Expression Range Calibration Example

This script demonstrates the Expression Range Calibration feature of the Face Modeling
component, showing how facial expressions are calibrated and adjusted based on
individual facial morphology.

Examples:
    Basic usage:
    python expression_calibration_example.py --image path/to/face/image.jpg

    Using a video file:
    python expression_calibration_example.py --video path/to/face/video.mp4

    Compare expressions:
    python expression_calibration_example.py --image path/to/face/image.jpg --compare-expressions
"""

import os
import sys
import json
import time
import logging
import argparse
import asyncio
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

# Add the parent directory to sys.path to allow importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.avatar.face_modeling import FaceModeling, FaceModelingResult

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("expression_calibration_example")

def display_expression_data(result: FaceModelingResult):
    """
    Display detailed information about the expression calibration data.
    
    Args:
        result: FaceModelingResult containing expression calibration data
    """
    if not result.expression_calibration_data:
        print("\nNo expression calibration data available.")
        return
    
    print("\n=== Expression Calibration Data ===")
    
    # Display meta information
    if "meta" in result.expression_calibration_data:
        meta = result.expression_calibration_data["meta"]
        print(f"\nCalibration Quality: {meta.get('calibration_quality', 'N/A'):.2f}")
        print(f"Expression Count: {meta.get('expression_count', 'N/A')}")
        print(f"Vertex Count: {meta.get('vertex_count', 'N/A')}")
    
    # Display available expressions and their ranges
    if "intensity_ranges" in result.expression_calibration_data:
        ranges = result.expression_calibration_data["intensity_ranges"]
        print("\nExpression Intensity Ranges:")
        for expr, range_values in ranges.items():
            print(f"  {expr}: {range_values[0]:.2f} to {range_values[1]:.2f}")
    
    # Display muscle groups for expressions
    if "muscle_groups" in result.expression_calibration_data:
        print("\nExpression Muscle Groups:")
        muscle_groups = result.expression_calibration_data["muscle_groups"]
        for expr, groups in list(muscle_groups.items())[:3]:  # Show first 3 expressions
            primary = ", ".join(groups.get("primary", []))
            secondary = ", ".join(groups.get("secondary", []))
            print(f"  {expr}:")
            print(f"    Primary: {primary}")
            print(f"    Secondary: {secondary}")
        
        if len(muscle_groups) > 3:
            print(f"  ... and {len(muscle_groups) - 3} more expressions")
    
    # Display interaction matrix (compatibility between expressions)
    if "interaction_matrix" in result.expression_calibration_data:
        print("\nExpression Compatibility Matrix (sample):")
        matrix = result.expression_calibration_data["interaction_matrix"]
        expressions = list(matrix.keys())[:3]  # Show first 3 expressions
        
        # Print header
        header = "Expression".ljust(12)
        for expr in expressions:
            header += expr.ljust(10)
        print("  " + header)
        
        # Print rows
        for expr1 in expressions:
            row = expr1.ljust(12)
            for expr2 in expressions:
                compatibility = matrix[expr1].get(expr2, 0.0)
                row += f"{compatibility:.2f}".ljust(10)
            print("  " + row)

async def analyze_expressions(face_modeling: FaceModeling, image_path: str, output_dir: str):
    """
    Analyze expressions for a face and save visualizations.
    
    Args:
        face_modeling: FaceModeling instance
        image_path: Path to the input image
        output_dir: Directory to save output files
    """
    logger.info(f"Analyzing expressions for image: {image_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 3D face model with expression calibration
    start_time = time.time()
    config = {
        "expression_calibration_enabled": True,
        "detail_level": "high",
        "identity_verification": True
    }
    
    face_modeling.config.update(config)
    result = await face_modeling.generate_from_image(image_path)
    processing_time = time.time() - start_time
    
    logger.info(f"Face model generated in {processing_time:.2f} seconds")
    
    # Display expression calibration data
    display_expression_data(result)
    
    # Save expression data to JSON file
    if result.expression_calibration_data:
        expression_data_path = os.path.join(output_dir, "expression_calibration_data.json")
        with open(expression_data_path, 'w') as f:
            json.dump(result.expression_calibration_data, f, indent=2)
        logger.info(f"Expression calibration data saved to {expression_data_path}")
    
    logger.info("Expression analysis completed successfully")
    return result

async def compare_expressions(face_modeling: FaceModeling, image_path: str, output_dir: str):
    """
    Compare different expressions for visualization.
    
    Args:
        face_modeling: FaceModeling instance
        image_path: Path to the input image
        output_dir: Directory to save output files
    """
    logger.info("Comparing different facial expressions")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 3D face model with expression calibration
    config = {
        "expression_calibration_enabled": True,
        "detail_level": "high"
    }
    face_modeling.config.update(config)
    result = await face_modeling.generate_from_image(image_path)
    
    if not result.expression_calibration_data or "blendshapes" not in result.expression_calibration_data:
        logger.error("No expression blendshapes available for comparison")
        return
    
    # In a real implementation, this would visualize the actual 3D model with different expressions
    # For this example, we'll create a simplified visualization
    
    # Create a visualization of expression intensity ranges
    if "intensity_ranges" in result.expression_calibration_data:
        plt.figure(figsize=(10, 6))
        ranges = result.expression_calibration_data["intensity_ranges"]
        expressions = list(ranges.keys())
        max_values = [ranges[expr][1] for expr in expressions]
        
        # Sort by max intensity for better visualization
        sorted_indices = np.argsort(max_values)
        sorted_expressions = [expressions[i] for i in sorted_indices]
        sorted_max_values = [max_values[i] for i in sorted_indices]
        
        # Plot bars for maximum intensity
        plt.barh(sorted_expressions, sorted_max_values, color='skyblue')
        plt.xlabel('Maximum Intensity')
        plt.ylabel('Expression')
        plt.title('Calibrated Expression Intensity Ranges')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "expression_intensity_ranges.png")
        plt.savefig(output_path)
        logger.info(f"Expression intensity visualization saved to {output_path}")
        plt.close()
    
    # Create a visualization of expression compatibility
    if "interaction_matrix" in result.expression_calibration_data:
        matrix = result.expression_calibration_data["interaction_matrix"]
        expressions = list(matrix.keys())
        
        # Create matrix data
        matrix_data = np.zeros((len(expressions), len(expressions)))
        for i, expr1 in enumerate(expressions):
            for j, expr2 in enumerate(expressions):
                matrix_data[i, j] = matrix[expr1].get(expr2, 0.0)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix_data, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Compatibility')
        plt.xticks(range(len(expressions)), expressions, rotation=90)
        plt.yticks(range(len(expressions)), expressions)
        plt.title('Expression Compatibility Matrix')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "expression_compatibility_matrix.png")
        plt.savefig(output_path)
        logger.info(f"Expression compatibility matrix saved to {output_path}")
        plt.close()
    
    logger.info("Expression comparison completed successfully")

async def analyze_from_video(face_modeling: FaceModeling, video_path: str, output_dir: str):
    """
    Analyze expressions from a video file.
    
    Args:
        face_modeling: FaceModeling instance
        video_path: Path to the input video
        output_dir: Directory to save output files
    """
    logger.info(f"Analyzing expressions from video: {video_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 3D face model with expression calibration
    start_time = time.time()
    config = {
        "expression_calibration_enabled": True,
        "detail_level": "high",
        "identity_verification": True
    }
    
    face_modeling.config.update(config)
    result = await face_modeling.generate_from_video(video_path)
    processing_time = time.time() - start_time
    
    logger.info(f"Face model generated from video in {processing_time:.2f} seconds")
    
    # Display expression calibration data
    display_expression_data(result)
    
    # Save expression data to JSON file
    if result.expression_calibration_data:
        expression_data_path = os.path.join(output_dir, "video_expression_calibration_data.json")
        with open(expression_data_path, 'w') as f:
            json.dump(result.expression_calibration_data, f, indent=2)
        logger.info(f"Expression calibration data saved to {expression_data_path}")
    
    logger.info("Video expression analysis completed successfully")
    return result

async def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Expression Range Calibration Example")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to the input image")
    input_group.add_argument("--video", type=str, help="Path to the input video")
    
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Directory to save output files")
    parser.add_argument("--compare-expressions", action="store_true",
                       help="Compare different expressions with visualizations")
    
    args = parser.parse_args()
    
    try:
        # Initialize the FaceModeling component
        face_modeling = FaceModeling()
        
        if args.image:
            # Image-based processing
            result = await analyze_expressions(face_modeling, args.image, args.output_dir)
            
            if args.compare_expressions:
                await compare_expressions(face_modeling, args.image, args.output_dir)
        
        elif args.video:
            # Video-based processing
            result = await analyze_from_video(face_modeling, args.video, args.output_dir)
            
        logger.info("Expression calibration example completed successfully")
            
    except Exception as e:
        logger.error(f"Error in expression calibration example: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main()) 