#!/usr/bin/env python3
"""
Face Detail Refinement Example

This script demonstrates the high-resolution detail refinement capabilities
of the Face Modeling component, focusing on features like pores, wrinkles,
and skin texture enhancements.

Example usage:
    # Generate with high detail level (default)
    python face_detail_refinement_example.py --image path/to/face/image.jpg
    
    # Generate with ultra detail level
    python face_detail_refinement_example.py --image path/to/face/image.jpg --detail-level ultra
    
    # Compare different detail levels
    python face_detail_refinement_example.py --image path/to/face/image.jpg --compare-all
"""

import os
import sys
import argparse
import logging
import json
import time
import asyncio
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to allow importing from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.avatar.face_modeling import FaceModeling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("face_detail_refinement_example")

def display_detail_info(detail_level: str, processing_time: float):
    """
    Display information about the detail refinement process.
    
    Args:
        detail_level: The level of detail used (low, medium, high, ultra)
        processing_time: Time taken to process in seconds
    """
    print(f"\n=== Detail Refinement at '{detail_level.upper()}' level ===")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Detail level descriptions
    descriptions = {
        "low": "Basic skin texture with minimal pore simulation",
        "medium": "Enhanced skin texture with pores and basic wrinkle simulation",
        "high": "Detailed skin texture with varied pore sizes, wrinkles, and texture variations",
        "ultra": "Ultra-realistic skin with multi-layered details, fine wrinkles, and subsurface scattering"
    }
    
    print(f"Features: {descriptions.get(detail_level.lower(), 'Custom detail level')}")

async def process_image_with_detail(
    face_modeling: FaceModeling,
    image_path: str, 
    output_dir: str,
    detail_level: str = "high"
) -> tuple:
    """
    Process an image with the specified detail level and save the result.
    
    Args:
        face_modeling: FaceModeling instance
        image_path: Path to the input image
        output_dir: Directory to save the output
        detail_level: Level of detail to apply
        
    Returns:
        Tuple of (output_path, processing_time)
    """
    logger.info(f"Processing image with '{detail_level}' detail level: {image_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Process with timing
    start_time = time.time()
    
    # Apply detail refinement directly
    # In a real implementation, this would generate the full 3D model
    # For this example, we'll just demonstrate the texture enhancement
    enhanced_texture = face_modeling._add_micro_detail(image, detail_level)
    
    processing_time = time.time() - start_time
    
    # Save the result
    output_filename = f"detail_refinement_{detail_level}_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, enhanced_texture)
    
    logger.info(f"Saved refined image to {output_path}")
    return output_path, processing_time

async def compare_detail_levels(
    face_modeling: FaceModeling,
    image_path: str, 
    output_dir: str
) -> None:
    """
    Compare different detail levels on the same image and create a comparison visualization.
    
    Args:
        face_modeling: FaceModeling instance
        image_path: Path to the input image
        output_dir: Directory to save the output
    """
    logger.info(f"Comparing all detail levels for image: {image_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Detail levels to compare
    detail_levels = ["low", "medium", "high", "ultra"]
    
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Process with each detail level
    enhanced_images = []
    processing_times = []
    
    for level in detail_levels:
        logger.info(f"Processing with '{level}' detail level")
        start_time = time.time()
        enhanced = face_modeling._add_micro_detail(original_image.copy(), level)
        processing_time = time.time() - start_time
        
        # Convert to RGB for matplotlib
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        enhanced_images.append(enhanced_rgb)
        processing_times.append(processing_time)
        
        logger.info(f"'{level}' detail processing time: {processing_time:.2f} seconds")
    
    # Create comparison visualization
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # Enhanced images
    for i, (level, img, ptime) in enumerate(zip(detail_levels, enhanced_images, processing_times)):
        plt.subplot(2, 3, i + 2)
        plt.imshow(img)
        plt.title(f"{level.upper()} Detail\n({ptime:.2f}s)")
        plt.axis('off')
    
    # Create a zoomed-in comparison of a detail area (e.g., around the eye)
    # In a real implementation, this would use face landmarks to select regions
    # Here we'll use a simplified approach with the center of the image
    
    # Define a small region to zoom in on (center of the image)
    h, w = original_rgb.shape[:2]
    crop_h, crop_w = h // 5, w // 5
    center_y, center_x = h // 2, w // 2
    crop_y1, crop_y2 = center_y - crop_h // 2, center_y + crop_h // 2
    crop_x1, crop_x2 = center_x - crop_w // 2, center_x + crop_w // 2
    
    # Show zoomed crops in the bottom row
    plt.subplot(2, 3, 6)
    crop_ultra = enhanced_images[3][crop_y1:crop_y2, crop_x1:crop_x2]
    plt.imshow(crop_ultra)
    plt.title(f"ULTRA Detail\n(Zoomed)")
    plt.axis('off')
    
    # Add a rectangle to the main ultra image to show the zoomed area
    plt.subplot(2, 3, 5)
    plt.plot([crop_x1, crop_x2, crop_x2, crop_x1, crop_x1],
             [crop_y1, crop_y1, crop_y2, crop_y2, crop_y1],
             'r-', linewidth=2)
    
    # Save the comparison
    comparison_path = os.path.join(output_dir, f"detail_level_comparison_{os.path.basename(image_path).split('.')[0]}.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300)
    logger.info(f"Saved comparison visualization to {comparison_path}")
    
    # Also save individual enhanced images
    for level, img in zip(detail_levels, enhanced_images):
        output_filename = f"detail_{level}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved {level} detail image to {output_path}")

async def main():
    """Main function to process command line arguments and run the example."""
    parser = argparse.ArgumentParser(description="Face Detail Refinement Example")
    parser.add_argument("--image", type=str, help="Path to the input face image")
    parser.add_argument("--output-dir", type=str, default="./outputs", 
                        help="Directory to save the outputs")
    parser.add_argument("--detail-level", type=str, default="high", 
                        choices=["low", "medium", "high", "ultra"],
                        help="Level of detail refinement")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all detail levels and create a visualization")
    
    args = parser.parse_args()
    
    if not args.image:
        parser.error("Please provide an input image with --image")
    
    if not os.path.exists(args.image):
        parser.error(f"Input image not found: {args.image}")
    
    try:
        # Initialize the FaceModeling component
        face_modeling = FaceModeling()
        
        if args.compare_all:
            # Compare all detail levels
            await compare_detail_levels(face_modeling, args.image, args.output_dir)
            print("\n=== Detail Level Comparison Complete ===")
            print(f"Comparison saved to {args.output_dir}")
        else:
            # Process with single detail level
            output_path, processing_time = await process_image_with_detail(
                face_modeling, args.image, args.output_dir, args.detail_level)
            
            display_detail_info(args.detail_level, processing_time)
            print(f"\nEnhanced image saved to: {output_path}")
            
    except Exception as e:
        logger.error(f"Error in face detail refinement example: {str(e)}")
        logger.exception("Detailed error information:")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 