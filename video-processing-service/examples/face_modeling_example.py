#!/usr/bin/env python3
"""
Face Modeling Example

This script demonstrates the facial modeling capabilities for avatar generation,
focusing on the 3D face reconstruction, high-fidelity texture mapping, and the
detailed feature preservation algorithm.

Example usage:
    # Generate from image
    python face_modeling_example.py --image path/to/face/image.jpg
    
    # Generate from video with ultra detail level
    python face_modeling_example.py --video path/to/face/video.mp4 --detail ultra
    
    # Generate without StyleGAN enhancements
    python face_modeling_example.py --image path/to/face/image.jpg --no-stylegan
"""

import os
import sys
import argparse
import logging
import json
import time
import asyncio
from pathlib import Path

# Add the parent directory to sys.path to allow importing from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.avatar.face_modeling import FaceModeling, FaceModelingResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("face_modeling_example")

def display_model_info(result: FaceModelingResult):
    """
    Display information about the generated 3D face model.
    
    Args:
        result: FaceModelingResult containing information about the generated model
    """
    print("\n=== 3D Face Model Generated ===")
    print(f"Model ID: {result.model_id}")
    print(f"Model Path: {result.model_path}")
    print(f"Texture Path: {result.texture_path}")
    print(f"Quality Score: {result.quality_score:.4f}")
    print(f"Processing Time: {result.processing_time:.2f} seconds")
    
    if result.identity_verification_score is not None:
        print(f"Identity Verification Score: {result.identity_verification_score:.4f}")
    
    print("\nLandmarks:")
    # Display a few key landmarks for illustration
    landmark_keys = list(result.landmarks.keys())
    for key in landmark_keys[:5]:  # Show first 5 landmarks
        print(f"  {key}: {result.landmarks[key]}")
    
    if len(landmark_keys) > 5:
        print(f"  ... and {len(landmark_keys) - 5} more landmarks")
    
    if result.expression_calibration_data:
        print("\nExpression Calibration:")
        for expr, value in list(result.expression_calibration_data.items())[:3]:
            print(f"  {expr}: {value}")
    
    print("\nModel saved successfully!")

async def generate_from_image_example(image_path: str, output_dir: str, detail_level: str = "high") -> None:
    """
    Generate a 3D face model from an image with detailed feature preservation.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output files
        detail_level: Level of detail for the model ('low', 'medium', 'high', 'ultra')
    """
    logger.info(f"Generating 3D face model from image: {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the FaceModeling service with configuration
    config = {
        "texture_resolution": (4096, 4096),  # 4K texture resolution
        "detail_level": detail_level,
        "detail_refinement_enabled": True,
        "expression_calibration_enabled": True,
        "identity_verification_threshold": 0.85
    }
    
    face_modeling = FaceModeling(config)
    
    # Generate 3D face model
    start_time = time.time()
    result = await face_modeling.generate_from_image(image_path)
    processing_time = time.time() - start_time
    
    logger.info(f"3D face model generated in {processing_time:.2f} seconds")
    logger.info(f"Model saved to: {result.model_path}")
    logger.info(f"Texture saved to: {result.texture_path}")
    
    # Save the result metadata to a JSON file for inspection
    result_summary = {
        "model_id": result.model_id,
        "model_path": result.model_path,
        "texture_path": result.texture_path,
        "quality_score": result.quality_score,
        "processing_time": result.processing_time,
        "identity_verification_score": result.identity_verification_score,
        "has_expression_data": result.expression_calibration_data is not None,
        "metadata": result.metadata,
        "feature_preservation": {
            "algorithm": "Detailed Feature Preservation",
            "key_regions": ["eyes", "nose", "mouth", "contour"],
            "weights": {
                "eyes": 0.95,      # High preservation for eyes (identity)
                "nose": 0.85,      # High preservation for nose (identity)
                "mouth": 0.80,     # Medium preservation for mouth (expressions vary)
                "contour": 0.70    # Lower preservation for contour
            }
        }
    }
    
    result_path = os.path.join(output_dir, f"{result.model_id}_metadata.json")
    with open(result_path, "w") as f:
        json.dump(result_summary, f, indent=2)
    
    logger.info(f"Result metadata saved to: {result_path}")

async def generate_from_video_example(video_path: str, output_dir: str, detail_level: str = "high") -> None:
    """
    Generate a 3D face model from a video with detailed feature preservation.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save output files
        detail_level: Level of detail for the model ('low', 'medium', 'high', 'ultra')
    """
    logger.info(f"Generating 3D face model from video: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the FaceModeling service with configuration
    config = {
        "texture_resolution": (4096, 4096),  # 4K texture resolution
        "detail_level": detail_level,
        "detail_refinement_enabled": True,
        "expression_calibration_enabled": True,
        "identity_verification_threshold": 0.85,
        "num_frames_to_extract": 15  # Extract 15 frames for better accuracy
    }
    
    face_modeling = FaceModeling(config)
    
    # Generate 3D face model from video
    start_time = time.time()
    result = await face_modeling.generate_from_video(video_path)
    processing_time = time.time() - start_time
    
    logger.info(f"3D face model generated in {processing_time:.2f} seconds")
    logger.info(f"Model saved to: {result.model_path}")
    logger.info(f"Texture saved to: {result.texture_path}")
    
    # Save the result metadata to a JSON file for inspection
    result_summary = {
        "model_id": result.model_id,
        "model_path": result.model_path,
        "texture_path": result.texture_path,
        "quality_score": result.quality_score,
        "processing_time": result.processing_time,
        "identity_verification_score": result.identity_verification_score,
        "has_expression_data": result.expression_calibration_data is not None,
        "metadata": result.metadata,
        "feature_preservation": {
            "algorithm": "Detailed Feature Preservation",
            "key_regions": ["eyes", "nose", "mouth", "contour"],
            "weights": {
                "eyes": 0.95,      # High preservation for eyes (identity)
                "nose": 0.85,      # High preservation for nose (identity)
                "mouth": 0.80,     # Medium preservation for mouth (expressions vary)
                "contour": 0.70    # Lower preservation for contour
            }
        }
    }
    
    result_path = os.path.join(output_dir, f"{result.model_id}_metadata.json")
    with open(result_path, "w") as f:
        json.dump(result_summary, f, indent=2)
    
    logger.info(f"Result metadata saved to: {result_path}")

async def display_feature_preservation_info() -> None:
    """Display information about the detailed feature preservation algorithm."""
    logger.info("=== Detailed Feature Preservation Algorithm ===")
    logger.info("This algorithm ensures that distinctive facial features are accurately preserved during")
    logger.info("the 3D reconstruction process, maintaining the subject's identity and unique characteristics.")
    logger.info("")
    logger.info("Key components:")
    logger.info("1. Region-based feature preservation - Applies different weights to facial regions")
    logger.info("2. Geometric constraints - Maintains correct facial proportions")
    logger.info("3. Local detail preservation - Preserves fine features using Laplacian mesh editing")
    logger.info("4. Statistical constraints - Ensures features remain within realistic anatomical bounds")
    logger.info("")
    logger.info("Region weights:")
    logger.info("- Eyes: 0.95 (High preservation for identity)")
    logger.info("- Nose: 0.85 (High preservation for identity)")
    logger.info("- Mouth: 0.80 (Medium preservation as expressions vary)")
    logger.info("- Contour: 0.70 (Lower preservation for contour)")

async def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Face Modeling Example")
    parser.add_argument("--mode", choices=["image", "video", "info"], default="info",
                        help="Mode: generate from image, video, or display info")
    parser.add_argument("--input", help="Path to input image or video")
    parser.add_argument("--output-dir", default="./output", help="Directory to save output files")
    parser.add_argument("--detail-level", choices=["low", "medium", "high", "ultra"], default="high",
                        help="Level of detail for the model")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "image":
            if not args.input:
                logger.error("Input image path is required for image mode")
                sys.exit(1)
            await generate_from_image_example(args.input, args.output_dir, args.detail_level)
        elif args.mode == "video":
            if not args.input:
                logger.error("Input video path is required for video mode")
                sys.exit(1)
            await generate_from_video_example(args.input, args.output_dir, args.detail_level)
        else:
            await display_feature_preservation_info()
    except Exception as e:
        logger.error(f"Error in face modeling example: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 