#!/usr/bin/env python3
"""
Identity Consistency Verification Example

This script demonstrates the identity consistency verification feature of the Face Modeling
component, which ensures that the generated 3D face model maintains the identity of the
person in the input image or video.

Example usage:
    # Verify identity from a single image
    python identity_verification_example.py --image path/to/face/image.jpg
    
    # Verify identity from a video
    python identity_verification_example.py --video path/to/face/video.mp4
    
    # Verify identity and compare to reference images
    python identity_verification_example.py --image path/to/face/image.jpg --reference path/to/reference/image.jpg
"""

import os
import sys
import argparse
import logging
import json
import time
import asyncio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil

# Add the parent directory to sys.path to allow importing from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.avatar.face_modeling import FaceModeling, FaceModelingResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("identity_verification_example")

def display_verification_results(result: FaceModelingResult, original_path: str):
    """
    Display information about the identity verification results.
    
    Args:
        result: FaceModelingResult containing identity verification data
        original_path: Path to the original image used for reconstruction
    """
    print("\n=== Identity Verification Results ===")
    print(f"Model ID: {result.model_id}")
    
    if result.identity_verification_score is not None:
        score = result.identity_verification_score
        print(f"Identity Verification Score: {score:.4f} (0-1)")
        
        # Interpret the score
        if score >= 0.9:
            print("Interpretation: Excellent match - Very high confidence")
        elif score >= 0.8:
            print("Interpretation: Good match - High confidence")
        elif score >= 0.7:
            print("Interpretation: Acceptable match - Moderate confidence")
        elif score >= 0.6:
            print("Interpretation: Borderline match - Low confidence")
        else:
            print("Interpretation: Poor match - Identity may not be preserved")
            
        # Display components of the verification (from metadata if available)
        if result.metadata and 'identity_verification' in result.metadata:
            verification_data = result.metadata['identity_verification']
            print("\nVerification Components:")
            
            if 'geometric_similarity' in verification_data:
                print(f"• Geometric Similarity: {verification_data['geometric_similarity']:.4f}")
                print("  (Based on facial landmark correspondence and proportions)")
                
            if 'texture_similarity' in verification_data:
                print(f"• Texture Similarity: {verification_data['texture_similarity']:.4f}")
                print("  (Based on skin tone, features, and texture patterns)")
                
            if 'deep_feature_similarity' in verification_data:
                print(f"• Deep Feature Similarity: {verification_data['deep_feature_similarity']:.4f}")
                print("  (Based on neural network feature extraction)")
    else:
        print("Identity verification was not performed or results not available")

async def verify_identity_from_image(image_path: str, output_dir: str):
    """
    Generate a 3D face model from an image and display identity verification results.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output files
    """
    logger.info(f"Starting identity verification from image: {image_path}")
    
    face_modeling = FaceModeling()
    
    start_time = time.time()
    
    # Generate the 3D face model with identity verification enabled
    options = {
        "detail_level": "high",
        "enable_texture_mapping": True,
        "enable_detail_refinement": True,
        "enable_identity_verification": True,  # Ensure identity verification is enabled
        "enable_stylegan_enhancements": True,
        "enable_expression_calibration": True
    }
    
    try:
        result = await face_modeling.generate_from_image(image_path, options)
        
        # Display the verification results
        display_verification_results(result, image_path)
        
        # Save the output files
        os.makedirs(output_dir, exist_ok=True)
        
        # Save a copy of the output model and texture
        model_output_path = os.path.join(output_dir, f"model_{result.model_id}.glb")
        texture_output_path = os.path.join(output_dir, f"texture_{result.model_id}.png")
        
        shutil.copy(result.model_path, model_output_path)
        shutil.copy(result.texture_path, texture_output_path)
        
        logger.info(f"Files saved to: {output_dir}")
        print(f"\nProcessing time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during identity verification: {str(e)}")
        raise

async def verify_identity_comparison(image_path: str, reference_path: str, output_dir: str):
    """
    Compare the identity of a generated 3D face model with a reference image.
    
    Args:
        image_path: Path to the input image for model generation
        reference_path: Path to a reference image of the same person
        output_dir: Directory to save output files
    """
    logger.info(f"Starting identity comparison between {image_path} and {reference_path}")
    
    face_modeling = FaceModeling()
    
    # Generate the 3D face model with identity verification enabled
    options = {
        "detail_level": "high",
        "enable_texture_mapping": True,
        "enable_detail_refinement": True,
        "enable_identity_verification": True,
        "enable_stylegan_enhancements": True,
        "enable_expression_calibration": True,
        "reference_image_path": reference_path  # Pass the reference image
    }
    
    try:
        result = await face_modeling.generate_from_image(image_path, options)
        
        # Display the verification results
        display_verification_results(result, image_path)
        
        # Create visualization of the comparison
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a comparison image
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Load and display the original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axs[0].imshow(original_img)
        axs[0].set_title("Source Image")
        axs[0].axis("off")
        
        # Load and display the reference image
        reference_img = cv2.imread(reference_path)
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        axs[1].imshow(reference_img)
        axs[1].set_title("Reference Image")
        axs[1].axis("off")
        
        # Load and display the generated texture
        texture_img = cv2.imread(result.texture_path)
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
        axs[2].imshow(texture_img)
        axs[2].set_title(f"Generated Texture\nIdentity Score: {result.identity_verification_score:.4f}")
        axs[2].axis("off")
        
        # Add the verification score as a figure title
        fig.suptitle(f"Identity Verification - Score: {result.identity_verification_score:.4f}", fontsize=16)
        
        # Save the comparison
        comparison_path = os.path.join(output_dir, f"identity_comparison_{result.model_id}.png")
        plt.tight_layout()
        plt.savefig(comparison_path)
        
        logger.info(f"Comparison saved to: {comparison_path}")
        
    except Exception as e:
        logger.error(f"Error during identity comparison: {str(e)}")
        raise

async def main():
    parser = argparse.ArgumentParser(description="Identity Consistency Verification Example")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to an image containing a face")
    group.add_argument("--video", help="Path to a video containing a face")
    parser.add_argument("--reference", help="Path to a reference image for comparison")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.image:
        if args.reference:
            await verify_identity_comparison(args.image, args.reference, args.output_dir)
        else:
            await verify_identity_from_image(args.image, args.output_dir)
    elif args.video:
        # Similar code for video processing could be added here
        logger.info("Video-based identity verification not implemented in this example")
    
if __name__ == "__main__":
    asyncio.run(main()) 