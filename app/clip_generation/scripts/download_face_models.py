#!/usr/bin/env python3
"""
Download Face Models Script

This script downloads and sets up the required models for the Face Tracking System.
It handles downloading YOLO, MediaPipe, RetinaFace, and ArcFace models.

Usage:
    python download_face_models.py [--output_dir MODELS_DIR]
"""

import os
import sys
import argparse
import logging
import requests
import zipfile
import gdown
import torch
import ultralytics
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model URLs and paths
MODELS = {
    "yolov8n-face": {
        "url": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
        "path": "yolov8n-face.pt"
    },
    "yolov8s-face": {
        "url": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8s-face.pt",
        "path": "yolov8s-face.pt"
    },
    "retinaface_resnet50": {
        "url": "https://drive.google.com/uc?id=1hzgOejAfCAB8WJwQigmMT_0jvNqjY1fg",
        "path": "retinaface_resnet50.pth"
    },
    "arcface_resnet50": {
        "url": "https://drive.google.com/uc?id=1c-ETpoIQsCKWquu3fuQJhtNcQNHmGc-6",
        "path": "arcface_model.pth"
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download Face Models")
    parser.add_argument("--output_dir", type=str, default="../models/face",
                        help="Directory to save models")
    return parser.parse_args()


def download_file(url, path, desc=None):
    """Download a file from a URL with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(path, 'wb') as f, tqdm(
            desc=desc or os.path.basename(path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)
        
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def download_from_gdrive(url, path, desc=None):
    """Download a file from Google Drive with progress bar."""
    try:
        gdown.download(url, path, quiet=False)
        return True
    except Exception as e:
        logger.error(f"Error downloading from Google Drive {url}: {e}")
        return False


def download_mediapipe_models():
    """Download MediaPipe models."""
    # MediaPipe models are downloaded automatically when first used
    # We'll just import the module to trigger the download
    try:
        import mediapipe as mp
        logger.info("MediaPipe imported successfully. Models will be downloaded on first use.")
        return True
    except ImportError:
        logger.error("Failed to import MediaPipe. Please install with: pip install mediapipe")
        return False


def main():
    """Main function."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Downloading face models to {output_dir}")
    
    # Download YOLO models
    for name, model_info in MODELS.items():
        model_path = output_dir / model_info["path"]
        
        if model_path.exists():
            logger.info(f"Model {name} already exists at {model_path}")
            continue
        
        logger.info(f"Downloading {name} model...")
        
        if "drive.google.com" in model_info["url"]:
            success = download_from_gdrive(model_info["url"], model_path, desc=name)
        else:
            success = download_file(model_info["url"], model_path, desc=name)
        
        if success:
            logger.info(f"Successfully downloaded {name} to {model_path}")
        else:
            logger.error(f"Failed to download {name}")
    
    # Download MediaPipe models
    logger.info("Setting up MediaPipe models...")
    download_mediapipe_models()
    
    logger.info("All models have been downloaded.")
    
    # Verify YOLO model works
    try:
        yolo_path = output_dir / "yolov8n-face.pt"
        if yolo_path.exists():
            logger.info("Testing YOLO model...")
            model = ultralytics.YOLO(str(yolo_path))
            logger.info("YOLO model loaded successfully!")
    except Exception as e:
        logger.error(f"Error testing YOLO model: {e}")
    
    logger.info("Face model setup completed!")


if __name__ == "__main__":
    main() 