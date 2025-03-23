"""
Clip Generation Service

A microservice for generating video clips from source videos with flexible configuration options and a scalable architecture.
"""

__version__ = "1.0.0"
__author__ = "AI Platform Team"
__license__ = "MIT"

# Make sure necessary subdirectories exist
import os
from pathlib import Path

# Create __init__.py files in subdirectories if they don't exist
for subdir in ["api", "config", "models", "services", "utils", "workers"]:
    dir_path = Path(__file__).parent / subdir
    
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = dir_path / "__init__.py"
    if not init_file.exists():
        with open(init_file, "w") as f:
            f.write(f'"""\nClip Generation Service - {subdir.capitalize()} Package\n"""\n') 