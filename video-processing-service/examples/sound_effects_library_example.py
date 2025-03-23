#!/usr/bin/env python3
"""
Sound Effects Library Example

This script demonstrates how to use the Sound Effects Library to manage
a collection of categorized sound effects.

Features demonstrated:
- Adding sound effects with metadata
- Browsing categories
- Searching sound effects by various criteria
- Retrieving sound effects
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, Any, List, Optional

# Add the project root to Python path to allow importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.sound_effects.sound_effects_library import SoundEffectsLibrary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_library() -> SoundEffectsLibrary:
    """Initialize the Sound Effects Library."""
    library = SoundEffectsLibrary()
    logger.info("Sound Effects Library initialized")
    return library

def add_example_sound_effects(library: SoundEffectsLibrary, sample_dir: str):
    """
    Add example sound effects to the library.
    
    Args:
        library: Sound Effects Library instance
        sample_dir: Directory containing sample sound effects
    """
    # Example sound effects with metadata
    sample_effects = [
        {
            "file": "thunder.wav",
            "name": "Thunder Crack",
            "category": "weather",
            "tags": ["thunder", "storm", "loud", "dramatic"],
            "description": "Powerful thunder crack for storm scenes",
            "trigger_words": ["thunder", "storm", "lightning", "rumble"]
        },
        {
            "file": "footsteps_wood.wav",
            "name": "Footsteps on Wood",
            "category": "foley",
            "tags": ["footsteps", "wood", "walking", "interior"],
            "description": "Person walking on wooden floor",
            "trigger_words": ["walk", "step", "footstep", "wood", "floor"]
        },
        {
            "file": "car_horn.wav",
            "name": "Car Horn",
            "category": "vehicles",
            "tags": ["car", "horn", "traffic", "urban"],
            "description": "Standard car horn sound",
            "trigger_words": ["honk", "car", "traffic", "horn"]
        },
        {
            "file": "wind_strong.wav",
            "name": "Strong Wind",
            "category": "weather",
            "tags": ["wind", "storm", "ambience", "outdoor"],
            "description": "Strong wind blowing through trees",
            "trigger_words": ["wind", "blow", "storm", "breeze"]
        },
        {
            "file": "door_creak.wav",
            "name": "Door Creak",
            "category": "foley",
            "tags": ["door", "creak", "spooky", "interior"],
            "description": "Creaking door hinge, spooky atmosphere",
            "trigger_words": ["door", "creak", "open", "hinge", "spooky"]
        }
    ]
    
    added_count = 0
    for effect in sample_effects:
        file_path = os.path.join(sample_dir, effect["file"])
        
        # Skip if file doesn't exist (this is just a demo)
        if not os.path.exists(file_path):
            logger.warning(f"Sample file not found: {file_path}")
            # Create a dummy file for demo purposes
            with open(file_path, "wb") as f:
                f.write(b"Dummy audio data for demo purposes")
        
        # Add to library
        result = library.add_sound_effect(
            file_path=file_path,
            name=effect["name"],
            category=effect["category"],
            tags=effect["tags"],
            description=effect["description"],
            trigger_words=effect["trigger_words"]
        )
        
        if "effect_id" in result:
            added_count += 1
            logger.info(f"Added sound effect: {effect['name']} (ID: {result['effect_id']})")
    
    logger.info(f"Added {added_count} example sound effects to the library")

def display_categories(library: SoundEffectsLibrary):
    """
    Display all sound effect categories.
    
    Args:
        library: Sound Effects Library instance
    """
    categories = library.get_categories()
    
    logger.info("=== Sound Effect Categories ===")
    for category_id, category_data in categories.items():
        logger.info(f"{category_data['name']}: {category_data['effects_count']} effects - {category_data['description']}")
    
    total_effects = sum(cat["effects_count"] for cat in categories.values())
    logger.info(f"Total effects across all categories: {total_effects}")

def search_effects_demo(library: SoundEffectsLibrary):
    """
    Demonstrate searching for sound effects.
    
    Args:
        library: Sound Effects Library instance
    """
    # Search by category
    logger.info("\n=== Search by Category: 'weather' ===")
    weather_effects = library.search_sound_effects(category="weather")
    for effect in weather_effects:
        logger.info(f"Found: {effect['name']} - {effect['description']}")
    
    # Search by tags
    logger.info("\n=== Search by Tags: ['spooky'] ===")
    spooky_effects = library.search_sound_effects(tags=["spooky"])
    for effect in spooky_effects:
        logger.info(f"Found: {effect['name']} - {effect['description']}")
    
    # Text search
    logger.info("\n=== Text Search: 'door' ===")
    door_effects = library.search_sound_effects(search_term="door")
    for effect in door_effects:
        logger.info(f"Found: {effect['name']} - {effect['description']}")

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Sound Effects Library Example")
    parser.add_argument("--sample-dir", default="./samples", help="Directory for sample sound effects")
    args = parser.parse_args()
    
    # Create sample directory if it doesn't exist
    os.makedirs(args.sample_dir, exist_ok=True)
    
    logger.info("Starting Sound Effects Library Example")
    
    # Initialize sound effects library
    library = init_library()
    
    # Add example sound effects
    add_example_sound_effects(library, args.sample_dir)
    
    # Display categories
    display_categories(library)
    
    # Search effects demo
    search_effects_demo(library)
    
    logger.info("Sound Effects Library Example completed")

if __name__ == "__main__":
    main() 