#!/usr/bin/env python3
"""
Sound Effects Recommendation Example

This example demonstrates the context-aware sound effect recommendation functionality
of the Sound Effects Library. It shows how to recommend sound effects based on:
- Text transcripts
- Scene descriptions
- Video categories
- Mood information
- Keywords
- Scene intensity
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.sound_effects.sound_effects_library import SoundEffectsLibrary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_library(config: Optional[Dict[str, Any]] = None) -> SoundEffectsLibrary:
    """
    Initialize the sound effects library.
    
    Args:
        config: Optional configuration parameters
        
    Returns:
        Initialized SoundEffectsLibrary instance
    """
    sfx_library = SoundEffectsLibrary(config)
    logger.info(f"Sound Effects Library initialized with {len(sfx_library.metadata)} effects")
    return sfx_library

def recommend_for_action_scene(sfx_library: SoundEffectsLibrary) -> None:
    """
    Demonstrate sound effect recommendations for an action scene.
    """
    logger.info("\n\n--- EXAMPLE 1: ACTION SCENE RECOMMENDATIONS ---")
    
    context = {
        "transcript": "The car speeds down the highway as sirens wail in the distance. "
                     "The driver swerves through traffic, narrowly missing other vehicles. "
                     "The engine roars as he accelerates to escape the pursuing police cars.",
        "scene_descriptions": [
            "High-speed car chase", 
            "Police pursuit",
            "Dangerous driving on highway"
        ],
        "video_category": "action",
        "mood": "tense",
        "keywords": ["car", "chase", "police", "speed", "highway", "engine"],
        "intensity": 0.8,
        "preceding_effects": []
    }
    
    results = sfx_library.recommend_sound_effects(
        context=context,
        max_results=5,
        include_details=True
    )
    
    logger.info(f"Found {len(results.get('recommendations', []))} recommended sound effects")
    
    # Display results
    print("\nRecommended sound effects for action scene:")
    print("-" * 80)
    for i, effect in enumerate(results.get('recommendations', []), 1):
        print(f"{i}. {effect.get('name', 'Unknown')} (ID: {effect.get('effect_id', 'N/A')})")
        print(f"   Category: {effect.get('category', 'Unknown')}")
        print(f"   Tags: {', '.join(effect.get('tags', []))}")
        print(f"   Relevance: {effect.get('relevance_score', 0):.2f}")
        print(f"   Description: {effect.get('description', 'No description')}")
        print("-" * 80)

def recommend_for_nature_scene(sfx_library: SoundEffectsLibrary) -> None:
    """
    Demonstrate sound effect recommendations for a nature scene.
    """
    logger.info("\n\n--- EXAMPLE 2: NATURE SCENE RECOMMENDATIONS ---")
    
    context = {
        "transcript": "The gentle stream flows through the forest, creating a soothing melody. "
                     "Birds chirp from the trees while the wind rustles the leaves. "
                     "A deer cautiously approaches the water to drink.",
        "scene_descriptions": [
            "Peaceful forest scene", 
            "Wildlife in natural habitat",
            "Flowing stream in woodland"
        ],
        "video_category": "nature",
        "mood": "peaceful",
        "keywords": ["forest", "stream", "birds", "wind", "wildlife", "deer"],
        "intensity": 0.3,
        "preceding_effects": []
    }
    
    results = sfx_library.recommend_sound_effects(
        context=context,
        max_results=5,
        include_details=True
    )
    
    logger.info(f"Found {len(results.get('recommendations', []))} recommended sound effects")
    
    # Display results
    print("\nRecommended sound effects for nature scene:")
    print("-" * 80)
    for i, effect in enumerate(results.get('recommendations', []), 1):
        print(f"{i}. {effect.get('name', 'Unknown')} (ID: {effect.get('effect_id', 'N/A')})")
        print(f"   Category: {effect.get('category', 'Unknown')}")
        print(f"   Tags: {', '.join(effect.get('tags', []))}")
        print(f"   Relevance: {effect.get('relevance_score', 0):.2f}")
        print(f"   Description: {effect.get('description', 'No description')}")
        print("-" * 80)

def recommend_for_horror_scene(sfx_library: SoundEffectsLibrary) -> None:
    """
    Demonstrate sound effect recommendations for a horror scene.
    """
    logger.info("\n\n--- EXAMPLE 3: HORROR SCENE RECOMMENDATIONS ---")
    
    context = {
        "transcript": "She walks slowly down the dark hallway, her footsteps echoing. "
                     "A door creaks in the distance. Something moves in the shadows. "
                     "Her breathing quickens as she senses she's not alone.",
        "scene_descriptions": [
            "Dark hallway in abandoned house", 
            "Character sensing danger",
            "Unseen presence stalking"
        ],
        "video_category": "horror",
        "mood": "suspenseful",
        "keywords": ["dark", "creaking", "shadow", "abandoned", "footsteps", "breathing"],
        "intensity": 0.7,
        "preceding_effects": []
    }
    
    results = sfx_library.recommend_sound_effects(
        context=context,
        max_results=5,
        include_details=True
    )
    
    logger.info(f"Found {len(results.get('recommendations', []))} recommended sound effects")
    
    # Display results
    print("\nRecommended sound effects for horror scene:")
    print("-" * 80)
    for i, effect in enumerate(results.get('recommendations', []), 1):
        print(f"{i}. {effect.get('name', 'Unknown')} (ID: {effect.get('effect_id', 'N/A')})")
        print(f"   Category: {effect.get('category', 'Unknown')}")
        print(f"   Tags: {', '.join(effect.get('tags', []))}")
        print(f"   Relevance: {effect.get('relevance_score', 0):.2f}")
        print(f"   Description: {effect.get('description', 'No description')}")
        print("-" * 80)

def recommend_for_custom_scene(sfx_library: SoundEffectsLibrary, transcript: str) -> None:
    """
    Demonstrate sound effect recommendations for a custom scene based on user input.
    
    Args:
        sfx_library: SoundEffectsLibrary instance
        transcript: User-provided transcript text
    """
    logger.info("\n\n--- EXAMPLE 4: CUSTOM SCENE RECOMMENDATIONS ---")
    
    context = {
        "transcript": transcript,
        "scene_descriptions": [],
        "video_category": "",
        "mood": "",
        "keywords": [],
        "intensity": 0.5,
        "preceding_effects": []
    }
    
    results = sfx_library.recommend_sound_effects(
        context=context,
        max_results=5,
        include_details=True
    )
    
    logger.info(f"Found {len(results.get('recommendations', []))} recommended sound effects")
    
    # Display results
    print(f"\nRecommended sound effects for custom scene: \"{transcript}\"")
    print("-" * 80)
    for i, effect in enumerate(results.get('recommendations', []), 1):
        print(f"{i}. {effect.get('name', 'Unknown')} (ID: {effect.get('effect_id', 'N/A')})")
        print(f"   Category: {effect.get('category', 'Unknown')}")
        print(f"   Tags: {', '.join(effect.get('tags', []))}")
        print(f"   Relevance: {effect.get('relevance_score', 0):.2f}")
        print(f"   Description: {effect.get('description', 'No description')}")
        print("-" * 80)

def recommend_with_preceding_effects(sfx_library: SoundEffectsLibrary) -> None:
    """
    Demonstrate how recommendations change when preceding effects are considered
    to avoid repetition.
    """
    logger.info("\n\n--- EXAMPLE 5: RECOMMENDATIONS WITH PRECEDING EFFECTS ---")
    
    # First, get recommendations without preceding effects
    context_without = {
        "transcript": "Explosions rock the building as debris falls from the ceiling.",
        "scene_descriptions": ["Building collapsing", "Explosion aftermath"],
        "video_category": "action",
        "mood": "intense",
        "keywords": ["explosion", "debris", "building", "collapse"],
        "intensity": 0.9,
        "preceding_effects": []
    }
    
    results_without = sfx_library.recommend_sound_effects(
        context=context_without,
        max_results=3,
        include_details=True
    )
    
    # Get the top effect IDs
    top_effect_ids = [effect.get('effect_id') for effect in results_without.get('recommendations', [])]
    
    # Now get recommendations with the top effects as preceding effects
    context_with = {
        "transcript": "More explosions continue as the hero runs through the collapsing corridor.",
        "scene_descriptions": ["Building collapsing", "Character escaping"],
        "video_category": "action",
        "mood": "intense",
        "keywords": ["explosion", "debris", "building", "collapse", "escape"],
        "intensity": 0.9,
        "preceding_effects": top_effect_ids
    }
    
    results_with = sfx_library.recommend_sound_effects(
        context=context_with,
        max_results=3,
        include_details=True
    )
    
    # Display both results for comparison
    print("\nRecommendations WITHOUT preceding effects:")
    print("-" * 80)
    for i, effect in enumerate(results_without.get('recommendations', []), 1):
        print(f"{i}. {effect.get('name', 'Unknown')} (ID: {effect.get('effect_id', 'N/A')})")
        print(f"   Category: {effect.get('category', 'Unknown')}")
        print(f"   Relevance: {effect.get('relevance_score', 0):.2f}")
    
    print("\nRecommendations WITH preceding effects (to avoid repetition):")
    print("-" * 80)
    for i, effect in enumerate(results_with.get('recommendations', []), 1):
        print(f"{i}. {effect.get('name', 'Unknown')} (ID: {effect.get('effect_id', 'N/A')})")
        print(f"   Category: {effect.get('category', 'Unknown')}")
        print(f"   Relevance: {effect.get('relevance_score', 0):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Sound Effects Recommendation Examples')
    parser.add_argument('--custom', type=str, help='Custom scene description for recommendations')
    parser.add_argument('--config', type=str, help='Path to library configuration file')
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return
    
    # Initialize library
    sfx_library = init_library(config)
    
    # Run examples
    recommend_for_action_scene(sfx_library)
    recommend_for_nature_scene(sfx_library)
    recommend_for_horror_scene(sfx_library)
    
    if args.custom:
        recommend_for_custom_scene(sfx_library, args.custom)
    
    recommend_with_preceding_effects(sfx_library)
    
    logger.info("Completed all sound effects recommendation examples")

if __name__ == "__main__":
    main() 