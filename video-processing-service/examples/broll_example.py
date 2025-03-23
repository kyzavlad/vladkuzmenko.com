#!/usr/bin/env python3
"""
B-Roll Insertion Example

This script demonstrates how to use the B-Roll Insertion Engine to automatically
suggest and insert B-Roll footage into a video based on content analysis.
"""

import os
import sys
import logging
import asyncio
import argparse
import json
import time
from pathlib import Path

# Add parent directory to path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.broll.broll_engine import BRollEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Sample transcript for demonstration
SAMPLE_TRANSCRIPT = {
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 5.2,
            "text": "Welcome to our tutorial on modern web development technologies."
        },
        {
            "id": 1,
            "start": 5.3,
            "end": 10.1,
            "text": "Today we'll be exploring how cloud computing has transformed the way we build applications."
        },
        {
            "id": 2,
            "start": 10.2,
            "end": 15.6,
            "text": "Cloud platforms like AWS, Google Cloud, and Azure provide scalable infrastructure."
        },
        {
            "id": 3,
            "start": 15.7,
            "end": 20.5,
            "text": "This allows developers to focus on code rather than managing servers."
        },
        {
            "id": 4,
            "start": 20.6,
            "end": 25.3,
            "text": "Containers have also revolutionized deployment, with Docker leading the way."
        },
        {
            "id": 5,
            "start": 25.4,
            "end": 30.8,
            "text": "Kubernetes orchestrates these containers, handling scaling and load balancing automatically."
        },
        {
            "id": 6,
            "start": 30.9,
            "end": 35.4,
            "text": "Modern front-end frameworks like React and Vue make creating interactive UIs easier."
        },
        {
            "id": 7,
            "start": 35.5,
            "end": 40.2,
            "text": "While back-end technologies like Node.js enable JavaScript across the entire stack."
        },
        {
            "id": 8,
            "start": 40.3,
            "end": 45.7,
            "text": "Database technologies have evolved too, with NoSQL options like MongoDB gaining popularity."
        },
        {
            "id": 9,
            "start": 45.8,
            "end": 50.0,
            "text": "That's a brief overview of today's web development landscape. Thanks for watching!"
        }
    ]
}

class ProgressReporter:
    """Simple class to report progress during B-Roll processing."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def report_progress(self, step, progress):
        """Report progress to console."""
        elapsed = time.time() - self.start_time
        logger.info(f"Step: {step} - Progress: {progress:.1f}% - Time elapsed: {elapsed:.1f}s")


async def process_video(
    video_path, 
    output_dir=None, 
    provider=None,
    use_semantic_matching=True,
    display_concepts=False
):
    """
    Process a video with the B-Roll Engine.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save outputs
        provider: Stock footage provider to use
        use_semantic_matching: Whether to use semantic matching
        display_concepts: Whether to display semantic concepts in console
    """
    # Create output directory
    if not output_dir:
        output_dir = f"broll_output_{int(time.time())}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize B-Roll Engine
    logger.info("Initializing B-Roll Engine...")
    
    config = {
        'content_analyzer': {
            'use_advanced_nlp': True,
        },
        'scene_detector': {
            'min_scene_length': 2.0,
            'use_face_detection': True,
            'use_audio_analysis': True,
        },
        'stock_provider': {
            'api_keys': {
                'pexels': os.environ.get('PEXELS_API_KEY', ''),
                'pixabay': os.environ.get('PIXABAY_API_KEY', ''),
            },
            'user_library_paths': [
                os.environ.get('USER_LIBRARY_PATH', './broll_library')
            ],
            'supported_providers': ['pexels', 'pixabay', 'local'],
        },
        'semantic_matcher': {
            'similarity_threshold': 0.65,
            'spacy_model': 'en_core_web_md',
        },
        'output_dir': output_dir,
    }
    
    engine = BRollEngine(config)
    progress_reporter = ProgressReporter()
    
    try:
        # Process the video
        logger.info(f"Processing video: {video_path}")
        
        options = {
            'output_dir': output_dir,
            'max_suggestions': 3,
            'provider': provider,
            'generate_preview': True,
            'use_semantic_matching': use_semantic_matching,
        }
        
        # Get transcript - use the provided transcript or the sample one
        logger.info("Using sample transcript for demonstration")
        transcript = SAMPLE_TRANSCRIPT
        
        # Process video
        start_time = time.time()
        results = await engine.process_video(video_path, transcript, options)
        end_time = time.time()
        
        # Print results summary
        analysis_results = results.get('analysis_results', {})
        edit_suggestions = results.get('edit_suggestions', {})
        preview_path = results.get('preview_path')
        
        logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"B-Roll count: {edit_suggestions.get('b_roll_count', 0)}")
        logger.info(f"B-Roll total time: {edit_suggestions.get('b_roll_time', 0):.2f} seconds")
        
        # Display semantic concepts if requested
        if display_concepts and use_semantic_matching:
            semantic_insights = results.get('suggestions', {}).get('semantic_insights', {})
            if semantic_insights:
                logger.info("\nSemantic Concepts Analysis:")
                # Display theme consistency
                theme_consistency = semantic_insights.get('theme_consistency', {})
                if theme_consistency:
                    consistency_score = theme_consistency.get('consistency_score', 0.0)
                    logger.info(f"Theme consistency score: {consistency_score:.2f}")
                    
                    logger.info("Consistent themes:")
                    for theme in theme_consistency.get('consistent_themes', []):
                        concept = theme.get('concept', '')
                        consistency = theme.get('consistency', 0.0)
                        logger.info(f"  - {concept}: {consistency:.2f}")
                
                # Display segment concept matches (limited to first 3 segments)
                segments = semantic_insights.get('segments', [])
                if segments:
                    logger.info("\nSegment concept matches (first 3 segments):")
                    for i, segment in enumerate(segments[:3]):
                        logger.info(f"\nSegment {i+1}: \"{segment.get('text', '')[:50]}...\"")
                        concept_matches = segment.get('concept_matches', {})
                        
                        # Show top 3 concept matches
                        for j, (concept, data) in enumerate(list(concept_matches.items())[:3]):
                            confidence = data.get('confidence', 0.0)
                            category = data.get('category', '')
                            visual_suggestions = data.get('visual_suggestions', [])
                            
                            logger.info(f"  Concept: {concept} (confidence: {confidence:.2f}, category: {category})")
                            if visual_suggestions:
                                logger.info(f"    Visual suggestions: {', '.join(visual_suggestions[:3])}")
        
        if preview_path and os.path.exists(preview_path):
            logger.info(f"Preview video generated at: {preview_path}")
        
        # Save full results for inspection
        results_file = os.path.join(output_dir, "detailed_results.json")
        
        with open(results_file, 'w') as f:
            # Create a serializable version of results
            serializable = {
                'video_path': video_path,
                'b_roll_count': edit_suggestions.get('b_roll_count', 0),
                'b_roll_time': edit_suggestions.get('b_roll_time', 0),
                'output_dir': output_dir,
                'preview_path': preview_path,
                'content_analysis': analysis_results.get('content_analysis', {}),
                'timestamp': time.time()
            }
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        return results
    
    finally:
        # Close the engine
        await engine.close()


def main():
    """Run the B-Roll example."""
    parser = argparse.ArgumentParser(description='B-Roll Insertion Example')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--output-dir', help='Directory to save outputs')
    parser.add_argument('--provider', help='Stock footage provider (pexels, pixabay, local)')
    parser.add_argument('--no-semantic', action='store_true', help='Disable semantic matching')
    parser.add_argument('--display-concepts', action='store_true', help='Display semantic concepts in console')
    
    args = parser.parse_args()
    
    # Check if the video file exists
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Run the example
    results = asyncio.run(process_video(
        args.video_path, 
        args.output_dir, 
        args.provider,
        use_semantic_matching=not args.no_semantic,
        display_concepts=args.display_concepts
    ))
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 