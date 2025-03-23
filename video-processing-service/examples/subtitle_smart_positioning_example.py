#!/usr/bin/env python3
"""
Example script demonstrating the smart positioning feature of the subtitle system.

This script:
1. Loads a video and its transcript
2. Analyzes the video content to determine optimal subtitle positions
3. Generates subtitles with smart positioning
4. Renders a video with positioned subtitles

Requirements:
- OpenCV (pip install opencv-python)
- FFmpeg installed and available in PATH
"""

import asyncio
import os
import json
import logging
from pathlib import Path

# Add parent directory to path to import from app
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.services.subtitles import (
    SubtitleService,
    SubtitleFormat,
    TextPosition,
    TextAlignment
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("subtitle_positioning_example")


async def main():
    # Paths
    video_path = "path/to/your/video.mp4"
    transcript_path = "path/to/your/transcript.json"
    output_subtitles = "output/smart_positioned_subtitles.srt"
    output_video = "output/video_with_smart_positioned_subtitles.mp4"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_subtitles), exist_ok=True)
    
    # Load transcript
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)
    
    # Create subtitle service
    subtitle_service = SubtitleService(
        config={
            "ffmpeg_path": "ffmpeg",  # Update with custom path if needed
            "ffprobe_path": "ffprobe",  # Update with custom path if needed
            "default_format": "srt",
            "default_style": "default",
            "positioning_config": {
                "enable_face_detection": True,
                "enable_object_detection": True,
                "enable_text_detection": True,
                "position_preference": ["bottom", "top", "center"],
                "frame_sample_rate": 24  # Analyze 1 frame per second for 24fps video
            }
        }
    )
    
    # Get available positioning options
    positioning_options = subtitle_service.get_positioning_options()
    logger.info(f"Available positioning options: {positioning_options}")
    
    # Method 1: Generate subtitles with smart positioning
    logger.info("Generating subtitles with smart positioning...")
    subtitles_path = await subtitle_service.generate_subtitles_with_smart_positioning(
        video_path=video_path,
        transcript=transcript,
        output_path=output_subtitles,
        format=SubtitleFormat.SRT,
        style_name="film",  # Using the film style template
        adjust_timing=True,  # Adjust timing based on reading speed
        detect_emphasis=True  # Detect and format emphasized text
    )
    logger.info(f"Subtitles generated at: {subtitles_path}")
    
    # Method 2: Render video with smart positioning
    logger.info("Rendering video with smart positioned subtitles...")
    output_video_path = await subtitle_service.render_video_with_smart_positioning(
        video_path=video_path,
        transcript=transcript,
        output_path=output_video,
        style_name="film",
        background_blur=True,  # Add background blur for better readability
        quality="high"  # High quality rendering
    )
    logger.info(f"Video rendered at: {output_video_path}")
    
    # Example of updating positioning config
    logger.info("Updating positioning configuration...")
    subtitle_service.update_positioning_config({
        "position_preference": ["top", "bottom", "center"],  # Prefer top positioning
        "enable_face_detection": True
    })
    
    logger.info("Smart positioning example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main()) 