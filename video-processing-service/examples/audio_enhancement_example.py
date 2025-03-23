#!/usr/bin/env python3
"""
Audio Enhancement Example

This script demonstrates how to use the Audio Enhancement Suite
for improving audio quality in videos or audio files.

Usage:
    python audio_enhancement_example.py input_file [output_file]
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Adjust the path to include the project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our audio enhancement modules
from app.services.audio.audio_enhancer import AudioEnhancer
from app.services.audio.noise_reduction import NoiseReducer
from app.services.audio.dynamics_processor import DynamicsProcessor
from app.services.audio.voice_enhancer import VoiceEnhancer
from app.services.audio.environmental_sound_classifier import EnvironmentalSoundClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Audio Enhancement Example')
    parser.add_argument('input_file', help='Path to input audio or video file')
    parser.add_argument('output_file', nargs='?', help='Path to output file (optional)')
    parser.add_argument('--temp-dir', help='Temporary directory for processing', default='/tmp/audio_enhancement')
    parser.add_argument('--ffmpeg-path', help='Path to ffmpeg binary', default='ffmpeg')
    parser.add_argument('--ffprobe-path', help='Path to ffprobe binary', default='ffprobe')
    
    # Noise reduction options
    parser.add_argument('--no-noise-reduction', action='store_true', help='Disable noise reduction')
    parser.add_argument('--no-auto-noise', action='store_true', help='Disable automatic noise detection')
    parser.add_argument('--noise-sample', help='Path to noise sample (if --no-auto-noise is used)')
    parser.add_argument('--noise-strength', type=float, default=0.75, help='Noise reduction strength (0.0-1.0)')
    
    # Voice enhancement options
    parser.add_argument('--apply-voice-enhancement', action='store_true', help='Apply voice enhancement')
    parser.add_argument('--male-voice', action='store_true', help='Optimize for male voice')
    parser.add_argument('--female-voice', action='store_true', help='Optimize for female voice')
    
    # Dynamics processing options
    parser.add_argument('--apply-dynamics', action='store_true', help='Apply dynamics processing')
    parser.add_argument('--dynamics-preset', choices=['voice_broadcast', 'voice_intimate', 'music_master', 'dialog_leveler', 'transparent'], 
                        help='Dynamics processing preset')
    parser.add_argument('--target-loudness', type=float, default=-14, help='Target loudness in LUFS (-14 is broadcast standard)')
    
    # Analysis options
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze audio, don\'t process')
    
    return parser.parse_args()

async def main():
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # Create the output directory if needed
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Ensure temp directory exists
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Create the audio enhancer configuration
    config = {
        'temp_dir': args.temp_dir,
        'ffmpeg_path': args.ffmpeg_path,
        'ffprobe_path': args.ffprobe_path,
        'noise_reduction': {
            'reduction_strength': args.noise_strength,
            'n_fft': 2048,
            'win_length': 2048,
            'hop_length': 512,
            'n_std_thresh': 1.5,
            'freq_mask_smooth_hz': 500,
            'time_mask_smooth_ms': 50
        },
        'voice_enhancement': {
            'male_voice_boost': args.male_voice,
            'female_voice_boost': args.female_voice,
            'clarity': 0.4,
            'warmth': 0.3
        },
        'dynamics_processing': {
            'preset': args.dynamics_preset,
            'comp_threshold': -24.0,
            'comp_ratio': 2.0,
            'limit_threshold': -1.5
        }
    }
    
    # Create the audio enhancer
    enhancer = AudioEnhancer(config)
    
    # Display available enhancements
    enhancements = enhancer.get_available_enhancements()
    logger.info(f"Available enhancements: {enhancements}")
    
    # Analyze the audio first
    logger.info(f"Analyzing audio in {args.input_file}...")
    analysis = await enhancer.analyze_audio(args.input_file)
    
    # Print analysis results
    logger.info("Audio Analysis Results:")
    
    # Print noise analysis
    if 'noise_profile' in analysis:
        noise = analysis['noise_profile']
        logger.info(f"  Noise Level: {noise.get('noise_level_db', 'Unknown')}")
        logger.info(f"  SNR (dB): {noise.get('snr_db', 'Unknown')}")
    
    # Print sound classification
    if 'sound_classification' in analysis and 'dominant_sounds' in analysis['sound_classification']:
        sounds = analysis['sound_classification']['dominant_sounds']
        if sounds:
            logger.info("  Detected sounds:")
            for sound in sounds[:3]:  # Top 3 sounds
                logger.info(f"    - {sound['class']} (confidence: {sound['mean_confidence']:.2f})")
    
    # Print voice analysis
    if 'voice_analysis' in analysis and analysis['voice_analysis']:
        voice = analysis['voice_analysis']
        logger.info(f"  Voice Type: {'Male' if voice.get('likely_male_voice', True) else 'Female'}")
        logger.info(f"  Has Sibilance: {voice.get('has_sibilance', False)}")
        logger.info(f"  Needs Clarity: {voice.get('needs_clarity', False)}")
    
    # Print dynamics analysis
    if 'dynamics_analysis' in analysis and analysis['dynamics_analysis']:
        dynamics = analysis['dynamics_analysis']
        if 'levels' in dynamics:
            levels = dynamics['levels']
            logger.info(f"  Peak Level: {levels.get('peak_db', 0):.1f} dB")
            logger.info(f"  RMS Level: {levels.get('rms_db', 0):.1f} dB")
            logger.info(f"  Dynamic Range: {levels.get('dynamic_range_db', 0):.1f} dB")
            logger.info(f"  Crest Factor: {levels.get('crest_factor_db', 0):.1f} dB")
        
        if 'suggested_settings' in dynamics and 'content_type' in dynamics['suggested_settings']:
            logger.info(f"  Content Type: {dynamics['suggested_settings']['content_type']}")
            if 'preset' in dynamics['suggested_settings']:
                logger.info(f"  Suggested Preset: {dynamics['suggested_settings']['preset']}")
    
    # Print recommendations
    if 'recommendations' in analysis:
        recommendations = analysis['recommendations']
        logger.info("  Recommendations:")
        
        for process, rec in recommendations.items():
            if rec.get('apply', False):
                logger.info(f"    - Apply {process.replace('_', ' ')}: {rec.get('reason', '')}")
    
    # Stop here if only analysis is requested
    if args.analyze_only:
        return 0
    
    # Determine output path if not provided
    if not args.output_file:
        input_path = Path(args.input_file)
        output_path = input_path.with_name(f"{input_path.stem}_enhanced{input_path.suffix}")
    else:
        output_path = args.output_file
    
    # Set up enhancement options
    options = {
        'apply_noise_reduction': not args.no_noise_reduction,
        'noise_reduction': {
            'auto_detect': not args.no_auto_noise,
            'strength': args.noise_strength
        },
        'apply_voice_enhancement': args.apply_voice_enhancement,
        'voice_enhancement': {
            'male_voice_boost': args.male_voice,
            'female_voice_boost': args.female_voice
        },
        'apply_dynamics_processing': args.apply_dynamics,
        'dynamics_processing': {
            'preset': args.dynamics_preset,
            'target_loudness': args.target_loudness
        }
    }
    
    # If a specific noise sample is provided
    if args.noise_sample:
        if not os.path.exists(args.noise_sample):
            logger.error(f"Noise sample file not found: {args.noise_sample}")
            return 1
        options['noise_reduction']['noise_sample'] = args.noise_sample
    
    # Process the audio
    logger.info(f"Enhancing audio from {args.input_file}...")
    
    # Determine if it's a video or audio file
    is_video = Path(args.input_file).suffix.lower() in [
        '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'
    ]
    
    # Process accordingly
    if is_video:
        logger.info("Processing video file...")
        result = await enhancer.enhance_video_audio(
            video_path=args.input_file,
            output_path=str(output_path),
            options=options
        )
    else:
        logger.info("Processing audio file...")
        result = await enhancer.enhance_audio(
            input_path=args.input_file,
            output_path=str(output_path),
            options=options
        )
    
    # Check result
    if "error" in result:
        logger.error(f"Error enhancing audio: {result['error']}")
        return 1
    
    # Log results
    logger.info(f"Audio enhancement complete!")
    logger.info(f"Output file: {result['output_path']}")
    
    # Log processing steps details
    if 'processing_steps' in result:
        logger.info("Processing steps applied:")
        for step in result['processing_steps']:
            logger.info(f"  - {step['step']} ({step['duration']:.2f}s)")
    
    logger.info(f"Total duration: {result.get('duration', 0):.2f} seconds")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1) 