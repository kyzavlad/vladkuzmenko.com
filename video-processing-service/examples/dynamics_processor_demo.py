#!/usr/bin/env python3
"""
Dynamics Processor Demo

This script demonstrates how to use the DynamicsProcessor class
to improve audio dynamics.

Usage:
    python dynamics_processor_demo.py input_file [output_file] [--preset PRESET_NAME]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Adjust the path to include the project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our dynamics processor module
from app.services.audio.dynamics_processor import DynamicsProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Dynamics Processor Demo')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('output_file', nargs='?', help='Path to output file (optional)')
    
    # Dynamics processing options
    parser.add_argument('--preset', choices=[
        'voice_broadcast', 'voice_intimate', 'music_master', 
        'dialog_leveler', 'transparent'
    ], help='Dynamics preset to use')
    
    parser.add_argument('--comp-threshold', type=float, default=-24.0, 
                        help='Compression threshold in dB')
    parser.add_argument('--comp-ratio', type=float, default=2.0, 
                        help='Compression ratio')
    parser.add_argument('--target-loudness', type=float, default=-14.0, 
                        help='Target loudness in LUFS (-14 is broadcast standard)')
    
    parser.add_argument('--no-compression', action='store_true', 
                        help='Disable compression')
    parser.add_argument('--no-limiting', action='store_true', 
                        help='Disable limiting')
    parser.add_argument('--apply-expansion', action='store_true', 
                        help='Apply expansion')
    parser.add_argument('--apply-gating', action='store_true', 
                        help='Apply noise gating')
    
    parser.add_argument('--analyze-only', action='store_true', 
                        help='Only analyze audio, don\'t process')
    
    parser.add_argument('--dry-wet', type=float, default=1.0, 
                        help='Dry/wet mix (0.0-1.0)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # Determine output path if not provided
    if not args.output_file:
        input_path = Path(args.input_file)
        output_path = input_path.with_name(f"{input_path.stem}_dynamics{input_path.suffix}")
    else:
        output_path = args.output_file
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Create dynamics processor with optional preset
    processor = DynamicsProcessor(
        preset=args.preset,
        comp_threshold=args.comp_threshold,
        comp_ratio=args.comp_ratio,
    )
    
    # First analyze the audio
    logger.info(f"Analyzing audio dynamics in {args.input_file}...")
    analysis = processor.analyze_dynamics(args.input_file)
    
    # Print analysis results
    logger.info("Audio Dynamics Analysis:")
    
    if analysis.get("status") == "success":
        levels = analysis.get("levels", {})
        logger.info(f"  Peak Level: {levels.get('peak_db', 0):.1f} dB")
        logger.info(f"  RMS Level: {levels.get('rms_db', 0):.1f} dB")
        logger.info(f"  Loudness: {levels.get('loudness_LUFS', 0):.1f} LUFS")
        logger.info(f"  Crest Factor: {levels.get('crest_factor_db', 0):.1f} dB")
        logger.info(f"  Dynamic Range: {levels.get('dynamic_range_db', 0):.1f} dB")
        logger.info(f"  Noise Floor: {levels.get('noise_floor_db', 0):.1f} dB")
        
        # Show spectral balance
        spectral = analysis.get("spectral_balance", {})
        logger.info("  Spectral Balance:")
        logger.info(f"    Low: {spectral.get('low_pct', 0):.1f}%")
        logger.info(f"    Mid: {spectral.get('mid_pct', 0):.1f}%")
        logger.info(f"    High: {spectral.get('high_pct', 0):.1f}%")
        
        # Show content type
        content = analysis.get("content_type", {})
        for content_type, detected in content.items():
            if detected:
                logger.info(f"  Detected Content Type: {content_type}")
                break
        
        # Show suggested settings
        suggested = analysis.get("suggested_settings", {})
        if suggested:
            logger.info("  Suggested Processing Settings:")
            if "preset" in suggested:
                logger.info(f"    Preset: {suggested['preset']}")
            if "compression" in suggested:
                comp = suggested["compression"]
                logger.info(f"    Compression: threshold={comp.get('threshold', 0):.1f}dB, ratio={comp.get('ratio', 0):.1f}")
            if "target_loudness" in suggested:
                logger.info(f"    Target Loudness: {suggested['target_loudness']} LUFS")
            if "gating" in suggested:
                logger.info(f"    Gating: {suggested['gating']}")
    else:
        logger.error(f"Analysis failed: {analysis.get('error', 'Unknown error')}")
    
    # Stop here if only analysis is requested
    if args.analyze_only:
        return 0
    
    # Process the audio
    logger.info(f"Processing audio dynamics in {args.input_file}...")
    
    # Get suggested preset if not specified by user
    if not args.preset and analysis.get("status") == "success":
        suggested_preset = analysis.get("suggested_settings", {}).get("preset")
        if suggested_preset:
            logger.info(f"Using suggested preset: {suggested_preset}")
            # Create a new processor with the suggested preset
            processor = DynamicsProcessor(preset=suggested_preset)
    
    # Process the audio
    result = processor.process_audio(
        audio_path=args.input_file,
        output_path=output_path,
        apply_compression=not args.no_compression,
        apply_limiting=not args.no_limiting,
        apply_expansion=args.apply_expansion,
        apply_gating=args.apply_gating,
        target_loudness=args.target_loudness,
        dry_wet_mix=args.dry_wet
    )
    
    # Check result
    if result.get("status") != "success":
        logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
        return 1
    
    # Log results
    logger.info(f"Audio dynamics processing complete!")
    logger.info(f"Output file: {result['output_path']}")
    logger.info(f"Processing steps: {result.get('processing_steps', [])}")
    
    # Show before/after levels
    if "levels" in result:
        levels = result["levels"]
        
        logger.info("Audio Levels Comparison:")
        logger.info("  Input:")
        logger.info(f"    Peak: {levels['input']['peak_dB']:.1f} dB")
        logger.info(f"    RMS: {levels['input']['rms_dB']:.1f} dB")
        logger.info(f"    LUFS: {levels['input']['loudness_LUFS']:.1f}")
        
        logger.info("  Output:")
        logger.info(f"    Peak: {levels['output']['peak_dB']:.1f} dB")
        logger.info(f"    RMS: {levels['output']['rms_dB']:.1f} dB")
        logger.info(f"    LUFS: {levels['output']['loudness_LUFS']:.1f}")
        
        # Calculate changes
        peak_change = levels['output']['peak_dB'] - levels['input']['peak_dB']
        rms_change = levels['output']['rms_dB'] - levels['input']['rms_dB']
        loudness_change = levels['output']['loudness_LUFS'] - levels['input']['loudness_LUFS']
        
        logger.info("  Changes:")
        logger.info(f"    Peak: {peak_change:+.1f} dB")
        logger.info(f"    RMS: {rms_change:+.1f} dB")
        logger.info(f"    LUFS: {loudness_change:+.1f}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1) 