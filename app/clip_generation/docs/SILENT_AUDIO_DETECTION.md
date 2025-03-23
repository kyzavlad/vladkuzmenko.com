# Silent/Unnecessary Audio Detection System

This document describes the Silent/Unnecessary Audio Detection system implemented for the Clip Generation Microservice. This system provides advanced audio analysis capabilities to identify and remove silent segments and unnecessary sounds from video content.

## Features

The Silent/Unnecessary Audio Detection system offers the following features:

### 1. Advanced Audio Analysis Pipeline

- **Spectral Analysis**: Background noise profiling to establish noise floor levels and adaptive thresholds.
- **Voice Activity Detection (VAD)**: High-precision speech detection with 50ms granularity, supporting multiple VAD models.
- **Speaker Diarization**: Optional detection and tracking of different speakers in multi-person videos.
- **Non-speech Sound Classification**: Identification of music, ambient noise, and sound effects.

### 2. Intelligent Silence Detection

- **Adaptive Thresholding**: Automatically adjusts silence threshold based on ambient noise level.
- **Contextual Awareness**: Preserves intentional pauses while removing excessive silence.
- **Configurable Duration Parameters**: Fine-tune minimum and maximum silence durations (default: 0.3s - 2.0s).
- **Energy-based Detection**: Uses spectral energy profiles to distinguish between silence and low-volume speech.

### 3. Filler Sound Removal

- **Language-specific Filler Word Detection**: Identifies common filler words ("um", "uh", "like") in multiple languages.
- **Hesitation Pattern Recognition**: Detects repetitions, stuttering, and hesitation patterns.
- **Breath/Mouth Sound Detection**: Identifies and optionally removes breath sounds and mouth noises.
- **Transcription-based Analysis**: Uses speech-to-text to locate filler sounds with precise timestamps.

### 4. Processing Options

- **Removal Mode**: Completely removes identified segments for maximum time reduction.
- **Speed-up Mode**: Accelerates unnecessary segments instead of removing them, maintaining context.
- **Batch Processing**: Process multiple videos in parallel for efficiency.
- **Detailed Reporting**: Generates comprehensive reports about detected segments and time savings.

## Architecture

The system consists of several interconnected components:

1. **AudioAnalyzer**: Base class providing foundational audio processing capabilities.
2. **VADProcessor**: Detects speech segments using state-of-the-art voice activity detection models.
3. **SpectralAnalyzer**: Performs spectral analysis for noise profiling and sound classification.
4. **FillerWordDetector**: Identifies filler words, hesitations, and other unnecessary sounds.
5. **SilenceDetector**: Integrates all analytical components to identify removable segments.
6. **SilenceProcessor**: Handles the actual processing of videos to remove or speed up silent segments.

## Usage

### Command-line Interface

The system can be used via the provided demo script:

```bash
python app/clip_generation/scripts/silence_detection_demo.py --input /path/to/video.mp4 --output /path/to/output
```

#### Key Command-line Options

```
# Basic options
--input, -i          Input video file or directory
--output, -o         Output directory (default: ./output)
--mode, -m           Processing mode: 'remove' or 'speedup' (default: remove)

# Processing options
--threshold, -t      Minimum duration threshold for removal (default: 0.5s)
--max-gap, -g        Maximum gap between segments to merge (default: 0.1s)
--speed-factor, -s   Speed factor for silence in speedup mode (default: 2.0)

# Audio analysis options
--min-silence        Minimum silence duration to detect (default: 0.3s)
--max-silence        Maximum silence duration to keep (default: 2.0s)
--silence-threshold  dB threshold for silence detection (default: -35.0)
--adaptive-threshold Use adaptive threshold based on noise

# Filler detection options
--detect-fillers     Enable filler word detection
--language           Language code for filler detection (default: en)

# Performance options
--parallel           Enable parallel processing
--workers            Number of worker threads (default: 4)
--device             Device for ML inference: 'cpu' or 'cuda'
```

### Programmatic Usage

The system can also be used programmatically in your Python code:

```python
from app.clip_generation.services.audio_analysis.silence_detector import SilenceDetectorConfig
from app.clip_generation.services.audio_analysis.silence_processor import SilenceProcessor, SilenceProcessorConfig

# Create configurations
detector_config = SilenceDetectorConfig(
    min_silence_duration=0.3,
    max_silence_duration=2.0,
    adaptive_threshold=True,
    enable_filler_detection=True
)

processor_config = SilenceProcessorConfig(
    output_dir="output",
    removal_threshold=0.5,
    speed_up_silence=False,  # Remove silence instead of speeding up
    silence_detector_config=detector_config
)

# Initialize processor
processor = SilenceProcessor(processor_config)

# Process a video
result_path = processor.process_video("input_video.mp4")
print(f"Processed video saved to: {result_path}")

# Or process multiple videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
report = processor.batch_process(video_paths)
print(f"Reduction: {report['summary']['reduction_percentage']}%")
```

## Configuration Options

### SilenceDetectorConfig

The `SilenceDetectorConfig` class configures the silence detection process:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_silence_duration` | Minimum silence duration to detect (seconds) | 0.3 |
| `max_silence_duration` | Maximum silence duration to keep (seconds) | 2.0 |
| `silence_threshold` | dB threshold for silence detection | -35.0 |
| `adaptive_threshold` | Use adaptive threshold based on noise profile | True |
| `vad_model` | VAD model to use (silero, webrtc, pyannote) | "silero" |
| `vad_mode` | VAD aggressiveness level (0-3) | 3 |
| `enable_filler_detection` | Enable filler word detection | True |
| `language` | Language for filler detection | "en" |
| `enable_spectral_analysis` | Enable spectral analysis | True |
| `enable_speaker_diarization` | Enable speaker diarization | False |
| `device` | Device for inference (cpu, cuda) | "cpu" |

### SilenceProcessorConfig

The `SilenceProcessorConfig` class configures the video processing:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Output directory for processed files | "output" |
| `removal_threshold` | Duration threshold for removal (seconds) | 0.5 |
| `max_segment_gap` | Maximum gap between segments to merge (seconds) | 0.1 |
| `crossfade_duration` | Crossfade duration between segments (seconds) | 0.05 |
| `speed_up_silence` | Speed up instead of removing silence | False |
| `speed_factor` | Speed factor for silence (when not removing) | 2.0 |
| `parallel_processing` | Enable parallel processing | True |
| `max_workers` | Maximum number of worker threads | 4 |
| `preserve_video_quality` | Preserve original video quality | True |
| `generate_report` | Generate processing report | True |

## Integrating with Clip Generation Service

The Silent/Unnecessary Audio Detection system is integrated with the Clip Generation Microservice as follows:

1. **Task Parameters**: The clip generation task can include silence detection parameters:
   ```json
   {
     "source_video": "video.mp4",
     "clip_start": 10.5,
     "clip_end": 45.2,
     "remove_silence": true,
     "silence_config": {
       "min_silence": 0.3,
       "max_silence": 2.0,
       "detect_fillers": true
     }
   }
   ```

2. **Processing Pipeline**: The silence detection is applied during the clip generation process:
   - Extract the clip from the source video
   - Apply silence detection and processing
   - Apply other effects (if any)
   - Generate the final output

3. **Status Updates**: The task processor provides status updates specific to silence detection:
   ```json
   {
     "status": "processing",
     "step": "silence_detection",
     "progress": 75,
     "stats": {
       "identified_segments": 12,
       "removable_segments": 8,
       "estimated_reduction": "18.5%"
     }
   }
   ```

## Dependencies

The Silent/Unnecessary Audio Detection system requires the following key dependencies:

- **FFmpeg**: For video and audio processing
- **NumPy**: For numerical operations
- **Librosa**: For audio analysis
- **PyTorch**: For ML-based audio processing (VAD models)
- **Transformers**: For speech-to-text transcription
- **Matplotlib**: For visualization (optional)

## Performance Considerations

1. **Processing Time**: Silence detection and removal typically adds 10-30% to the overall clip generation time, depending on the options enabled.
   
2. **Memory Usage**: The system is designed to process audio in a streaming fashion to minimize memory usage.
   
3. **GPU Acceleration**: ML components (VAD, transcription) can utilize GPU acceleration when available, significantly improving performance.
   
4. **Batch Processing**: When processing multiple videos, parallel processing can be enabled to utilize multi-core CPUs effectively.

## Examples and Use Cases

### Example 1: Removing Long Silences

For a vlog or instructional video with frequent pauses:
```bash
python silence_detection_demo.py -i input.mp4 -o output --min-silence 0.5 --max-silence 1.0
```

### Example 2: Speeding Up Silences

For a podcast or interview with natural pauses you want to preserve but shorten:
```bash
python silence_detection_demo.py -i input.mp4 -o output -m speedup -s 1.5
```

### Example 3: Removing Filler Words

For a presentation with many filler words:
```bash
python silence_detection_demo.py -i input.mp4 -o output --detect-fillers --language en
```

### Example 4: Batch Processing

For processing multiple videos with the same settings:
```bash
python silence_detection_demo.py -i /path/to/videos/ -o output --parallel --workers 8
```

## Future Enhancements

- **Enhanced ML Models**: Improved neural models for more accurate silence and filler detection
- **Real-time Processing**: Support for real-time silence detection during recording
- **Contextual Analysis**: More advanced analysis of speech context to better preserve important pauses
- **User Preferences**: Learning from user preferences to customize silence detection parameters
- **Additional Languages**: Expanding filler word detection to more languages 