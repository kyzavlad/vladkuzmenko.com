# Clip Generation Microservice

A powerful, AI-driven microservice for automatically generating engaging video clips from longer content.

## Overview

The Clip Generation Microservice provides a comprehensive solution for analyzing video content and extracting the most engaging moments as short-form clips. This service incorporates multiple analysis strategies, including audio analysis, visual analysis, sentiment analysis, and engagement prediction.

## Key Features

- **Face Tracking System**: Track faces across frames with temporal consistency using Kalman filtering
- **Silent/Unnecessary Audio Detection**: Detect and process silent or unnecessary audio segments
- **Interesting Moment Detection**: Identify engaging moments through multi-faceted content analysis
- **Clip Assembly & Optimization**: Smart clip generation with vertical format optimization and processing pipeline
- **Performance Optimizations**: GPU acceleration, batch processing, and adaptive sampling

## Architecture

The microservice is organized into several key components:

```
app/clip_generation/
├── services/                  # Core service components
│   ├── face_tracking/         # Face tracking services
│   │   ├── face_tracking_kalman.py
│   │   ├── face_tracking_manager.py
│   │   └── face_tracking_optimizer.py
│   ├── audio_analysis/        # Audio analysis services
│   │   ├── audio_analyzer.py
│   │   ├── silence_detector.py
│   │   ├── vad_processor.py
│   │   └── filler_word_detector.py
│   ├── moment_detection/      # Interesting moment detection
│   │   ├── moment_analyzer.py
│   │   ├── content_analysis.py
│   │   ├── voice_analysis.py  
│   │   └── transcript_analysis.py
│   ├── clip_assembly.py       # Clip assembly & optimization
│   └── clip_generator.py      # Main clip generation service
├── utils/                     # Utility functions and helpers
├── models/                    # ML model definitions and weights
├── demo/                      # Demo scripts and examples
│   ├── face_tracking_demo.py
│   ├── silence_detection_demo.py
│   ├── moment_detection_demo.py
│   └── clip_assembly_demo.py
├── tests/                     # Unit and integration tests
├── benchmark/                 # Benchmarking tools
└── docs/                      # Documentation
    ├── FACE_TRACKING_SYSTEM.md
    ├── SILENT_AUDIO_DETECTION.md
    ├── INTERESTING_MOMENT_DETECTION.md
    ├── CLIP_ASSEMBLY_OPTIMIZATION.md
    ├── PERFORMANCE_OPTIMIZATION_GUIDE.md
    └── IMPLEMENTATION_SUMMARY.md
```

## Components

### 1. Face Tracking System

The Face Tracking System provides robust face detection and tracking with temporal consistency across video frames.

- Kalman filter-based tracking for smooth trajectory estimation
- Multi-face tracking with track management
- Performance optimizations for real-time processing

See [Face Tracking System Documentation](docs/FACE_TRACKING_SYSTEM.md) for details.

### 2. Silent/Unnecessary Audio Detection

The Silent/Unnecessary Audio Detection component analyzes audio to identify and process silent segments and unnecessary sounds.

- Advanced audio analysis pipeline with spectral analysis
- Voice activity detection with precision timing
- Intelligent silence detection with adaptive thresholds
- Filler sound removal (um, uh, like, etc.)

See [Silent Audio Detection Documentation](docs/SILENT_AUDIO_DETECTION.md) for details.

### 3. Interesting Moment Detection

The Interesting Moment Detection system automatically identifies engaging moments in video content through multi-faceted analysis.

- Audio energy peak detection
- Voice tone and emphasis analysis
- Transcript sentiment analysis
- Keyword/phrase importance scoring
- Engagement prediction

See [Interesting Moment Detection Documentation](docs/INTERESTING_MOMENT_DETECTION.md) for details.

### 4. Clip Assembly & Optimization

The Clip Assembly & Optimization module generates polished video clips with advanced features for maximum engagement.

- Smart clip generation with duration optimization (5s to 60s, prioritizing 15-30s)
- Vertical format (9:16) conversion with content-aware cropping
- Professional processing pipeline with audio normalization (-14 LUFS) and color optimization

See [Clip Assembly & Optimization Documentation](docs/CLIP_ASSEMBLY_OPTIMIZATION.md) for details.

### 5. Performance Optimizations

The microservice includes various performance optimizations to ensure efficient processing of video content.

- Frame sampling strategies (uniform, adaptive, keyframe-based)
- GPU acceleration via CUDA for computer vision tasks
- Multi-threaded batch processing
- Memory optimization techniques

See [Performance Optimization Guide](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md) for details.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clip-generation-service.git
cd clip-generation-service

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU acceleration dependencies
pip install -r requirements-gpu.txt
```

## Usage

### Basic Usage

```python
from app.clip_generation.services.clip_generator import ClipGenerator
from app.clip_generation.services.clip_assembly import ClipAssemblyOptimizer

# Initialize the components
generator = ClipGenerator()
optimizer = ClipAssemblyOptimizer()

# Generate basic clip
basic_clip = generator.generate_clip(
    source_video="path/to/video.mp4",
    output_path="output/basic_clip.mp4",
    start_time=60.0,
    end_time=75.0
)

# Generate optimized clip with vertical format
optimized_clip = optimizer.generate_smart_clip(
    source_video="path/to/video.mp4",
    output_path="output/optimized_clip.mp4",
    start_time=60.0,
    end_time=75.0,
    optimize_endpoints=True,
    vertical_format=True,
    audio_normalize=True
)
```

### Command-line Interface

The service provides command-line tools for each major component:

```bash
# Face tracking demo
python -m app.clip_generation.demo.face_tracking_demo path/to/video.mp4 --output-dir output

# Silent audio detection demo
python -m app.clip_generation.demo.silence_detection_demo path/to/audio.wav --visualize

# Interesting moment detection demo
python -m app.clip_generation.demo.moment_detection_demo path/to/video.mp4 --extract-clips --visualize

# Clip assembly & optimization demo
python -m app.clip_generation.demo.clip_assembly_demo path/to/video.mp4 --vertical --assembly --duration 30

# Full clip generation
python -m app.clip_generation.cli path/to/video.mp4 --output-dir output --max-clips 5
```

## Configuration

The service can be configured through YAML configuration files or programmatically:

```yaml
# config.yaml
clip_generation:
  face_tracking:
    enabled: true
    process_noise: 0.01
    measurement_noise: 0.1
    
  audio_analysis:
    silence_detection:
      enabled: true
      min_silence_duration: 0.3
      max_silence_duration: 2.0
      
  moment_detection:
    enabled: true
    min_moment_duration: 3.0
    max_moment_duration: 15.0
  
  clip_assembly:
    enable_vertical_optimization: true
    target_min_duration: 15.0
    target_max_duration: 30.0
    target_audio_lufs: -14.0
    
  performance:
    device: "cuda"  # or "cpu"
    batch_size: 8
    sampling_strategy: "adaptive"
```

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- FFmpeg
- PyTorch (optional, for GPU acceleration)
- Librosa (for audio analysis)
- NLTK/spaCy (for text analysis)
- Matplotlib (for visualizations)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

For more detailed information, see the documentation files in the `docs/` directory:

- [Face Tracking System](docs/FACE_TRACKING_SYSTEM.md)
- [Silent Audio Detection](docs/SILENT_AUDIO_DETECTION.md)
- [Interesting Moment Detection](docs/INTERESTING_MOMENT_DETECTION.md)
- [Clip Assembly & Optimization](docs/CLIP_ASSEMBLY_OPTIMIZATION.md)
- [Performance Optimization Guide](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md) 