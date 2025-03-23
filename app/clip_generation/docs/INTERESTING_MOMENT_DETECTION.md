# Interesting Moment Detection

The Interesting Moment Detection system is a sophisticated component of the Clip Generation Microservice that automatically identifies engaging moments in video content. This document outlines the system's architecture, features, and usage instructions.

## Overview

The system employs multiple analysis strategies to detect moments that are likely to engage viewers. These include:

1. **Multi-faceted Content Analysis**
   - Audio energy peak detection
   - Voice tone and emphasis analysis
   - Laughter and reaction detection
   - Transcript sentiment analysis
   - Keyword/phrase importance scoring
   - Gesture and facial expression detection

2. **Engagement Prediction**
   - Fine-tuned transformer models for predicting viewer engagement
   - Content category-specific scoring

3. **Narrative Cohesion Analysis**
   - Key point extraction from transcript
   - Topic segmentation
   - Complete thought preservation

## Architecture

The system follows a modular architecture with these main components:

### MomentAnalyzer

The `MomentAnalyzer` is the main entry point and coordinates all detection components. It:
- Combines results from different analyzers
- Ranks and scores potential moments
- Extracts clip segments

### ContentAnalyzer

The `ContentAnalyzer` handles multi-faceted content analysis through several specialized components:

- **AudioAnalyzer**: Detects audio energy peaks and other audio features
- **VoiceAnalyzer**: Analyzes voice tone, emphasis, and detects laughter/reactions
- **TranscriptAnalyzer**: Analyzes transcript for sentiment and important keywords

## Transcript Analysis

The `TranscriptAnalyzer` component specifically focuses on extracting valuable insights from video transcripts, including:

### Sentiment Analysis

The `SentimentAnalyzer` identifies emotionally charged segments in the transcript:
- Detects sentiment peaks (both positive and negative)
- Identifies emotional reactions
- Tracks sentiment transitions

### Keyword Analysis

The `KeywordAnalyzer` identifies important keywords and phrases:
- Scores keywords based on relevance and importance
- Identifies named entities (people, places, organizations)
- Detects domain-specific terminology

## Usage

### Basic Usage

```python
from app.clip_generation.services.moment_detection import MomentAnalyzer, MomentAnalyzerConfig

# Create analyzer configuration
config = MomentAnalyzerConfig(
    temp_dir="temp",
    output_dir="output",
    min_moment_duration=3.0,
    max_moment_duration=15.0,
    min_detection_score=0.6
)

# Initialize analyzer
analyzer = MomentAnalyzer(config)

# Analyze video
moments = analyzer.analyze_video("path/to/video.mp4", "path/to/transcript.json")

# Extract highlight clips
highlights = analyzer.extract_highlights(
    "path/to/video.mp4",
    "output/directory",
    max_highlights=5
)
```

### Command-line Interface

The system includes a demo script for command-line usage:

```bash
python -m app.clip_generation.demo.moment_detection_demo path/to/video.mp4 --transcript path/to/transcript.json --extract-clips --visualize
```

Available options:
- `--transcript`, `-t`: Path to transcript file (optional)
- `--output-dir`, `-o`: Directory to save outputs
- `--extract-clips`, `-e`: Extract highlight clips
- `--visualize`, `-v`: Generate visualization
- `--clip-count`, `-c`: Number of highlight clips to extract
- `--min-duration`, `-min`: Minimum clip duration in seconds
- `--max-duration`, `-max`: Maximum clip duration in seconds
- `--debug`, `-d`: Enable debug logging

## Configuration Options

The `MomentAnalyzerConfig` class provides numerous configuration options:

### General Settings
- `temp_dir`: Directory for temporary files
- `output_dir`: Directory for output files
- `ffmpeg_path`: Path to FFmpeg binary
- `device`: Computation device ("cpu" or "cuda")

### Component Enablement
- `enable_audio_analysis`: Enable audio analysis
- `enable_transcript_analysis`: Enable transcript analysis
- `enable_visual_analysis`: Enable visual analysis
- `enable_engagement_prediction`: Enable engagement prediction
- `enable_narrative_analysis`: Enable narrative analysis

### Thresholds and Parameters
- `min_moment_duration`: Minimum duration of detected moments (seconds)
- `max_moment_duration`: Maximum duration of detected moments (seconds)
- `min_detection_score`: Minimum score for a moment to be included
- `sentiment_threshold`: Threshold for sentiment peaks
- `sentiment_window_size`: Window size for sentiment analysis
- `keyword_importance_threshold`: Threshold for important keywords

## Transcript Format

The system supports various transcript formats:

### JSON Format
```json
[
  {
    "start": 0.0,
    "end": 5.0,
    "text": "Welcome to our video on interesting moment detection."
  },
  {
    "start": 5.1,
    "end": 10.0,
    "text": "This system can automatically identify engaging content."
  }
]
```

### VTT Format
```
WEBVTT

00:00:00.000 --> 00:00:05.000
Welcome to our video on interesting moment detection.

00:00:05.100 --> 00:00:10.000
This system can automatically identify engaging content.
```

### SRT Format
```
1
00:00:00,000 --> 00:00:05,000
Welcome to our video on interesting moment detection.

2
00:00:05,100 --> 00:00:10,000
This system can automatically identify engaging content.
```

## Output Format

The system generates JSON output with detailed information about detected moments:

```json
{
  "video_path": "path/to/video.mp4",
  "analysis_duration_seconds": 15.2,
  "moment_count": 10,
  "moments": [
    {
      "start_time": 30.5,
      "end_time": 42.3,
      "duration": 11.8,
      "combined_score": 0.85,
      "scores": [
        {
          "type": "sentiment_peak",
          "score": 0.9,
          "confidence": 0.85,
          "metadata": {
            "sentiment_value": 0.75,
            "sentiment_text": "This is absolutely amazing!"
          }
        },
        {
          "type": "audio_peak",
          "score": 0.8,
          "confidence": 0.95,
          "metadata": {
            "energy_level": 0.82
          }
        }
      ],
      "transcript": "This is absolutely amazing! I can't believe how well it works.",
      "preview_image_path": "output/preview_30.5.jpg"
    },
    // Additional moments...
  ]
}
```

## Visualizations

The visualization capabilities create timeline charts showing:
- Individual moment scores by type
- Combined moment scores
- Top moments highlighted on the timeline

The visualization is saved as a PNG file and provides an intuitive overview of the interesting moments throughout the video.

## Dependencies

- FFmpeg: Required for audio/video processing
- NumPy: For numerical operations
- Matplotlib: For visualizations (optional)
- Librosa: For advanced audio analysis (optional)
- NLTK/spaCy: For text analysis (optional)
- PyTorch: For transformer-based models (optional)

## Performance Considerations

- Processing time depends on video length and enabled components
- Audio analysis is generally faster than visual analysis
- GPU acceleration can significantly improve performance for visual analysis and transformer models
- For very long videos, consider using frame sampling strategies 