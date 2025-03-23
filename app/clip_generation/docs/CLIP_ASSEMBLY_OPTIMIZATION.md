# Clip Assembly & Optimization

The Clip Assembly & Optimization module provides advanced functionality for generating polished video clips with smart duration optimization, vertical format transformation, and platform-specific processing.

## Overview

This module enhances the basic clip generation process with intelligent features that ensure the final clips are optimized for engagement and platform-specific requirements. The key components include:

1. **Smart Clip Generation Algorithm**: Intelligently optimizes clip duration and identifies natural start/end points.
2. **Vertical Format Optimization (9:16)**: Transforms horizontal content to vertical format for mobile platforms.
3. **Final Processing Pipeline**: Applies professional-grade audio and video enhancements.
4. **Advanced Face-Aware Cropping**: Ensures faces remain centered in vertical videos.

## Smart Clip Generation Algorithm

The smart clip generation algorithm optimizes clips for maximum engagement:

### Duration Optimization

- **Duration Range Management**: Enforces minimum (5s) and maximum (60s) duration constraints.
- **Target Duration Prioritization**: Prioritizes clips between 15-30 seconds, which tend to have optimal engagement.
- **Maximum Clip Length Enforcement**: Enforces an absolute maximum of 120 seconds if needed.
- **Score-Based Prioritization**: When trimming is necessary, preserves the highest-scoring moments.

### Optimal Start/End Point Detection

- **Silence Detection**: Identifies natural silence points to create seamless clip boundaries.
- **Scene Change Detection**: Looks for visual transitions for optimal cutting points.
- **Expanded Search Range**: Examines ±2 seconds around specified start/end times to find optimal transition points.

### Content-Aware Transitions

- **Crossfading Transitions**: Applies subtle audio and video crossfades between segments.
- **Audio-Driven Cuts**: Aligns cuts with natural pauses in speech.
- **Multi-Segment Assembly**: Intelligently combines multiple interesting moments into a cohesive clip.

## Vertical Format Optimization (9:16)

The module provides sophisticated transformation of horizontal content to vertical format:

### Aspect Ratio Conversion

- **Smart Crop Detection**: Identifies the most important content regions to preserve during cropping.
- **Dynamic Composition Adjustment**: Adapts framing based on the content type and subject position.
- **Content-Aware Scaling**: Applies appropriate scaling and cropping to maintain visual integrity.

### Vertical Viewing Optimization

- **Dynamic Zooming**: Applies subtle zoom to emphasize important elements.
- **Panning Optimization**: Simulates camera movement to keep subjects centered when needed.
- **Text/Graphic Repositioning**: Ensures any on-screen text or graphics remain visible.

### Mobile Platform Adaptation

- **Platform-Specific Presets**: Includes optimizations for specific platforms (TikTok, Instagram, YouTube Shorts).
- **Device-Optimized Rendering**: Ensures content looks good on smaller screens with appropriate contrast and saturation.

## Final Processing Pipeline

The final processing pipeline applies professional-grade enhancements:

### Seamless Transitions

- **Crossfade Implementation**: Applies audio and video crossfades between segments.
- **J-Cut/L-Cut Simulation**: Creates professional transitions by offsetting audio and video cuts slightly.
- **Motion-Aware Transitions**: Respects motion within the scene for more natural transitions.

### Color Enhancement

- **Color Grading Optimization**: Enhances contrast, saturation, and brightness for mobile viewing.
- **Platform-Specific Color Profiles**: Adapts color grading based on target platform specifications.
- **HDR to SDR Conversion**: Properly maps HDR content to standard dynamic range when needed.

### Audio Optimization

- **Audio Normalization**: Normalizes audio to a target loudness of -14 LUFS for mobile playback.
- **Dynamic Range Compression**: Applies appropriate compression to ensure clarity on mobile speakers.
- **Background Noise Reduction**: Reduces ambient noise while preserving speech clarity.

## Advanced Face-Aware Cropping

The advanced face-aware cropping feature ensures that faces remain properly framed in vertical videos:

### Face Detection

- **Multi-Face Tracking**: Detects and tracks all faces in the video segment.
- **Weighted Face Prioritization**: Prioritizes larger faces that are likely the main subjects.
- **Temporal Face Analysis**: Analyzes multiple frames to ensure stable tracking across time.

### Smart Crop Positioning

- **Face-Centered Cropping**: Automatically positions the crop window to keep faces centered.
- **Dynamic Reframing**: Adjusts the crop position as faces move throughout the clip.
- **Optimal Region Selection**: Calculates the best crop region that maximizes face visibility.

### Implementation Approach

- **Sample Frame Analysis**: Extracts multiple frames from the clip for face detection.
- **Weighted Center Calculation**: Computes the weighted center point based on face sizes and positions.
- **Intelligent Fallback**: Uses center crop when no faces are detected.

## Key Classes

### `ClipAssemblyConfig`

Configuration class with customizable settings:

```python
@dataclass
class ClipAssemblyConfig:
    # General settings
    ffmpeg_path: str = "ffmpeg"
    temp_dir: str = "temp/assembly"
    output_dir: str = "output/clips"
    
    # Smart clip generation settings
    min_clip_duration: float = 5.0
    max_clip_duration: float = 60.0
    target_min_duration: float = 15.0
    target_max_duration: float = 30.0
    enforce_absolute_max: float = 120.0  # Hard maximum
    
    # Vertical format settings
    enable_vertical_optimization: bool = True
    vertical_aspect_ratio: str = "9:16"
    smart_crop_detection: bool = True
    
    # Final processing settings
    target_audio_lufs: float = -14.0
    enable_color_optimization: bool = True
    enable_audio_normalization: bool = True
```

### `ClipAssemblyOptimizer`

Main class that implements the basic optimization functionality:

```python
# Initialize the optimizer
optimizer = ClipAssemblyOptimizer(config)

# Generate a single smart clip
output_path = optimizer.generate_smart_clip(
    source_video="input.mp4",
    output_path="output.mp4",
    start_time=10.5,
    end_time=25.0,
    optimize_endpoints=True,
    vertical_format=True,
    audio_normalize=True
)

# Assemble multiple moments into a cohesive clip
output_path = optimizer.assemble_multi_moment_clip(
    source_video="input.mp4",
    output_path="assembled.mp4",
    moments=moments,
    vertical_format=True,
    target_duration=30,
    optimize_transitions=True
)
```

### `AdvancedClipAssemblyOptimizer`

Extended class that adds face-aware cropping and other advanced features:

```python
# Initialize the advanced optimizer
advanced_optimizer = AdvancedClipAssemblyOptimizer(config)

# Generate a vertical clip with face-aware cropping
output_path = advanced_optimizer.generate_face_aware_vertical_clip(
    source_video="input.mp4",
    output_path="face_aware_vertical.mp4",
    start_time=10.5,
    end_time=25.0,
    audio_normalize=True
)
```

## Usage Examples

### Basic Smart Clip Generation

```python
from app.clip_generation.services.clip_assembly import ClipAssemblyOptimizer, ClipAssemblyConfig

# Create optimizer config
config = ClipAssemblyConfig(
    temp_dir="temp/assembly",
    output_dir="output/clips",
    target_min_duration=15.0,
    target_max_duration=30.0
)

# Initialize optimizer
optimizer = ClipAssemblyOptimizer(config)

# Generate a smart clip with optimal endpoints
output_path = optimizer.generate_smart_clip(
    "input.mp4",
    "output.mp4",
    start_time=60.0,
    end_time=75.0,
    optimize_endpoints=True,
    vertical_format=False,
    audio_normalize=True
)
```

### Vertical Format Conversion

```python
# Generate a vertical format clip optimized for mobile viewing
vertical_clip = optimizer.generate_vertical_clip(
    "input.mp4",
    "vertical.mp4",
    start_time=60.0,
    end_time=75.0,
    audio_normalize=True
)
```

### Face-Aware Vertical Format

```python
from app.clip_generation.services.clip_assembly_optimizer import AdvancedClipAssemblyOptimizer

# Initialize advanced optimizer
advanced_optimizer = AdvancedClipAssemblyOptimizer(config)

# Generate a face-aware vertical clip
face_aware_clip = advanced_optimizer.generate_face_aware_vertical_clip(
    "input.mp4",
    "face_aware_vertical.mp4",
    start_time=60.0,
    end_time=75.0,
    audio_normalize=True
)
```

### Multi-Moment Assembly

```python
from app.clip_generation.services.moment_detection import MomentAnalyzer

# Initialize moment analyzer
moment_analyzer = MomentAnalyzer()

# Detect interesting moments
moments = moment_analyzer.analyze_video("input.mp4")

# Convert to dictionary format for the optimizer
moment_dicts = [moment.to_dict() for moment in moments]

# Assemble moments into a cohesive clip
assembled_clip = optimizer.assemble_multi_moment_clip(
    "input.mp4",
    "assembled.mp4",
    moment_dicts,
    vertical_format=True,
    target_duration=30,
    optimize_transitions=True
)
```

## Command-Line Interface

The module includes a demo script for command-line usage:

```bash
# Basic usage
python -m app.clip_generation.demo.clip_assembly_demo input.mp4 --vertical --assembly --duration 30

# With face-aware cropping
python -m app.clip_generation.demo.clip_assembly_demo input.mp4 --vertical --face-aware --advanced
```

Available options:
- `--output-dir`, `-o`: Directory to save output clips
- `--vertical`, `-v`: Generate vertical format clips
- `--assembly`, `-a`: Demonstrate multi-moment assembly
- `--duration`, `-d`: Target duration for assembled clip
- `--face-aware`, `-f`: Use face-aware cropping for vertical format
- `--advanced`: Use advanced optimization features
- `--debug`: Enable debug logging

## Implementation Details

### Smart Endpoint Detection

The `_detect_optimal_endpoints` method uses FFmpeg's silencedetect filter to find natural silence points near the specified start and end times:

```python
silence_cmd = [
    ffmpeg_path,
    "-i", audio_file,
    "-af", "silencedetect=noise=-30dB:d=0.5",
    "-f", "null",
    "-"
]
```

This identifies segments where the audio drops below -30dB for at least 0.5 seconds, which typically indicates good cutting points.

### Vertical Format Transformation

The vertical format transformation uses FFmpeg's crop and scale filters:

```python
# For horizontal to vertical conversion
crop_x = original_width / 2 - target_width / 2
video_filter = (
    f"crop={target_width}:{target_height}:{crop_x}:0,"
    f"scale={int(target_width)}:{int(target_height)}"
)
```

### Face-Aware Cropping

The face-aware cropping uses OpenCV for face detection:

```python
# Detect faces using Haar cascades
faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Calculate weighted center of all faces
total_weight = sum(face["area"] for face in faces)
weighted_center_x = sum(face["center_x"] * face["area"] for face in faces) / total_weight

# Calculate crop region
crop_x = int(weighted_center_x - target_width / 2)
```

### Audio Normalization

Audio normalization uses FFmpeg's loudnorm filter, which implements the EBU R128 standard:

```python
audio_filter = f"loudnorm=I={target_lufs}:LRA=11:TP=-1.5"
```

This ensures consistent audio levels across all generated clips, optimized for mobile playback. 