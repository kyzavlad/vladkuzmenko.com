# Face Tracking Performance Optimization Guide

This guide provides practical advice for optimizing the Face Tracking System performance for different use cases and hardware configurations.

## Quick Reference

| Use Case | Recommended Configuration | Expected Performance |
|----------|---------------------------|----------------------|
| Real-time processing (webcam) | Adaptive sampling, GPU enabled | 20-30 FPS |
| Maximum accuracy (offline) | Uniform sampling (interval=1), Ensemble=parallel | 5-15 FPS |
| Low-resource device | Keyframe sampling, Single model | 10-20 FPS |
| High-throughput batch processing | Uniform sampling, Batch size 4×4 | 100-300 FPM |
| Mobile device | Motion sampling, Single model, No GPU | 8-15 FPS |

## Configuration Parameters

### Core Parameters

```python
tracker = FaceTrackingManager(
    # Performance settings
    sampling_strategy="adaptive",  # Options: "uniform", "adaptive", "keyframe", "motion"
    batch_size=4,                 # Number of frames in each batch
    worker_threads=2,             # Number of parallel workers
    use_gpu=True,                 # Enable GPU acceleration
    
    # Detection settings
    detection_interval=5,         # Process every N frames (for uniform strategy)
    max_faces=10,                 # Maximum faces to track
)
```

### Command-line Options (Demo/Benchmark)

```bash
python -m app.clip_generation.demo_face_tracking \
  --video_path input.mp4 \
  --sampling_strategy adaptive \
  --batch_size 4 \
  --worker_threads 2 \
  --use_gpu
```

## Optimization Strategies

### 1. For Real-time Applications (Video Calls, Live Streaming)

Priority: Low latency, consistent frame rate

```python
tracker = FaceTrackingManager(
    sampling_strategy="adaptive",
    batch_size=2,
    worker_threads=2,
    use_gpu=True,
    detection_interval=3
)
```

Tips:
- Use smaller input resolution (e.g., 640×480)
- Consider using single model detection instead of ensemble
- Set reasonable detection confidence threshold (e.g., 0.4)
- Skip face recognition for maximum performance

### 2. For Maximum Accuracy (Video Analysis, Security)

Priority: Detect and track all faces reliably

```python
tracker = FaceTrackingManager(
    sampling_strategy="uniform",
    detection_interval=1,   # Process every frame
    batch_size=4,
    worker_threads=4,
    use_gpu=True,
    max_faces=20            # Track more faces
)
```

Tips:
- Use ensemble detection in "parallel" mode
- Enable face recognition for consistent identity tracking
- Use full resolution input for best results
- Consider using GPU with >=4GB VRAM

### 3. For Low-resource Environments (Edge Devices, Older Computers)

Priority: Functional tracking with minimal resource usage

```python
tracker = FaceTrackingManager(
    sampling_strategy="keyframe",
    batch_size=1,           # No batching
    worker_threads=0,       # Single-threaded
    use_gpu=False,
    detection_interval=10   # Fallback interval
)
```

Tips:
- Use a single detector model (YOLO recommended for balance of speed/accuracy)
- Reduce input resolution (e.g., 480×360)
- Disable face recognition if not needed
- Use a larger detection interval (8-10)

### 4. For Batch Video Processing (Content Analysis, Video Libraries)

Priority: Process large volumes of video efficiently

```python
tracker = FaceTrackingManager(
    sampling_strategy="keyframe",
    batch_size=8,
    worker_threads=8,        # Use more worker threads
    use_gpu=True,
    detection_interval=15    # Sparse detection to find keyframes
)
```

Tips:
- Adjust worker count to match available CPU cores
- Use larger batch sizes for GPU efficiency
- Consider processing multiple videos in parallel
- Split long videos into segments for parallel processing

## Hardware-specific Recommendations

### CPU-only Systems

- Focus on sampling strategies over batch processing
- Use "keyframe" or "motion" sampling for best efficiency
- Reduce model complexity (use YOLO or MediaPipe, not ensemble)
- Keep resolution modest (720p or lower)
- Set worker threads to (CPU cores - 1)

### NVIDIA GPU Systems

- Enable GPU acceleration
- Use batch processing for better efficiency
- CUDA-enabled OpenCV enhances performance significantly
- For >2GB VRAM: Increase batch_size to 4-8
- For RTX cards: Enable FP16 inference for ~2x speedup

### Apple Silicon (M1/M2)

- Enable Metal performance shaders if available
- CoreML optimizations improve detection speed
- Balance batch_size with thermal constraints
- Use native ARM builds of OpenCV when possible

## Monitoring Performance

Use the benchmark tool to evaluate configurations:

```bash
python -m app.clip_generation.benchmark_face_tracking \
  --video_path sample.mp4 \
  --num_frames 300 \
  --plot_results
```

Key metrics to track:
- Processing FPS
- Percentage of frames processed
- CPU/GPU utilization
- Memory usage
- Detection accuracy (missed faces)

## Troubleshooting

### Common Issues

1. **Poor detection in challenging lighting**
   - Use ensemble detection in "parallel" mode
   - Increase detection confidence threshold
   - Try different ensemble combinations

2. **High CPU usage**
   - Reduce batch_size and worker_threads
   - Use more aggressive sampling strategy
   - Reduce input resolution

3. **GPU memory errors**
   - Reduce batch_size
   - Use smaller models
   - Enable memory optimizations

4. **Inconsistent tracking**
   - Decrease detection_interval
   - Adjust Kalman filter parameters
   - Use uniform sampling instead of adaptive/keyframe

5. **Slow performance with GPU**
   - Ensure GPU acceleration is properly configured
   - Check for CPU bottlenecks in pre/post-processing
   - Increase batch size to leverage GPU parallelism

## Advanced Configuration

Fine-tune the optimizer for specific needs:

```python
from app.clip_generation.services.face_tracking_optimizer import FaceTrackingOptimizer, SamplingStrategy

# Create custom optimizer
optimizer = FaceTrackingOptimizer(
    sampling_strategy=SamplingStrategy.ADAPTIVE,
    sampling_rate=5,
    batch_size=4,
    worker_threads=2,
    motion_threshold=0.05,    # Adjust motion sensitivity
    device="cuda"
)

# Use it in your tracking workflow
tracker = FaceTrackingManager(...)
tracker.optimizer = optimizer
```

## Real-world Performance Examples

| Hardware | Configuration | Video Resolution | Performance |
|----------|---------------|------------------|------------|
| Intel i7-10700K, RTX 3070 | Adaptive, Batch 4×4, GPU | 1080p | 45-60 FPS |
| Intel i5-8400, GTX 1060 | Adaptive, Batch 2×2, GPU | 720p | 25-35 FPS |
| MacBook Pro M1 | Keyframe, Batch 2×2, GPU | 1080p | 20-30 FPS |
| Raspberry Pi 4 (4GB) | Keyframe, No batching, CPU | 480p | 3-5 FPS |
| AWS g4dn.xlarge | Uniform, Batch 8×4, GPU | 1080p | 40-50 FPS |

Remember that performance can vary significantly based on video content, face count, and other factors. Always benchmark with representative content for your specific use case. 