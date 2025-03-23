import pytest
import os
import numpy as np
from src.clip_generator import (
    ClipGenerator,
    ClipGeneratorError,
    InvalidVideoError,
    ProcessingError,
    AudioProcessingError
)

@pytest.fixture
def sample_video_path(tmp_path):
    """Create a dummy video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    # Create an empty file for now
    video_path.touch()
    return str(video_path)

@pytest.fixture
def mock_video_clip(mocker):
    """Create a mock video clip for testing."""
    mock_clip = mocker.MagicMock()
    mock_clip.duration = 60.0
    mock_clip.w = 1920
    mock_clip.h = 1080
    mock_clip.get_frame.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return mock_clip

def test_clip_generator_initialization(sample_video_path):
    """Test basic initialization."""
    generator = ClipGenerator(sample_video_path)
    assert generator.input_path == sample_video_path

def test_invalid_video_path():
    """Test initialization with invalid video path."""
    with pytest.raises(InvalidVideoError):
        ClipGenerator("nonexistent.mp4")

def test_duration_optimization(sample_video_path):
    """Test duration optimization."""
    generator = ClipGenerator(sample_video_path)
    optimized = generator.optimize_duration(target_duration=15.0)
    assert optimized.duration <= 15.0

def test_invalid_duration(sample_video_path):
    """Test duration validation."""
    generator = ClipGenerator(sample_video_path)
    with pytest.raises(ValueError):
        generator.optimize_duration(target_duration=3.0)
    with pytest.raises(ValueError):
        generator.optimize_duration(target_duration=121.0)

def test_vertical_conversion(sample_video_path):
    """Test vertical format conversion."""
    generator = ClipGenerator(sample_video_path)
    vertical = generator.convert_to_vertical(target_width=1080, target_height=1920)
    assert vertical.w == 1080
    assert vertical.h == 1920

def test_invalid_dimensions(sample_video_path):
    """Test dimension validation."""
    generator = ClipGenerator(sample_video_path)
    with pytest.raises(ValueError):
        generator.convert_to_vertical(target_width=50, target_height=1920)
    with pytest.raises(ValueError):
        generator.convert_to_vertical(target_width=1080, target_height=50)
    with pytest.raises(ValueError):
        generator.convert_to_vertical(target_width=4000, target_height=1920)

def test_audio_optimization(sample_video_path):
    """Test audio optimization."""
    generator = ClipGenerator(sample_video_path)
    optimized_audio = generator.optimize_audio(target_lufs=-14.0)
    assert optimized_audio is not None

def test_invalid_lufs(sample_video_path):
    """Test LUFS validation."""
    generator = ClipGenerator(sample_video_path)
    with pytest.raises(ValueError):
        generator.optimize_audio(target_lufs=-32.0)
    with pytest.raises(ValueError):
        generator.optimize_audio(target_lufs=-5.0)

def test_scene_change_detection(sample_video_path, mock_video_clip):
    """Test scene change detection."""
    generator = ClipGenerator(sample_video_path)
    generator.video = mock_video_clip
    scene_changes = generator._analyze_scene_changes()
    assert isinstance(scene_changes, list)

def test_audio_energy_analysis(sample_video_path, mock_video_clip):
    """Test audio energy analysis."""
    generator = ClipGenerator(sample_video_path)
    generator.video = mock_video_clip
    audio_peaks = generator._analyze_audio_energy()
    assert isinstance(audio_peaks, list)

def test_motion_analysis(sample_video_path, mock_video_clip):
    """Test motion analysis."""
    generator = ClipGenerator(sample_video_path)
    generator.video = mock_video_clip
    motion_points = generator._analyze_motion()
    assert isinstance(motion_points, list)

def test_zoom_pan_effects(sample_video_path, mock_video_clip):
    """Test zoom and pan effects."""
    generator = ClipGenerator(sample_video_path)
    generator.video = mock_video_clip
    processed = generator._apply_zoom_pan(mock_video_clip)
    assert processed is not None

def test_full_processing_pipeline(sample_video_path):
    """Test the complete processing pipeline."""
    generator = ClipGenerator(sample_video_path)
    output_path, duration = generator.process_clip(target_duration=30.0)
    assert os.path.exists(output_path)
    assert duration <= 30.0

def test_error_handling(sample_video_path):
    """Test error handling during processing."""
    generator = ClipGenerator(sample_video_path)
    with pytest.raises(ProcessingError):
        # Simulate processing error by corrupting the video
        generator.video = None
        generator.process_clip() 