import pytest
import numpy as np
from src.services.clip_generator import ClipGenerator
from src.services.audio_processor import AudioProcessor
from src.services.video_processor import VideoProcessor
from src.services.quality_analyzer import QualityAnalyzer

class TestClipGenerator:
    @pytest.fixture
    def clip_generator(self, test_video_file):
        return ClipGenerator(test_video_file)

    def test_video_metadata_extraction(self, clip_generator):
        """Test video metadata extraction."""
        metadata = clip_generator.get_video_metadata()
        assert isinstance(metadata, dict)
        assert "duration" in metadata
        assert "width" in metadata
        assert "height" in metadata
        assert "fps" in metadata

    def test_duration_calculation(self, clip_generator):
        """Test duration calculation for different target durations."""
        test_durations = [15.0, 30.0, 60.0]
        for target_duration in test_durations:
            segments = clip_generator.calculate_segments(target_duration)
            assert len(segments) > 0
            assert all(0 <= start < end <= target_duration for start, end in segments)

    def test_segment_optimization(self, clip_generator):
        """Test segment optimization algorithm."""
        segments = clip_generator.calculate_segments(30.0)
        optimized = clip_generator.optimize_segments(segments)
        assert len(optimized) <= len(segments)
        assert all(0 <= start < end <= 30.0 for start, end in optimized)

class TestAudioProcessor:
    @pytest.fixture
    def audio_processor(self):
        return AudioProcessor()

    def test_loudness_calculation(self, audio_processor):
        """Test audio loudness calculation."""
        # Create dummy audio data
        audio_data = np.random.rand(44100)  # 1 second of audio at 44.1kHz
        lufs = audio_processor.calculate_lufs(audio_data)
        assert isinstance(lufs, float)
        assert -70 <= lufs <= 0  # Typical LUFS range

    def test_audio_normalization(self, audio_processor):
        """Test audio normalization."""
        target_lufs = -14.0
        audio_data = np.random.rand(44100)
        normalized = audio_processor.normalize_audio(audio_data, target_lufs)
        assert len(normalized) == len(audio_data)
        assert abs(audio_processor.calculate_lufs(normalized) - target_lufs) < 0.5

class TestVideoProcessor:
    @pytest.fixture
    def video_processor(self):
        return VideoProcessor()

    def test_resolution_scaling(self, video_processor):
        """Test video resolution scaling."""
        test_cases = [
            (1920, 1080, 1280, 720),
            (1280, 720, 854, 480),
            (854, 480, 640, 360)
        ]
        for orig_w, orig_h, target_w, target_h in test_cases:
            scaled_w, scaled_h = video_processor.calculate_scaled_dimensions(
                orig_w, orig_h, target_w, target_h
            )
            assert scaled_w <= target_w
            assert scaled_h <= target_h
            assert scaled_w / scaled_h == orig_w / orig_h

    def test_frame_extraction(self, video_processor):
        """Test frame extraction timing."""
        test_duration = 30.0
        fps = 30
        frames = video_processor.calculate_frame_timings(test_duration, fps)
        assert len(frames) == int(test_duration * fps)
        assert all(0 <= t <= test_duration for t in frames)

class TestQualityAnalyzer:
    @pytest.fixture
    def quality_analyzer(self):
        return QualityAnalyzer()

    def test_video_quality_metrics(self, quality_analyzer):
        """Test video quality metrics calculation."""
        # Create dummy video data
        video_data = np.random.rand(1080, 1920, 3)  # One frame
        metrics = quality_analyzer.analyze_video_quality(video_data)
        assert isinstance(metrics, dict)
        assert "sharpness" in metrics
        assert "contrast" in metrics
        assert "noise" in metrics
        assert all(0 <= value <= 1 for value in metrics.values())

    def test_audio_quality_metrics(self, quality_analyzer):
        """Test audio quality metrics calculation."""
        # Create dummy audio data
        audio_data = np.random.rand(44100)
        metrics = quality_analyzer.analyze_audio_quality(audio_data)
        assert isinstance(metrics, dict)
        assert "clarity" in metrics
        assert "dynamic_range" in metrics
        assert "noise_floor" in metrics
        assert all(0 <= value <= 1 for value in metrics.values())

    def test_overall_quality_score(self, quality_analyzer):
        """Test overall quality score calculation."""
        video_metrics = {"sharpness": 0.8, "contrast": 0.7, "noise": 0.9}
        audio_metrics = {"clarity": 0.8, "dynamic_range": 0.7, "noise_floor": 0.9}
        score = quality_analyzer.calculate_overall_score(video_metrics, audio_metrics)
        assert isinstance(score, float)
        assert 0 <= score <= 1 