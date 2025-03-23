import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import psutil
import os

from src.services.clip_generator import ClipGenerator
from src.services.audio_processor import AudioProcessor
from src.services.video_processor import VideoProcessor
from src.services.quality_analyzer import QualityAnalyzer

class TestPerformance:
    @pytest.fixture
    def sample_videos(self):
        """Create sample videos of different sizes and durations."""
        videos = []
        # Create test videos with different characteristics
        # This would require actual video generation or sample files
        return videos

    def test_video_processing_performance(self, sample_videos):
        """Test video processing performance with different input sizes."""
        results = []
        for video in sample_videos:
            start_time = time.time()
            generator = ClipGenerator(video["path"])
            output_path, duration = generator.process_clip(
                target_duration=30.0
            )
            processing_time = time.time() - start_time
            
            results.append({
                "input_size": video["size"],
                "duration": video["duration"],
                "processing_time": processing_time,
                "output_size": os.path.getsize(output_path)
            })
        
        # Calculate performance metrics
        processing_times = [r["processing_time"] for r in results]
        avg_time = statistics.mean(processing_times)
        std_dev = statistics.stdev(processing_times)
        
        # Assert performance requirements
        assert avg_time < 60.0  # Average processing time should be under 60 seconds
        assert std_dev < 20.0   # Standard deviation should be under 20 seconds

    def test_concurrent_processing(self, sample_videos):
        """Test system performance under concurrent load."""
        max_workers = 5
        results = []
        
        def process_video(video):
            start_time = time.time()
            generator = ClipGenerator(video["path"])
            output_path, duration = generator.process_clip(
                target_duration=30.0
            )
            return time.time() - start_time
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_video, video)
                for video in sample_videos[:max_workers]
            ]
            results = [future.result() for future in futures]
        
        # Calculate concurrent processing metrics
        avg_time = statistics.mean(results)
        max_time = max(results)
        
        # Assert concurrent processing requirements
        assert avg_time < 90.0  # Average time under concurrent load
        assert max_time < 120.0  # Maximum processing time

    def test_memory_usage(self, sample_videos):
        """Test memory usage during video processing."""
        process = psutil.Process()
        memory_usage = []
        
        for video in sample_videos:
            initial_memory = process.memory_info().rss
            generator = ClipGenerator(video["path"])
            output_path, duration = generator.process_clip(
                target_duration=30.0
            )
            final_memory = process.memory_info().rss
            memory_usage.append(final_memory - initial_memory)
        
        # Calculate memory metrics
        avg_memory = statistics.mean(memory_usage)
        max_memory = max(memory_usage)
        
        # Assert memory usage requirements
        assert avg_memory < 1024 * 1024 * 1024  # Average memory usage under 1GB
        assert max_memory < 2 * 1024 * 1024 * 1024  # Maximum memory usage under 2GB

    def test_cpu_utilization(self, sample_videos):
        """Test CPU utilization during video processing."""
        cpu_percentages = []
        
        for video in sample_videos:
            # Monitor CPU usage during processing
            cpu_percent = psutil.cpu_percent(interval=1)
            generator = ClipGenerator(video["path"])
            output_path, duration = generator.process_clip(
                target_duration=30.0
            )
            cpu_percentages.append(cpu_percent)
        
        # Calculate CPU metrics
        avg_cpu = statistics.mean(cpu_percentages)
        max_cpu = max(cpu_percentages)
        
        # Assert CPU utilization requirements
        assert avg_cpu < 80.0  # Average CPU usage under 80%
        assert max_cpu < 95.0  # Maximum CPU usage under 95%

    def test_disk_io_performance(self, sample_videos):
        """Test disk I/O performance during video processing."""
        disk_io = []
        
        for video in sample_videos:
            # Monitor disk I/O during processing
            initial_io = psutil.disk_io_counters()
            generator = ClipGenerator(video["path"])
            output_path, duration = generator.process_clip(
                target_duration=30.0
            )
            final_io = psutil.disk_io_counters()
            
            disk_io.append({
                "read_bytes": final_io.read_bytes - initial_io.read_bytes,
                "write_bytes": final_io.write_bytes - initial_io.write_bytes
            })
        
        # Calculate I/O metrics
        avg_read = statistics.mean([io["read_bytes"] for io in disk_io])
        avg_write = statistics.mean([io["write_bytes"] for io in disk_io])
        
        # Assert I/O performance requirements
        assert avg_read < 1024 * 1024 * 100  # Average read speed under 100MB/s
        assert avg_write < 1024 * 1024 * 50  # Average write speed under 50MB/s

    def test_api_response_time(self, client, test_user):
        """Test API endpoint response times."""
        endpoints = [
            ("GET", "/jobs/"),
            ("GET", "/token/balance"),
            ("GET", "/subscription/plans"),
            ("POST", "/jobs/"),
            ("GET", "/analytics/user/me")
        ]
        
        response_times = []
        for method, endpoint in endpoints:
            start_time = time.time()
            if method == "GET":
                response = client.get(endpoint, headers={"Authorization": f"Bearer {token}"})
            else:
                response = client.post(endpoint, headers={"Authorization": f"Bearer {token}"})
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            assert response.status_code in [200, 201]
        
        # Calculate API performance metrics
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        # Assert API performance requirements
        assert avg_response_time < 0.5  # Average response time under 500ms
        assert max_response_time < 1.0  # Maximum response time under 1s

    def test_quality_metrics_performance(self, sample_videos):
        """Test performance of quality analysis operations."""
        analyzer = QualityAnalyzer()
        analysis_times = []
        
        for video in sample_videos:
            start_time = time.time()
            # Perform quality analysis
            video_metrics = analyzer.analyze_video_quality(video["path"])
            audio_metrics = analyzer.analyze_audio_quality(video["path"])
            overall_score = analyzer.calculate_overall_score(video_metrics, audio_metrics)
            analysis_time = time.time() - start_time
            
            analysis_times.append(analysis_time)
            
            # Verify quality metrics
            assert 0 <= overall_score <= 1
            assert all(0 <= value <= 1 for value in video_metrics.values())
            assert all(0 <= value <= 1 for value in audio_metrics.values())
        
        # Calculate quality analysis performance metrics
        avg_analysis_time = statistics.mean(analysis_times)
        max_analysis_time = max(analysis_times)
        
        # Assert quality analysis performance requirements
        assert avg_analysis_time < 10.0  # Average analysis time under 10 seconds
        assert max_analysis_time < 15.0  # Maximum analysis time under 15 seconds 