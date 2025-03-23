import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch
from ..models.database import Job, JobStatus, JobType
from ..models.config import AppConfig
from ..services.job_queue import JobQueueManager
from ..services.job_processor import JobProcessor

@pytest.fixture
def config():
    """Create test configuration."""
    return AppConfig(
        processing=AppConfig.ProcessingConfig(
            max_concurrent_jobs=2
        )
    )

@pytest.fixture
def job_repository():
    """Create mock job repository."""
    return Mock()

@pytest.fixture
def job_log_repository():
    """Create mock job log repository."""
    return Mock()

@pytest.fixture
def job_processor():
    """Create mock job processor."""
    return Mock(spec=JobProcessor)

@pytest.fixture
def job_queue(config, job_repository, job_log_repository, job_processor):
    """Create job queue manager with mocked dependencies."""
    with patch('..services.job_queue.JobProcessor', return_value=job_processor):
        queue = JobQueueManager(
            job_repository=job_repository,
            job_log_repository=job_log_repository,
            config=config,
            max_concurrent_jobs=2
        )
        return queue

@pytest.fixture
def test_job():
    """Create a test job."""
    return Job(
        id="1",
        job_id="test_job_1",
        user_id="test_user_1",
        job_type=JobType.VIDEO_EDIT,
        status=JobStatus.PENDING,
        progress=0.0,
        input_data={
            "input_path": "test_video.mp4",
            "target_duration": 30.0,
            "target_width": 1080,
            "target_height": 1920,
            "target_lufs": -14.0
        },
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

@pytest.mark.asyncio
async def test_start_stop(job_queue):
    """Test job queue start and stop."""
    # Start queue
    await job_queue.start()
    assert job_queue._running is True
    assert job_queue._task is not None

    # Stop queue
    await job_queue.stop()
    assert job_queue._running is False
    assert job_queue._task is None
    assert len(job_queue.active_jobs) == 0

@pytest.mark.asyncio
async def test_submit_job(job_queue, job_processor, job_log_repository):
    """Test job submission."""
    # Arrange
    job_data = {
        "job_id": "test_job_1",
        "user_id": "test_user_1",
        "job_type": JobType.VIDEO_EDIT,
        "input_data": {"input_path": "test_video.mp4"}
    }
    job_processor.create_job.return_value = Job(**job_data)

    # Act
    job = await job_queue.submit_job(job_data)

    # Assert
    assert job.job_id == "test_job_1"
    job_processor.create_job.assert_called_once_with(job_data)
    job_log_repository.create.assert_called_once()

@pytest.mark.asyncio
async def test_get_job_status(job_queue, job_repository):
    """Test getting job status."""
    # Arrange
    job_repository.get_by_id.return_value = Job(
        id="1",
        job_id="test_job_1",
        user_id="test_user_1",
        job_type=JobType.VIDEO_EDIT,
        status=JobStatus.PROCESSING,
        progress=50.0,
        input_data={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    # Act
    job = await job_queue.get_job_status("test_job_1")

    # Assert
    assert job.job_id == "test_job_1"
    assert job.status == JobStatus.PROCESSING
    assert job.progress == 50.0
    job_repository.get_by_id.assert_called_once_with("test_job_1")

@pytest.mark.asyncio
async def test_cancel_job(job_queue, job_processor):
    """Test job cancellation."""
    # Arrange
    job_id = "test_job_1"
    job_processor.cancel_job.return_value = Job(
        id="1",
        job_id=job_id,
        user_id="test_user_1",
        job_type=JobType.VIDEO_EDIT,
        status=JobStatus.CANCELLED,
        progress=0.0,
        input_data={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    # Act
    success = await job_queue.cancel_job(job_id)

    # Assert
    assert success is True
    job_processor.cancel_job.assert_called_once_with(job_id)

@pytest.mark.asyncio
async def test_process_queue(job_queue, job_repository, job_processor):
    """Test job queue processing."""
    # Arrange
    job_queue._running = True
    job_repository.get_pending_jobs.return_value = [
        Job(
            id="1",
            job_id="test_job_1",
            user_id="test_user_1",
            job_type=JobType.VIDEO_EDIT,
            status=JobStatus.PENDING,
            progress=0.0,
            input_data={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    ]

    # Act
    task = asyncio.create_task(job_queue._process_queue())
    await asyncio.sleep(0.1)  # Allow task to start
    job_queue._running = False
    await task

    # Assert
    assert "test_job_1" in job_queue.active_jobs
    job_repository.get_pending_jobs.assert_called_once()
    job_processor.process_job.assert_called_once_with("test_job_1")

@pytest.mark.asyncio
async def test_process_queue_error(job_queue, job_repository):
    """Test job queue error handling."""
    # Arrange
    job_queue._running = True
    job_repository.get_pending_jobs.side_effect = Exception("Database error")

    # Act
    task = asyncio.create_task(job_queue._process_queue())
    await asyncio.sleep(0.1)  # Allow task to start
    job_queue._running = False
    await task

    # Assert
    assert len(job_queue.active_jobs) == 0
    job_repository.get_pending_jobs.assert_called_once() 