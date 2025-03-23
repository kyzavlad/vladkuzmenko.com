import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator, Dict
import os
import tempfile
import shutil

from src.main import app
from src.database.models import Base
from src.config import settings

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session")
def test_db_engine():
    """Create a test database engine."""
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_db(test_db_engine):
    """Create a fresh database session for each test."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture(scope="function")
def client(test_db) -> Generator:
    """Create a test client with a test database session."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def test_user() -> Dict[str, str]:
    """Create a test user."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "username": "testuser"
    }

@pytest.fixture(scope="function")
def test_video_file():
    """Create a temporary test video file."""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_video.mp4")
    
    # Create a dummy video file
    with open(video_path, "wb") as f:
        f.write(b"dummy video content")
    
    yield video_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def test_job_data(test_user):
    """Create test job data."""
    return {
        "user_id": test_user["id"],
        "input_video": "test_video.mp4",
        "target_duration": 30.0,
        "target_width": 1080,
        "target_height": 1920,
        "target_lufs": -14.0,
        "status": "pending"
    }

@pytest.fixture(scope="function")
def mock_stripe():
    """Mock Stripe API calls."""
    # TODO: Implement Stripe mocking
    pass

@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis cache."""
    # TODO: Implement Redis mocking
    pass

@pytest.fixture(scope="function")
def mock_s3():
    """Mock S3 storage."""
    # TODO: Implement S3 mocking
    pass

@pytest.fixture(scope="function")
def mock_ffmpeg():
    """Mock FFmpeg operations."""
    # TODO: Implement FFmpeg mocking
    pass 