import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime
import json

from src.models.job import JobCreate, JobStatus
from src.models.user import UserCreate
from src.services.job_service import JobService
from src.services.token_service import TokenService
from src.services.storage_service import StorageService

class TestJobWorkflow:
    def test_create_and_process_job(self, client, test_user, test_video_file):
        """Test complete job creation and processing workflow."""
        # Create user
        user_response = client.post(
            "/users/",
            json={
                "email": test_user["email"],
                "password": test_user["password"],
                "username": test_user["username"]
            }
        )
        assert user_response.status_code == 201
        user_id = user_response.json()["id"]

        # Login
        login_response = client.post(
            "/auth/login",
            data={
                "username": test_user["email"],
                "password": test_user["password"]
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Create job
        with open(test_video_file, "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            job_response = client.post(
                "/jobs/",
                files=files,
                data={
                    "target_duration": "30.0",
                    "target_width": "1080",
                    "target_height": "1920",
                    "target_lufs": "-14.0"
                },
                headers={"Authorization": f"Bearer {token}"}
            )
        assert job_response.status_code == 201
        job_id = job_response.json()["id"]

        # Check job status
        status_response = client.get(
            f"/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert status_response.status_code == 200
        assert status_response.json()["status"] in ["pending", "processing"]

        # Wait for job completion (with timeout)
        max_attempts = 10
        for _ in range(max_attempts):
            status_response = client.get(
                f"/jobs/{job_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
            if status_response.json()["status"] == "completed":
                break
            elif status_response.json()["status"] == "failed":
                pytest.fail("Job processing failed")
            time.sleep(2)

        # Verify job completion
        assert status_response.json()["status"] == "completed"
        assert "output_url" in status_response.json()

    def test_job_cancellation(self, client, test_user, test_video_file):
        """Test job cancellation workflow."""
        # Create and start a job
        # ... (similar setup as above)

        # Cancel job
        cancel_response = client.delete(
            f"/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert cancel_response.status_code == 200

        # Verify job cancellation
        status_response = client.get(
            f"/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert status_response.json()["status"] == "cancelled"

class TestTokenIntegration:
    def test_token_consumption(self, client, test_user):
        """Test token consumption during job processing."""
        # Create user and get token
        # ... (similar setup as above)

        # Check initial token balance
        balance_response = client.get(
            "/token/balance",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert balance_response.status_code == 200
        initial_balance = balance_response.json()["balance"]

        # Create and process a job
        # ... (job creation code)

        # Check final token balance
        balance_response = client.get(
            "/token/balance",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert balance_response.status_code == 200
        final_balance = balance_response.json()["balance"]
        assert final_balance < initial_balance

        # Check token usage history
        history_response = client.get(
            "/token/usage",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert history_response.status_code == 200
        assert len(history_response.json()["transactions"]) > 0

class TestStorageIntegration:
    def test_file_upload_and_download(self, client, test_user, test_video_file):
        """Test file storage operations."""
        # Create user and get token
        # ... (similar setup as above)

        # Upload file
        with open(test_video_file, "rb") as f:
            upload_response = client.post(
                "/storage/upload",
                files={"file": ("test_video.mp4", f, "video/mp4")},
                headers={"Authorization": f"Bearer {token}"}
            )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

        # Download file
        download_response = client.get(
            f"/storage/download/{file_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert download_response.status_code == 200

        # Verify file content
        assert len(download_response.content) > 0

class TestWebSocketIntegration:
    def test_job_status_updates(self, client, test_user):
        """Test real-time job status updates via WebSocket."""
        # Create user and get token
        # ... (similar setup as above)

        # Connect to WebSocket
        with client.websocket_connect(
            f"/ws/{user_id}",
            headers={"Authorization": f"Bearer {token}"}
        ) as websocket:
            # Create a job
            job_response = client.post(
                "/jobs/",
                json={
                    "input_video": "test_video.mp4",
                    "target_duration": 30.0
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            job_id = job_response.json()["id"]

            # Receive status updates
            received_updates = []
            for _ in range(5):  # Receive up to 5 updates
                data = websocket.receive_json()
                received_updates.append(data)
                if data["status"] in ["completed", "failed"]:
                    break

            # Verify updates
            assert len(received_updates) > 0
            assert any(update["job_id"] == job_id for update in received_updates)
            assert received_updates[-1]["status"] in ["completed", "failed"] 