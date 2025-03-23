import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import time
import json
from datetime import datetime, timedelta

from src.models.job import JobCreate, JobStatus
from src.models.user import UserCreate
from src.models.token import TokenTransactionType

class TestUserJourney:
    def test_complete_user_journey(self, client, test_user, test_video_file):
        """Test complete user journey from registration to job completion."""
        # 1. User Registration
        register_response = client.post(
            "/users/",
            json={
                "email": test_user["email"],
                "password": test_user["password"],
                "username": test_user["username"]
            }
        )
        assert register_response.status_code == 201
        user_id = register_response.json()["id"]

        # 2. User Login
        login_response = client.post(
            "/auth/login",
            data={
                "username": test_user["email"],
                "password": test_user["password"]
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # 3. Check Initial Token Balance
        balance_response = client.get(
            "/token/balance",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert balance_response.status_code == 200
        initial_balance = balance_response.json()["balance"]

        # 4. Purchase Additional Tokens
        purchase_response = client.post(
            "/token/purchase",
            json={
                "amount": 1000,
                "payment_method": "test_card"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert purchase_response.status_code == 200
        transaction_id = purchase_response.json()["transaction_id"]

        # 5. Verify Token Balance After Purchase
        balance_response = client.get(
            "/token/balance",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert balance_response.status_code == 200
        assert balance_response.json()["balance"] > initial_balance

        # 6. Upload and Process Video
        with open(test_video_file, "rb") as f:
            job_response = client.post(
                "/jobs/",
                files={"file": ("test_video.mp4", f, "video/mp4")},
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

        # 7. Monitor Job Progress via WebSocket
        with client.websocket_connect(
            f"/ws/{user_id}",
            headers={"Authorization": f"Bearer {token}"}
        ) as websocket:
            received_updates = []
            for _ in range(10):  # Receive up to 10 updates
                data = websocket.receive_json()
                received_updates.append(data)
                if data["status"] in ["completed", "failed"]:
                    break
                time.sleep(1)

        # 8. Verify Job Completion
        status_response = client.get(
            f"/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "completed"
        assert "output_url" in status_response.json()

        # 9. Download Processed Video
        download_response = client.get(
            status_response.json()["output_url"],
            headers={"Authorization": f"Bearer {token}"}
        )
        assert download_response.status_code == 200
        assert len(download_response.content) > 0

        # 10. Check Token Usage
        usage_response = client.get(
            "/token/usage",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert usage_response.status_code == 200
        transactions = usage_response.json()["transactions"]
        assert any(
            t["type"] == TokenTransactionType.USAGE and t["job_id"] == job_id
            for t in transactions
        )

class TestSubscriptionJourney:
    def test_subscription_workflow(self, client, test_user):
        """Test subscription management workflow."""
        # 1. User Registration and Login
        # ... (similar setup as above)

        # 2. View Available Plans
        plans_response = client.get(
            "/subscription/plans",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert plans_response.status_code == 200
        plans = plans_response.json()["plans"]
        assert len(plans) > 0

        # 3. Subscribe to a Plan
        plan_id = plans[0]["id"]
        subscribe_response = client.post(
            "/subscription/subscribe",
            json={
                "plan_id": plan_id,
                "payment_method": "test_card"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert subscribe_response.status_code == 200
        subscription_id = subscribe_response.json()["subscription_id"]

        # 4. Verify Subscription Status
        status_response = client.get(
            f"/subscription/{subscription_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "active"

        # 5. Check Subscription Benefits
        benefits_response = client.get(
            "/subscription/benefits",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert benefits_response.status_code == 200
        benefits = benefits_response.json()["benefits"]
        assert len(benefits) > 0

        # 6. Update Subscription
        update_response = client.put(
            f"/subscription/{subscription_id}",
            json={
                "plan_id": plans[1]["id"]  # Upgrade to a different plan
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert update_response.status_code == 200

        # 7. Cancel Subscription
        cancel_response = client.delete(
            f"/subscription/{subscription_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert cancel_response.status_code == 200

        # 8. Verify Cancellation
        status_response = client.get(
            f"/subscription/{subscription_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert status_response.json()["status"] == "cancelled"

class TestErrorHandling:
    def test_error_scenarios(self, client, test_user):
        """Test various error scenarios and recovery."""
        # 1. Invalid Video Format
        with open("invalid.txt", "w") as f:
            f.write("invalid content")
        with open("invalid.txt", "rb") as f:
            response = client.post(
                "/jobs/",
                files={"file": ("invalid.txt", f, "text/plain")},
                headers={"Authorization": f"Bearer {token}"}
            )
        assert response.status_code == 400
        assert "Invalid video format" in response.json()["detail"]

        # 2. Insufficient Tokens
        # Set token balance to 0
        client.post(
            "/token/transfer",
            json={
                "amount": -1000,
                "type": "withdrawal"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Try to create a job
        response = client.post(
            "/jobs/",
            json={"input_video": "test.mp4"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 402
        assert "Insufficient tokens" in response.json()["detail"]

        # 3. Invalid Job Parameters
        response = client.post(
            "/jobs/",
            json={
                "input_video": "test.mp4",
                "target_duration": -1,
                "target_width": 0,
                "target_height": 0
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422
        assert "Invalid parameters" in response.json()["detail"]

        # 4. Network Error Recovery
        # Simulate network error by temporarily disabling connection
        # This would require mocking the network layer
        pass

        # 5. Rate Limiting
        for _ in range(100):  # Make many requests quickly
            client.get("/jobs/", headers={"Authorization": f"Bearer {token}"})
        response = client.get("/jobs/", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"] 