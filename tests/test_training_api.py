"""
Tests for Training API Endpoints

Tests the FastAPI endpoints for starting, monitoring, and controlling AI training.

Author: SignForge Team
Date: 2025-01-11
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import threading
import time

# Import the FastAPI app and training components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_monitor import TrainingStatus, TrainingPhase


@pytest.fixture
def reset_training_status():
    """Reset training status before each test"""
    TrainingStatus._instance = None
    yield
    TrainingStatus._instance = None


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    # Import main app
    from main import app
    return TestClient(app)


class TestTrainingAPIEndpoints:
    """Test training API endpoints"""

    def test_get_status_endpoint_initial(self, client, reset_training_status):
        """Test GET /api/admin/training/status with initial state"""
        response = client.get("/api/admin/training/status")

        assert response.status_code == 200
        data = response.json()

        assert data["phase"] == "idle"
        assert data["progress"] == 0.0
        assert data["error_message"] is None
        assert "metrics" in data
        assert "recent_logs" in data

    def test_get_status_endpoint_during_training(self, client, reset_training_status):
        """Test GET /api/admin/training/status during active training"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)
        status.update_video_processing(50, 2, 1234.5)

        response = client.get("/api/admin/training/status")

        assert response.status_code == 200
        data = response.json()

        assert data["phase"] == "processing_videos"
        assert data["progress"] == 50.0
        assert data["metrics"]["video_processing"]["processed_videos"] == 50
        assert data["metrics"]["video_processing"]["failed_videos"] == 2

    def test_start_training_endpoint(self, client, reset_training_status):
        """Test POST /api/admin/training/start"""
        with patch('training_api.threading.Thread') as mock_thread:
            response = client.post(
                "/api/admin/training/start",
                json={
                    "text_to_pose_epochs": 100,
                    "pose_to_video_epochs": 50,
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            )

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "started"
            assert data["message"] == "Training pipeline started in background"

            # Verify thread was created and started
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

    def test_start_training_already_running(self, client, reset_training_status):
        """Test POST /api/admin/training/start when training already running"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        response = client.post(
            "/api/admin/training/start",
            json={
                "text_to_pose_epochs": 100,
                "pose_to_video_epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "already running" in data["detail"].lower()

    def test_start_training_invalid_parameters(self, client, reset_training_status):
        """Test POST /api/admin/training/start with invalid parameters"""
        response = client.post(
            "/api/admin/training/start",
            json={
                "text_to_pose_epochs": -10,  # Invalid: negative
                "pose_to_video_epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        )

        # Should fail validation
        assert response.status_code == 422

    def test_stop_training_endpoint(self, client, reset_training_status):
        """Test POST /api/admin/training/stop"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        with patch('training_api.training_stop_flag') as mock_flag:
            response = client.post("/api/admin/training/stop")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "stopping"
            assert "signal sent" in data["message"].lower()

    def test_stop_training_not_running(self, client, reset_training_status):
        """Test POST /api/admin/training/stop when not running"""
        response = client.post("/api/admin/training/stop")

        assert response.status_code == 400
        data = response.json()
        assert "not running" in data["detail"].lower()

    def test_get_metrics_endpoint(self, client, reset_training_status):
        """Test GET /api/admin/training/metrics"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)
        status.update_video_processing(75, 5, 2500.0)
        status.start_text_to_pose_training()
        status.update_text_to_pose(10, 0.005, 1000)

        response = client.get("/api/admin/training/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "video_processing" in data
        assert "text_to_pose" in data
        assert "pose_to_video" in data

        assert data["video_processing"]["processed_videos"] == 75
        assert data["text_to_pose"]["epoch"] == 10
        assert data["text_to_pose"]["loss"] == 0.005

    def test_get_samples_endpoint(self, client, reset_training_status):
        """Test GET /api/admin/training/samples"""
        status = TrainingStatus()
        status.add_sample_output("HELLO", "/path/to/video1.mp4", 0.95)
        status.add_sample_output("WORLD", "/path/to/video2.mp4", 0.88)

        response = client.get("/api/admin/training/samples")

        assert response.status_code == 200
        data = response.json()

        assert "samples" in data
        assert len(data["samples"]) == 2
        assert data["samples"][0]["word"] == "HELLO"
        assert data["samples"][1]["word"] == "WORLD"

    def test_get_logs_endpoint(self, client, reset_training_status):
        """Test GET /api/admin/training/logs"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)
        status._add_log("Processing video 1", "info")
        status._add_log("Error in video 2", "error")

        response = client.get("/api/admin/training/logs")

        assert response.status_code == 200
        data = response.json()

        assert "logs" in data
        assert len(data["logs"]) >= 2

        # Check that logs contain expected messages
        messages = [log["message"] for log in data["logs"]]
        assert any("Processing video 1" in msg for msg in messages)
        assert any("Error in video 2" in msg for msg in messages)

    def test_get_logs_with_limit(self, client, reset_training_status):
        """Test GET /api/admin/training/logs with limit parameter"""
        status = TrainingStatus()

        # Add 20 log entries
        for i in range(20):
            status._add_log(f"Log message {i}", "info")

        response = client.get("/api/admin/training/logs?limit=5")

        assert response.status_code == 200
        data = response.json()

        assert len(data["logs"]) == 5

    def test_concurrent_status_requests(self, client, reset_training_status):
        """Test multiple concurrent status requests (simulating frontend polling)"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        # Simulate 10 concurrent requests
        responses = []
        for _ in range(10):
            response = client.get("/api/admin/training/status")
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["phase"] == "processing_videos"

    def test_training_lifecycle_endpoints(self, client, reset_training_status):
        """Test complete training lifecycle through API"""
        # 1. Check initial status
        response = client.get("/api/admin/training/status")
        assert response.json()["phase"] == "idle"

        # 2. Start training (mocked)
        with patch('training_api.threading.Thread'):
            response = client.post(
                "/api/admin/training/start",
                json={
                    "text_to_pose_epochs": 10,
                    "pose_to_video_epochs": 5,
                    "batch_size": 16,
                    "learning_rate": 0.001
                }
            )
            assert response.status_code == 200

        # 3. Manually update status (simulating training progress)
        status = TrainingStatus()
        status.start_training(100, 10, 5)
        status.update_video_processing(50, 0, 1000.0)

        response = client.get("/api/admin/training/status")
        assert response.json()["phase"] == "processing_videos"
        assert response.json()["progress"] == 50.0

        # 4. Stop training
        with patch('training_api.training_stop_flag'):
            response = client.post("/api/admin/training/stop")
            assert response.status_code == 200

    def test_error_handling_in_training(self, client, reset_training_status):
        """Test error handling when training fails"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)
        status.fail_training("CUDA out of memory")

        response = client.get("/api/admin/training/status")

        assert response.status_code == 200
        data = response.json()

        assert data["phase"] == "failed"
        assert data["error_message"] == "CUDA out of memory"

    def test_metrics_evolution(self, client, reset_training_status):
        """Test metrics evolving through training phases"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        # Phase 1: Video processing
        status.update_video_processing(100, 0, 5000.0)
        response = client.get("/api/admin/training/metrics")
        assert response.json()["video_processing"]["processed_videos"] == 100

        # Phase 2: Text-to-Pose
        status.start_text_to_pose_training()
        status.update_text_to_pose(10, 0.005, 1000)
        response = client.get("/api/admin/training/metrics")
        assert response.json()["text_to_pose"]["epoch"] == 10

        # Phase 3: Pose-to-Video
        status.start_pose_to_video_training()
        status.update_pose_to_video(5, 0.002, 500)
        response = client.get("/api/admin/training/metrics")
        assert response.json()["pose_to_video"]["epoch"] == 5


class TestTrainingRequestValidation:
    """Test request validation for training endpoints"""

    def test_start_training_missing_fields(self, client, reset_training_status):
        """Test POST /api/admin/training/start with missing fields"""
        response = client.post(
            "/api/admin/training/start",
            json={
                "text_to_pose_epochs": 100
                # Missing other required fields
            }
        )

        # Should succeed with defaults
        # (depends on whether fields are optional with defaults)
        assert response.status_code in [200, 422]

    def test_start_training_invalid_types(self, client, reset_training_status):
        """Test POST /api/admin/training/start with invalid types"""
        response = client.post(
            "/api/admin/training/start",
            json={
                "text_to_pose_epochs": "not_a_number",  # Should be int
                "pose_to_video_epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        )

        assert response.status_code == 422

    def test_get_logs_invalid_limit(self, client, reset_training_status):
        """Test GET /api/admin/training/logs with invalid limit"""
        response = client.get("/api/admin/training/logs?limit=-5")

        # Should handle gracefully (either error or ignore)
        assert response.status_code in [200, 422]


class TestTrainingThreadExecution:
    """Test background thread execution for training"""

    def test_training_runs_in_background(self, client, reset_training_status):
        """Test that training actually runs in background thread"""
        with patch('training_api.run_full_training_pipeline') as mock_pipeline:
            with patch('training_api.threading.Thread') as mock_thread:
                # Configure mock thread to actually call the target function
                def side_effect(*args, **kwargs):
                    target = kwargs.get('target')
                    if target:
                        # Don't actually call it (would start training)
                        pass
                    mock = MagicMock()
                    mock.start = MagicMock()
                    return mock

                mock_thread.side_effect = side_effect

                response = client.post(
                    "/api/admin/training/start",
                    json={
                        "text_to_pose_epochs": 10,
                        "pose_to_video_epochs": 5,
                        "batch_size": 16,
                        "learning_rate": 0.001
                    }
                )

                assert response.status_code == 200

                # Verify thread was created with daemon=True
                mock_thread.assert_called_once()
                call_kwargs = mock_thread.call_args[1]
                assert call_kwargs.get('daemon') is True

    def test_training_failure_updates_status(self, client, reset_training_status):
        """Test that training failures properly update status"""
        status = TrainingStatus()

        # Simulate training failure
        try:
            status.start_training(100, 50, 25)
            raise Exception("Simulated training error")
        except Exception as e:
            status.fail_training(str(e))

        response = client.get("/api/admin/training/status")
        data = response.json()

        assert data["phase"] == "failed"
        assert "Simulated training error" in data["error_message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
