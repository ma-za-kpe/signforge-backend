"""
Tests for Training Monitor

Tests the training status tracking singleton class.

Author: SignForge Team
Date: 2025-01-11
"""

import pytest
from datetime import datetime
from training_monitor import TrainingStatus, TrainingPhase


class TestTrainingStatus:
    """Test TrainingStatus singleton"""

    def setup_method(self):
        """Reset training status before each test"""
        TrainingStatus._instance = None

    def test_singleton_pattern(self):
        """Test that TrainingStatus is a singleton"""
        status1 = TrainingStatus()
        status2 = TrainingStatus()

        assert status1 is status2
        assert id(status1) == id(status2)

    def test_initial_state(self):
        """Test initial state after creation"""
        status = TrainingStatus()

        assert status.phase == TrainingPhase.IDLE
        assert status.progress == 0.0
        assert status.current_step == ""
        assert status.start_time is None
        assert status.error_message is None
        assert len(status.logs) == 0
        assert len(status.sample_outputs) == 0

    def test_start_training(self):
        """Test starting training session"""
        status = TrainingStatus()
        status.start_training(
            total_videos=100,
            text_to_pose_epochs=50,
            pose_to_video_epochs=25
        )

        assert status.phase == TrainingPhase.PROCESSING_VIDEOS
        assert status.progress == 0.0
        assert status.start_time is not None
        assert status.metrics["video_processing"]["total_videos"] == 100
        assert status.metrics["text_to_pose"]["total_epochs"] == 50
        assert status.metrics["pose_to_video"]["total_epochs"] == 25
        assert len(status.logs) == 1  # "Training started" log

    def test_update_video_processing(self):
        """Test updating video processing metrics"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        status.update_video_processing(50, 2, 1234.5)

        assert status.metrics["video_processing"]["processed_videos"] == 50
        assert status.metrics["video_processing"]["failed_videos"] == 2
        assert status.metrics["video_processing"]["storage_saved_mb"] == 1234.5
        assert status.progress == 50.0  # 50/100 = 50%
        assert "50/100" in status.current_step

    def test_start_text_to_pose_training(self):
        """Test starting text-to-pose phase"""
        status = TrainingStatus()
        status.start_text_to_pose_training()

        assert status.phase == TrainingPhase.TRAINING_TEXT_TO_POSE
        assert status.progress == 0.0
        assert "Text-to-Pose" in status.current_step

    def test_update_text_to_pose(self):
        """Test updating text-to-pose metrics"""
        status = TrainingStatus()
        status.metrics["text_to_pose"]["total_epochs"] = 100
        status.start_text_to_pose_training()

        status.update_text_to_pose(epoch=10, loss=0.005, samples=1000)

        assert status.metrics["text_to_pose"]["epoch"] == 10
        assert status.metrics["text_to_pose"]["loss"] == 0.005
        assert status.metrics["text_to_pose"]["samples_processed"] == 1000
        assert status.progress == 10.0  # 10/100 = 10%

    def test_best_loss_tracking(self):
        """Test that best loss is tracked correctly"""
        status = TrainingStatus()
        status.metrics["text_to_pose"]["total_epochs"] = 100
        status.start_text_to_pose_training()

        # First update - becomes best
        status.update_text_to_pose(1, 0.010, 100)
        assert status.metrics["text_to_pose"]["best_loss"] == 0.010

        # Second update - worse loss, best stays same
        status.update_text_to_pose(2, 0.015, 200)
        assert status.metrics["text_to_pose"]["best_loss"] == 0.010

        # Third update - better loss, best updates
        status.update_text_to_pose(3, 0.005, 300)
        assert status.metrics["text_to_pose"]["best_loss"] == 0.005

    def test_add_sample_output(self):
        """Test adding sample outputs"""
        status = TrainingStatus()

        status.add_sample_output("HELLO", "/path/to/video.mp4", 0.95)

        assert len(status.sample_outputs) == 1
        assert status.sample_outputs[0]["word"] == "HELLO"
        assert status.sample_outputs[0]["video_path"] == "/path/to/video.mp4"
        assert status.sample_outputs[0]["quality_score"] == 0.95

    def test_sample_outputs_limit(self):
        """Test that sample outputs are limited to 10"""
        status = TrainingStatus()

        # Add 15 samples
        for i in range(15):
            status.add_sample_output(f"WORD{i}", f"/path/{i}.mp4", 0.9)

        # Should keep only last 10
        assert len(status.sample_outputs) == 10
        assert status.sample_outputs[0]["word"] == "WORD5"
        assert status.sample_outputs[-1]["word"] == "WORD14"

    def test_complete_training(self):
        """Test completing training"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        status.complete_training()

        assert status.phase == TrainingPhase.COMPLETED
        assert status.progress == 100.0
        assert "complete" in status.current_step.lower()

    def test_fail_training(self):
        """Test failing training"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        error_msg = "CUDA out of memory"
        status.fail_training(error_msg)

        assert status.phase == TrainingPhase.FAILED
        assert status.error_message == error_msg
        assert error_msg in status.current_step

    def test_get_status_dict(self):
        """Test getting status as dictionary"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)
        status.update_video_processing(50, 2, 1234.5)

        status_dict = status.get_status()

        assert isinstance(status_dict, dict)
        assert status_dict["phase"] == "processing_videos"
        assert status_dict["progress"] == 50.0
        assert "elapsed_seconds" in status_dict
        assert "metrics" in status_dict
        assert status_dict["metrics"]["video_processing"]["processed_videos"] == 50

    def test_logs_limit(self):
        """Test that logs are limited to 100"""
        status = TrainingStatus()

        # Add 150 log entries
        for i in range(150):
            status._add_log(f"Log message {i}")

        # Should keep only last 100
        assert len(status.logs) == 100
        assert status.logs[0]["message"] == "Log message 50"
        assert status.logs[-1]["message"] == "Log message 149"

    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        # Simulate some time passing
        import time
        time.sleep(0.1)

        status_dict = status.get_status()
        assert status_dict["elapsed_seconds"] > 0

    def test_estimated_remaining_time(self):
        """Test estimated remaining time calculation"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)

        import time
        time.sleep(0.1)

        # Set progress to 50%
        status.update_video_processing(50, 0, 100)

        status_dict = status.get_status()
        assert "estimated_remaining_seconds" in status_dict
        assert status_dict["estimated_remaining_seconds"] > 0

    def test_reset(self):
        """Test resetting training status"""
        status = TrainingStatus()
        status.start_training(100, 50, 25)
        status.update_video_processing(50, 2, 1234.5)
        status.add_sample_output("TEST", "/path", 0.9)

        status.reset()

        assert status.phase == TrainingPhase.IDLE
        assert status.progress == 0.0
        assert status.start_time is None
        assert len(status.logs) == 0
        assert len(status.sample_outputs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
