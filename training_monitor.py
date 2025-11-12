#!/usr/bin/env python3
"""
AI Training Monitor - Backend Service

Manages real AI model training with live status updates.
Provides API endpoints for admin dashboard to monitor training progress.

Author: SignForge Team
Date: 2025-01-11
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrainingPhase(str, Enum):
    """Training phases"""
    IDLE = "idle"
    PROCESSING_VIDEOS = "processing_videos"
    TRAINING_TEXT_TO_POSE = "training_text_to_pose"
    TRAINING_POSE_TO_VIDEO = "training_pose_to_video"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingStatus:
    """Singleton class to track training status"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.phase = TrainingPhase.IDLE
        self.progress = 0.0
        self.current_step = ""
        self.metrics = {
            "video_processing": {
                "total_videos": 0,
                "processed_videos": 0,
                "failed_videos": 0,
                "storage_saved_mb": 0
            },
            "text_to_pose": {
                "epoch": 0,
                "total_epochs": 0,
                "loss": 0.0,
                "best_loss": float('inf'),
                "samples_processed": 0
            },
            "pose_to_video": {
                "epoch": 0,
                "total_epochs": 0,
                "loss": 0.0,
                "best_loss": float('inf'),
                "samples_processed": 0
            }
        }
        self.start_time = None
        self.estimated_completion = None
        self.error_message = None
        self.sample_outputs = []  # List of sample generation outputs
        self.logs = []  # Recent log messages

    def start_training(self, total_videos: int, text_to_pose_epochs: int, pose_to_video_epochs: int):
        """Initialize training session"""
        self.phase = TrainingPhase.PROCESSING_VIDEOS
        self.progress = 0.0
        self.current_step = "Starting video processing..."
        self.start_time = datetime.now()
        self.error_message = None

        self.metrics["video_processing"]["total_videos"] = total_videos
        self.metrics["text_to_pose"]["total_epochs"] = text_to_pose_epochs
        self.metrics["pose_to_video"]["total_epochs"] = pose_to_video_epochs

        self._add_log("Training started", "info")

    def update_video_processing(self, processed: int, failed: int, storage_saved_mb: float):
        """Update video processing metrics"""
        self.metrics["video_processing"]["processed_videos"] = processed
        self.metrics["video_processing"]["failed_videos"] = failed
        self.metrics["video_processing"]["storage_saved_mb"] = storage_saved_mb

        total = self.metrics["video_processing"]["total_videos"]
        if total > 0:
            self.progress = (processed / total) * 100

        self.current_step = f"Processing videos: {processed}/{total}"

    def start_text_to_pose_training(self):
        """Start text-to-pose training phase"""
        self.phase = TrainingPhase.TRAINING_TEXT_TO_POSE
        self.progress = 0.0
        self.current_step = "Starting Text-to-Pose model training..."
        self._add_log("Text-to-Pose training started", "info")

    def update_text_to_pose(self, epoch: int, loss: float, samples: int):
        """Update text-to-pose training metrics"""
        self.metrics["text_to_pose"]["epoch"] = epoch
        self.metrics["text_to_pose"]["loss"] = loss
        self.metrics["text_to_pose"]["samples_processed"] = samples

        if loss < self.metrics["text_to_pose"]["best_loss"]:
            self.metrics["text_to_pose"]["best_loss"] = loss
            self._add_log(f"New best Text-to-Pose loss: {loss:.6f}", "success")

        total_epochs = self.metrics["text_to_pose"]["total_epochs"]
        if total_epochs > 0:
            self.progress = (epoch / total_epochs) * 100

        self.current_step = f"Training Text-to-Pose: Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}"

    def start_pose_to_video_training(self):
        """Start pose-to-video training phase"""
        self.phase = TrainingPhase.TRAINING_POSE_TO_VIDEO
        self.progress = 0.0
        self.current_step = "Starting Pose-to-Video model training..."
        self._add_log("Pose-to-Video training started", "info")

    def update_pose_to_video(self, epoch: int, loss: float, samples: int):
        """Update pose-to-video training metrics"""
        self.metrics["pose_to_video"]["epoch"] = epoch
        self.metrics["pose_to_video"]["loss"] = loss
        self.metrics["pose_to_video"]["samples_processed"] = samples

        if loss < self.metrics["pose_to_video"]["best_loss"]:
            self.metrics["pose_to_video"]["best_loss"] = loss
            self._add_log(f"New best Pose-to-Video loss: {loss:.6f}", "success")

        total_epochs = self.metrics["pose_to_video"]["total_epochs"]
        if total_epochs > 0:
            self.progress = (epoch / total_epochs) * 100

        self.current_step = f"Training Pose-to-Video: Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}"

    def add_sample_output(self, word: str, video_path: str, quality_score: float):
        """Add a sample generation output"""
        sample = {
            "word": word,
            "video_path": video_path,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat()
        }
        self.sample_outputs.append(sample)

        # Keep only last 10 samples
        if len(self.sample_outputs) > 10:
            self.sample_outputs = self.sample_outputs[-10:]

    def complete_training(self):
        """Mark training as completed"""
        self.phase = TrainingPhase.COMPLETED
        self.progress = 100.0
        self.current_step = "Training completed successfully!"
        self._add_log("Training completed", "success")

    def fail_training(self, error: str):
        """Mark training as failed"""
        self.phase = TrainingPhase.FAILED
        self.error_message = error
        self.current_step = f"Training failed: {error}"
        self._add_log(f"Training failed: {error}", "error")

    def _add_log(self, message: str, level: str = "info"):
        """Add log entry"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level
        }
        self.logs.append(log_entry)

        # Keep only last 100 logs
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

        logger.info(f"[{level.upper()}] {message}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status as dictionary"""
        status = {
            "phase": self.phase.value,
            "progress": self.progress,
            "current_step": self.current_step,
            "metrics": self.metrics,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "error_message": self.error_message,
            "sample_outputs": self.sample_outputs,
            "recent_logs": self.logs[-20:]  # Last 20 logs
        }

        # Calculate elapsed time and estimate
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            status["elapsed_seconds"] = elapsed

            if self.progress > 0 and self.phase != TrainingPhase.COMPLETED:
                total_estimated = (elapsed / self.progress) * 100
                remaining = total_estimated - elapsed
                status["estimated_remaining_seconds"] = remaining
            else:
                status["estimated_remaining_seconds"] = 0
        else:
            # No training started yet
            status["elapsed_seconds"] = 0
            status["estimated_remaining_seconds"] = 0

        return status

    def reset(self):
        """Reset training status"""
        self.phase = TrainingPhase.IDLE
        self.progress = 0.0
        self.current_step = ""
        self.metrics = {
            "video_processing": {
                "total_videos": 0,
                "processed_videos": 0,
                "failed_videos": 0,
                "storage_saved_mb": 0
            },
            "text_to_pose": {
                "epoch": 0,
                "total_epochs": 0,
                "loss": 0.0,
                "best_loss": float('inf'),
                "samples_processed": 0
            },
            "pose_to_video": {
                "epoch": 0,
                "total_epochs": 0,
                "loss": 0.0,
                "best_loss": float('inf'),
                "samples_processed": 0
            }
        }
        self.start_time = None
        self.error_message = None
        self.sample_outputs = []
        self.logs = []


# Global training status instance
training_status = TrainingStatus()
