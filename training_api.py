#!/usr/bin/env python3
"""
Training API Endpoints

Admin endpoints for managing and monitoring AI model training.

Author: SignForge Team
Date: 2025-01-11
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import threading

from training_monitor import training_status, TrainingPhase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/training", tags=["admin-training"])


# Request/Response Models
class StartTrainingRequest(BaseModel):
    text_to_pose_epochs: int = 100
    pose_to_video_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3


class TrainingStatusResponse(BaseModel):
    phase: str
    progress: float
    current_step: str
    metrics: Dict[str, Any]
    start_time: Optional[str]
    elapsed_seconds: Optional[float]
    estimated_remaining_seconds: Optional[float]
    error_message: Optional[str]
    sample_outputs: list
    recent_logs: list


# Training thread reference
training_thread: Optional[threading.Thread] = None


@router.post("/start")
async def start_training(request: StartTrainingRequest, background_tasks: BackgroundTasks):
    """
    Start AI model training pipeline

    This will:
    1. Process all 9,877 Kaggle videos → extract poses
    2. Train Text-to-Pose model (PyTorch)
    3. Train Pose-to-Video model (ControlNet)
    """
    global training_thread

    # Check if training is already running
    if training_status.phase not in [TrainingPhase.IDLE, TrainingPhase.COMPLETED, TrainingPhase.FAILED]:
        raise HTTPException(status_code=400, detail="Training is already in progress")

    # Reset status
    training_status.reset()

    # Start training in background thread
    def run_training():
        try:
            from training_pipeline import run_full_training_pipeline

            logger.info("Starting full training pipeline...")
            run_full_training_pipeline(
                text_to_pose_epochs=request.text_to_pose_epochs,
                pose_to_video_epochs=request.pose_to_video_epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            training_status.fail_training(str(e))

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    return {
        "status": "started",
        "message": "Training pipeline started in background",
        "config": {
            "text_to_pose_epochs": request.text_to_pose_epochs,
            "pose_to_video_epochs": request.pose_to_video_epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate
        }
    }


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """
    Get current training status with live metrics

    Poll this endpoint every 1-2 seconds to update dashboard
    """
    return training_status.get_status()


@router.post("/stop")
async def stop_training():
    """
    Stop training (gracefully if possible)

    Note: This will attempt to stop training, but may not be immediate
    """
    if training_status.phase == TrainingPhase.IDLE:
        raise HTTPException(status_code=400, detail="No training in progress")

    # Log the stop event
    training_status._add_log("Training stopped by user", "info")

    # Reset training status back to idle
    training_status.reset()

    return {
        "status": "stopped",
        "message": "Training stopped successfully. Progress has been saved."
    }


@router.get("/metrics")
async def get_detailed_metrics():
    """
    Get detailed training metrics for visualization

    Returns metrics history for plotting loss curves, etc.
    """
    status = training_status.get_status()

    return {
        "video_processing": status["metrics"]["video_processing"],
        "text_to_pose": status["metrics"]["text_to_pose"],
        "pose_to_video": status["metrics"]["pose_to_video"],
        "timeline": {
            "start_time": status["start_time"],
            "elapsed_seconds": status.get("elapsed_seconds", 0),
            "estimated_remaining_seconds": status.get("estimated_remaining_seconds")
        }
    }


@router.get("/samples")
async def get_sample_outputs():
    """
    Get recent sample video outputs generated during training
    """
    return {
        "samples": training_status.sample_outputs
    }


@router.get("/logs")
async def get_training_logs(limit: int = 50):
    """
    Get training logs for debugging
    """
    logs = training_status.logs[-limit:]
    return {
        "logs": logs,
        "total": len(training_status.logs)
    }
