#!/usr/bin/env python3
"""
Full AI Training Pipeline

Orchestrates the complete training workflow:
1. Video Processing → Pose Extraction
2. Text-to-Pose Model Training (PyTorch)
3. Pose-to-Video Model Training (ControlNet + Stable Diffusion)

Integrates with TrainingStatus for live monitoring.

Author: SignForge Team
Date: 2025-01-11
"""

import sys
import logging
from pathlib import Path

# Import training monitor
from training_monitor import training_status

# Import training scripts
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

logger = logging.getLogger(__name__)


def run_full_training_pipeline(
    text_to_pose_epochs: int = 100,
    pose_to_video_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3
):
    """
    Execute complete AI training pipeline with live monitoring

    Args:
        text_to_pose_epochs: Number of epochs for text-to-pose training
        pose_to_video_epochs: Number of epochs for pose-to-video training
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    try:
        logger.info("=" * 70)
        logger.info("FULL AI TRAINING PIPELINE - REAL NEURAL NETWORK TRAINING")
        logger.info("=" * 70)

        # ==================================================================
        # PHASE 1: VIDEO PROCESSING
        # ==================================================================
        logger.info("\nPHASE 1: Processing Videos → Extracting Poses")
        logger.info("=" * 70)

        from process_all_videos import VideoProcessor

        processor = VideoProcessor(
            input_dir='data/signtalk-gsl/SignTalk-GH/Videos',
            output_dir='data/processed_poses'
        )

        # Get total video count
        video_files = list(Path('data/signtalk-gsl/SignTalk-GH/Videos').glob('*.mp4'))
        total_videos = len(video_files)

        logger.info(f"Found {total_videos} videos to process")

        # Initialize training status
        training_status.start_training(
            total_videos=total_videos,
            text_to_pose_epochs=text_to_pose_epochs,
            pose_to_video_epochs=pose_to_video_epochs
        )

        # Process videos with progress updates (use max_workers=2 to prevent segfaults)
        def progress_callback(processed, failed, storage_saved_mb):
            training_status.update_video_processing(processed, failed, storage_saved_mb)

        report = processor.process_all_videos(max_workers=2, progress_callback=progress_callback)

        logger.info(f"\n✅ Video processing complete:")
        logger.info(f"   - Processed: {report['summary']['success']}")
        logger.info(f"   - Failed: {report['summary']['failed']}")
        logger.info(f"   - Storage saved: {report['summary']['space_saved_mb']:.2f} MB")

        # ==================================================================
        # PHASE 2: TEXT-TO-POSE MODEL TRAINING
        # ==================================================================
        logger.info("\n\nPHASE 2: Training Text-to-Pose Model (PyTorch)")
        logger.info("=" * 70)

        training_status.start_text_to_pose_training()

        from train_text_to_pose_real import train_text_to_pose_model

        text_to_pose_model = train_text_to_pose_model(
            pose_data_dir='data/processed_poses',
            epochs=text_to_pose_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            progress_callback=lambda epoch, loss, samples: training_status.update_text_to_pose(epoch, loss, samples)
        )

        logger.info(f"\n✅ Text-to-Pose model training complete")
        logger.info(f"   Best loss: {training_status.metrics['text_to_pose']['best_loss']:.6f}")

        # ==================================================================
        # PHASE 3: POSE-TO-VIDEO MODEL TRAINING
        # ==================================================================
        logger.info("\n\nPHASE 3: Training Pose-to-Video Model (ControlNet)")
        logger.info("=" * 70)

        training_status.start_pose_to_video_training()

        from train_pose_to_video_real import train_pose_to_video_model

        pose_to_video_model = train_pose_to_video_model(
            pose_data_dir='data/processed_poses',
            video_data_dir='data/signtalk-gsl/SignTalk-GH/Videos',
            epochs=pose_to_video_epochs,
            batch_size=4,  # ControlNet needs smaller batch size
            learning_rate=1e-5,  # Smaller LR for fine-tuning
            progress_callback=lambda epoch, loss, samples: training_status.update_pose_to_video(epoch, loss, samples)
        )

        logger.info(f"\n✅ Pose-to-Video model training complete")
        logger.info(f"   Best loss: {training_status.metrics['pose_to_video']['best_loss']:.6f}")

        # ==================================================================
        # COMPLETE
        # ==================================================================
        training_status.complete_training()

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info("=" * 70)
        logger.info("\n✅ Models saved:")
        logger.info("   - Text-to-Pose: models/text_to_pose_best.pth")
        logger.info("   - Pose-to-Video: models/pose_to_video_controlnet/")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        training_status.fail_training(str(e))
        raise
