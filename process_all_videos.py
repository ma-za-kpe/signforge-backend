#!/usr/bin/env python3
"""
Process All Videos - Phase 2: Data Processing

Extracts MediaPipe pose landmarks from ALL videos in the dataset.
This converts large video files into compact pose JSON files for training.

Input:  data/signtalk-gsl/*.mp4 (10,000 videos, ~50 GB)
Output: data/processed_poses/*.json (10,000 pose files, ~250 MB)

Usage:
    python scripts/process_all_videos.py
    python scripts/process_all_videos.py --input data/signtalk-gsl --output data/processed_poses

Author: SignForge Team
Date: 2025-01-11
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed_poses/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process videos to extract pose landmarks"""

    def __init__(self, input_dir='data/signtalk-gsl', output_dir='data/processed_poses'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("VIDEO PROCESSING - POSE LANDMARK EXTRACTION")
        logger.info("=" * 70)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        # Initialize MediaPipe - will be created per-thread to avoid segfaults
        self.mp_holistic = mp.solutions.holistic

    def find_all_videos(self):
        """Find all video files in input directory"""
        logger.info("\nScanning for video files...")

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []

        for ext in video_extensions:
            videos.extend(self.input_dir.rglob(f'*{ext}'))

        logger.info(f"Found {len(videos)} video files")
        return videos

    def extract_word_from_filename(self, video_path):
        """
        Extract word label from video filename

        Examples:
            fever_001.mp4 -> FEVER
            pain_video_02.mp4 -> PAIN
            doctor.mp4 -> DOCTOR
        """
        stem = video_path.stem.lower()

        # Remove common patterns like numbers and "video"
        word = stem.split('_')[0]  # Take first part before underscore
        word = ''.join(c for c in word if c.isalpha())  # Remove numbers

        return word.upper()

    def extract_poses_from_video(self, video_path):
        """Extract MediaPipe pose landmarks from video"""
        cap = None
        holistic = None

        try:
            # Create MediaPipe instance for this video (thread-safe)
            holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            poses = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            max_frames = min(total_frames, 300)  # Limit to 300 frames (10 seconds at 30fps) to prevent memory issues

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                try:
                    # Convert BGR to RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process with MediaPipe
                    results = holistic.process(rgb)

                    if results.pose_landmarks:
                        # Extract pose landmarks (33 points)
                        pose_frame = []
                        for landmark in results.pose_landmarks.landmark:
                            pose_frame.append([
                                float(landmark.x),
                                float(landmark.y),
                                float(landmark.z),
                                float(landmark.visibility)
                            ])

                        # Extract left hand landmarks (21 points)
                        if results.left_hand_landmarks:
                            for landmark in results.left_hand_landmarks.landmark:
                                pose_frame.append([
                                    float(landmark.x),
                                    float(landmark.y),
                                    float(landmark.z),
                                    1.0  # Hands don't have visibility, use 1.0
                                ])
                        else:
                            # No left hand detected, add 21 empty landmarks
                            for _ in range(21):
                                pose_frame.append([0.0, 0.0, 0.0, 0.0])

                        # Extract right hand landmarks (21 points)
                        if results.right_hand_landmarks:
                            for landmark in results.right_hand_landmarks.landmark:
                                pose_frame.append([
                                    float(landmark.x),
                                    float(landmark.y),
                                    float(landmark.z),
                                    1.0  # Hands don't have visibility, use 1.0
                                ])
                        else:
                            # No right hand detected, add 21 empty landmarks
                            for _ in range(21):
                                pose_frame.append([0.0, 0.0, 0.0, 0.0])

                        # Now pose_frame has 75 landmarks total (33 + 21 + 21)
                        poses.append(pose_frame)
                except Exception as e:
                    # Skip this frame if MediaPipe fails
                    logger.debug(f"Frame {frame_count} processing failed for {video_path.name}: {e}")
                    continue

            return {
                'poses': poses,
                'fps': fps,
                'total_frames': total_frames,
                'extracted_frames': len(poses)
            }

        except Exception as e:
            raise Exception(f"Video processing failed: {str(e)}")

        finally:
            # Clean up resources
            if cap is not None:
                cap.release()
            if holistic is not None:
                holistic.close()

    def process_single_video(self, video_path):
        """Process a single video file"""
        try:
            # Extract word label
            word = self.extract_word_from_filename(video_path)

            # Output filename
            output_file = self.output_dir / f"{video_path.stem}_poses.json"

            # Skip if already processed
            if output_file.exists():
                logger.debug(f"Skipping {video_path.name} (already processed)")
                return {
                    'status': 'skipped',
                    'video': str(video_path),
                    'output': str(output_file)
                }

            # Extract poses
            result = self.extract_poses_from_video(video_path)

            # Calculate file sizes
            video_size_mb = video_path.stat().st_size / (1024 * 1024)

            # Save pose data
            pose_data = {
                'word': word,
                'video_path': str(video_path),
                'video_filename': video_path.name,
                'fps': result['fps'],
                'total_frames': result['total_frames'],
                'extracted_frames': result['extracted_frames'],
                'pose_sequence': result['poses'],
                'processed_at': datetime.now().isoformat()
            }

            with open(output_file, 'w') as f:
                json.dump(pose_data, f)

            pose_size_mb = output_file.stat().st_size / (1024 * 1024)
            reduction = video_size_mb / pose_size_mb if pose_size_mb > 0 else 0

            return {
                'status': 'success',
                'video': str(video_path),
                'output': str(output_file),
                'word': word,
                'frames': result['extracted_frames'],
                'video_size_mb': video_size_mb,
                'pose_size_mb': pose_size_mb,
                'reduction': reduction
            }

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return {
                'status': 'failed',
                'video': str(video_path),
                'error': str(e)
            }

    def process_all_videos(self, max_workers=2, progress_callback=None):
        """
        Process all videos with parallel processing

        Args:
            max_workers: Number of parallel workers (default 2 to prevent segfaults)
            progress_callback: Optional callback function(processed, failed, storage_saved_mb)
                              called after each video is processed
        """
        videos = self.find_all_videos()

        if not videos:
            logger.error("No videos found to process!")
            return

        logger.info(f"\nProcessing {len(videos)} videos with {max_workers} workers...")
        logger.info("This may take several hours depending on dataset size.")
        logger.info("Note: Using reduced parallelism to prevent MediaPipe segmentation faults.\n")

        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }

        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_video, video): video
                      for video in videos}

            # Progress bar
            with tqdm(total=len(videos), desc="Processing videos") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results[result['status']].append(result)
                    pbar.update(1)

                    # Call progress callback
                    if progress_callback:
                        processed = len(results['success']) + len(results['skipped'])
                        failed = len(results['failed'])

                        # Calculate storage saved
                        storage_saved_mb = 0
                        if results['success']:
                            total_video_size = sum(r['video_size_mb'] for r in results['success'])
                            total_pose_size = sum(r['pose_size_mb'] for r in results['success'])
                            storage_saved_mb = total_video_size - total_pose_size

                        progress_callback(processed, failed, storage_saved_mb)

        # Generate summary report
        report = self.generate_report(results)

        return report

    def generate_report(self, results):
        """Generate processing summary report"""
        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)

        total = len(results['success']) + len(results['failed']) + len(results['skipped'])

        logger.info(f"\nTotal videos: {total}")
        logger.info(f"Successfully processed: {len(results['success'])}")
        logger.info(f"Failed: {len(results['failed'])}")
        logger.info(f"Skipped (already processed): {len(results['skipped'])}")

        # Calculate statistics
        total_video_size = 0
        total_pose_size = 0
        avg_reduction = 0
        space_saved_mb = 0

        if results['success']:
            total_video_size = sum(r['video_size_mb'] for r in results['success'])
            total_pose_size = sum(r['pose_size_mb'] for r in results['success'])
            avg_reduction = sum(r['reduction'] for r in results['success']) / len(results['success'])
            space_saved_mb = total_video_size - total_pose_size

            logger.info(f"\nStorage Statistics:")
            logger.info(f"Total video size: {total_video_size:.2f} MB")
            logger.info(f"Total pose size: {total_pose_size:.2f} MB")
            logger.info(f"Average reduction: {avg_reduction:.0f}× smaller")
            logger.info(f"Space saved: {space_saved_mb:.2f} MB")

        # Create report data
        report_data = {
            'summary': {
                'total': total,
                'success': len(results['success']),
                'failed': len(results['failed']),
                'skipped': len(results['skipped']),
                'space_saved_mb': space_saved_mb
            },
            'results': results,
            'processed_at': datetime.now().isoformat()
        }

        # Save detailed report
        report_file = self.output_dir / 'processing_report.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"\nDetailed report saved: {report_file}")

        if results['failed']:
            logger.warning(f"\n{len(results['failed'])} videos failed to process:")
            for r in results['failed'][:10]:  # Show first 10 failures
                logger.warning(f"  - {r['video']}: {r['error']}")

        return report_data


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process videos to extract pose landmarks')
    parser.add_argument('--input', default='data/signtalk-gsl',
                       help='Input directory containing videos')
    parser.add_argument('--output', default='data/processed_poses',
                       help='Output directory for pose JSON files')
    parser.add_argument('--workers', type=int, default=6,
                       help='Number of parallel workers (optimized for CPU performance)')

    args = parser.parse_args()

    processor = VideoProcessor(args.input, args.output)
    processor.process_all_videos(max_workers=args.workers)


if __name__ == '__main__':
    main()
