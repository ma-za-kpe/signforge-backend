"""
Tests for Video Processing Script

Tests the video processor that extracts MediaPipe pose landmarks from videos.

Author: SignForge Team
Date: 2025-01-11
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import json
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))


class TestVideoProcessor:
    """Test VideoProcessor class"""

    @pytest.fixture
    def mock_mediapipe(self):
        """Mock MediaPipe holistic module"""
        with patch('process_all_videos.mp') as mock_mp:
            # Mock holistic solution
            mock_holistic_instance = MagicMock()
            mock_mp.solutions.holistic.Holistic.return_value = mock_holistic_instance

            yield mock_mp, mock_holistic_instance

    @pytest.fixture
    def mock_video_capture(self):
        """Mock OpenCV VideoCapture"""
        with patch('process_all_videos.cv2.VideoCapture') as mock_cap:
            cap_instance = MagicMock()
            cap_instance.isOpened.return_value = True
            cap_instance.get.side_effect = lambda prop: {
                0: 30.0,  # FPS
                7: 100,   # Total frames
            }.get(prop, 0)

            # Mock frames
            cap_instance.read.side_effect = [
                (True, np.zeros((480, 640, 3), dtype=np.uint8)),
                (True, np.zeros((480, 640, 3), dtype=np.uint8)),
                (False, None)  # End of video
            ]

            mock_cap.return_value = cap_instance
            yield mock_cap

    def test_processor_initialization(self, mock_mediapipe):
        """Test VideoProcessor initializes correctly"""
        from process_all_videos import VideoProcessor

        processor = VideoProcessor(
            input_dir='test_input',
            output_dir='test_output'
        )

        assert processor.input_dir == Path('test_input')
        assert processor.output_dir == Path('test_output')

    def test_find_all_videos(self, tmp_path, mock_mediapipe):
        """Test finding video files in directory"""
        from process_all_videos import VideoProcessor

        # Create test video files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()
        (tmp_path / "video3.avi").touch()
        (tmp_path / "not_video.txt").touch()

        processor = VideoProcessor(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / 'output')
        )

        videos = processor.find_all_videos()

        # Should find 3 video files
        video_names = [v.name for v in videos]
        assert len(videos) == 3
        assert "video1.mp4" in video_names
        assert "video2.mp4" in video_names
        assert "video3.avi" in video_names
        assert "not_video.txt" not in video_names

    def test_extract_word_from_filename(self, mock_mediapipe):
        """Test extracting word labels from filenames"""
        from process_all_videos import VideoProcessor

        processor = VideoProcessor()

        # Test various filename formats
        assert processor.extract_word_from_filename(Path("fever_001.mp4")) == "FEVER"
        assert processor.extract_word_from_filename(Path("pain_video_02.mp4")) == "PAIN"
        assert processor.extract_word_from_filename(Path("doctor.mp4")) == "DOCTOR"
        assert processor.extract_word_from_filename(Path("hello_world_123.mp4")) == "HELLO"

    def test_extract_poses_from_video(self, mock_mediapipe, mock_video_capture):
        """Test extracting pose landmarks from video"""
        from process_all_videos import VideoProcessor

        mock_mp, mock_holistic = mock_mediapipe

        # Mock pose landmarks
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmark.visibility = 1.0

        mock_pose_landmarks = MagicMock()
        mock_pose_landmarks.landmark = [mock_landmark] * 33  # 33 pose landmarks

        mock_results = MagicMock()
        mock_results.pose_landmarks = mock_pose_landmarks

        mock_holistic.process.return_value = mock_results

        processor = VideoProcessor()
        result = processor.extract_poses_from_video(Path("test.mp4"))

        assert 'poses' in result
        assert 'fps' in result
        assert 'total_frames' in result
        assert 'extracted_frames' in result
        assert len(result['poses']) > 0

    def test_process_single_video_success(self, tmp_path, mock_mediapipe, mock_video_capture):
        """Test successfully processing a single video"""
        from process_all_videos import VideoProcessor

        mock_mp, mock_holistic = mock_mediapipe

        # Mock pose landmarks
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmark.visibility = 1.0

        mock_pose_landmarks = MagicMock()
        mock_pose_landmarks.landmark = [mock_landmark] * 33

        mock_results = MagicMock()
        mock_results.pose_landmarks = mock_pose_landmarks

        mock_holistic.process.return_value = mock_results

        # Create test video file
        test_video = tmp_path / "fever_001.mp4"
        test_video.write_bytes(b"fake video data")

        processor = VideoProcessor(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / 'output')
        )

        result = processor.process_single_video(test_video)

        assert result['status'] == 'success'
        assert result['word'] == 'FEVER'
        assert 'video_size_mb' in result
        assert 'pose_size_mb' in result

    def test_process_single_video_skip_existing(self, tmp_path, mock_mediapipe):
        """Test skipping already processed videos"""
        from process_all_videos import VideoProcessor

        # Create test video file
        test_video = tmp_path / "test.mp4"
        test_video.write_bytes(b"fake video")

        # Create existing output file
        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        output_file = output_dir / "test_poses.json"
        output_file.write_text('{"poses": []}')

        processor = VideoProcessor(
            input_dir=str(tmp_path),
            output_dir=str(output_dir)
        )

        result = processor.process_single_video(test_video)

        assert result['status'] == 'skipped'

    def test_process_single_video_failure(self, tmp_path, mock_mediapipe, mock_video_capture):
        """Test handling video processing failure"""
        from process_all_videos import VideoProcessor

        mock_mp, mock_holistic = mock_mediapipe

        # Make processing raise an error
        mock_holistic.process.side_effect = Exception("Processing error")

        # Create test video file
        test_video = tmp_path / "test.mp4"
        test_video.write_bytes(b"fake video")

        processor = VideoProcessor(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / 'output')
        )

        result = processor.process_single_video(test_video)

        assert result['status'] == 'failed'
        assert 'error' in result

    def test_progress_callback(self, tmp_path, mock_mediapipe, mock_video_capture):
        """Test progress callback functionality"""
        from process_all_videos import VideoProcessor

        mock_mp, mock_holistic = mock_mediapipe

        # Mock successful processing
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmark.visibility = 1.0

        mock_pose_landmarks = MagicMock()
        mock_pose_landmarks.landmark = [mock_landmark] * 33

        mock_results = MagicMock()
        mock_results.pose_landmarks = mock_pose_landmarks

        mock_holistic.process.return_value = mock_results

        # Create test video
        test_video = tmp_path / "test.mp4"
        test_video.write_bytes(b"fake video")

        processor = VideoProcessor(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / 'output')
        )

        # Track callback invocations
        callback_calls = []

        def progress_callback(processed, failed, storage_saved_mb):
            callback_calls.append({
                'processed': processed,
                'failed': failed,
                'storage_saved_mb': storage_saved_mb
            })

        # Process with callback
        with patch.object(processor, 'find_all_videos', return_value=[test_video]):
            processor.process_all_videos(max_workers=1, progress_callback=progress_callback)

        # Callback should have been called
        assert len(callback_calls) > 0

    def test_generate_report(self, mock_mediapipe):
        """Test generating processing summary report"""
        from process_all_videos import VideoProcessor

        processor = VideoProcessor()

        results = {
            'success': [
                {
                    'status': 'success',
                    'word': 'HELLO',
                    'video_size_mb': 10.0,
                    'pose_size_mb': 0.5,
                    'reduction': 20.0
                },
                {
                    'status': 'success',
                    'word': 'WORLD',
                    'video_size_mb': 8.0,
                    'pose_size_mb': 0.4,
                    'reduction': 20.0
                }
            ],
            'failed': [
                {'status': 'failed', 'video': 'bad.mp4', 'error': 'Corrupt file'}
            ],
            'skipped': []
        }

        with patch('builtins.open', mock_open()):
            with patch('json.dump'):
                report = processor.generate_report(results)

        assert 'summary' in report
        assert report['summary']['total'] == 3
        assert report['summary']['success'] == 2
        assert report['summary']['failed'] == 1
        assert report['summary']['space_saved_mb'] > 0


class TestVideoProcessorIntegration:
    """Integration tests for video processor"""

    def test_full_pipeline_mock(self, tmp_path):
        """Test full processing pipeline with mocked components"""
        from process_all_videos import VideoProcessor

        with patch('process_all_videos.mp') as mock_mp, \
             patch('process_all_videos.cv2.VideoCapture') as mock_cap:

            # Setup mocks
            mock_holistic = MagicMock()
            mock_mp.solutions.holistic.Holistic.return_value = mock_holistic

            # Mock video capture
            cap_instance = MagicMock()
            cap_instance.isOpened.return_value = True
            cap_instance.get.side_effect = lambda prop: {0: 30.0, 7: 10}.get(prop, 0)
            cap_instance.read.side_effect = [(False, None)]
            cap_instance.release = MagicMock()
            mock_cap.return_value = cap_instance

            # Mock pose results
            mock_landmark = MagicMock()
            mock_landmark.x = 0.5
            mock_landmark.y = 0.5
            mock_landmark.z = 0.0
            mock_landmark.visibility = 1.0

            mock_pose_landmarks = MagicMock()
            mock_pose_landmarks.landmark = [mock_landmark] * 33

            mock_results = MagicMock()
            mock_results.pose_landmarks = mock_pose_landmarks

            mock_holistic.process.return_value = mock_results

            # Create test videos
            (tmp_path / "video1.mp4").write_bytes(b"fake video 1")
            (tmp_path / "video2.mp4").write_bytes(b"fake video 2")

            processor = VideoProcessor(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / 'output')
            )

            # Process all videos
            report = processor.process_all_videos(max_workers=1)

            assert report is not None
            assert 'summary' in report


class TestDataStructures:
    """Test data structures and formats"""

    def test_pose_data_structure(self):
        """Test that pose data has correct structure"""
        pose_data = {
            'word': 'HELLO',
            'video_path': '/path/to/video.mp4',
            'video_filename': 'video.mp4',
            'fps': 30.0,
            'total_frames': 100,
            'extracted_frames': 95,
            'pose_sequence': [
                # Frame 1: 33 landmarks × 4 coordinates
                [[0.5, 0.5, 0.0, 1.0] for _ in range(33)]
            ],
            'processed_at': '2025-01-11T10:00:00'
        }

        # Validate structure
        assert 'word' in pose_data
        assert 'pose_sequence' in pose_data
        assert len(pose_data['pose_sequence']) > 0
        assert len(pose_data['pose_sequence'][0]) == 33
        assert len(pose_data['pose_sequence'][0][0]) == 4

    def test_processing_report_structure(self):
        """Test that processing report has correct structure"""
        report = {
            'summary': {
                'total': 100,
                'success': 95,
                'failed': 5,
                'skipped': 0,
                'space_saved_mb': 5000.0
            },
            'results': {
                'success': [],
                'failed': [],
                'skipped': []
            },
            'processed_at': '2025-01-11T10:00:00'
        }

        assert 'summary' in report
        assert 'results' in report
        assert report['summary']['total'] == report['summary']['success'] + report['summary']['failed']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
