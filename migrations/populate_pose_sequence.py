#!/usr/bin/env python3
"""
Data Migration: Populate pose_sequence from frames_data
Transforms the old frames_data format into the new pose_sequence format
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db, Contribution
from sqlalchemy.orm import Session

def transform_frames_data_to_pose_sequence(frames_data):
    """
    Transform frames_data format to pose_sequence format

    frames_data format:
    [
        {
            "frame_number": 0,
            "timestamp": 0.052,
            "pose_landmarks": [{"x": 0.5, "y": 0.5, "z": -1.8, "visibility": 0.99}, ...],
            "left_hand_landmarks": [...],  # optional
            "right_hand_landmarks": [...]  # optional
        },
        ...
    ]

    pose_sequence format:
    [
        [[x, y, z, v], [x, y, z, v], ...],  # Frame 0: 75 landmarks (33 pose + 21 left hand + 21 right hand)
        [[x, y, z, v], [x, y, z, v], ...],  # Frame 1
        ...
    ]
    """
    if not frames_data:
        return None

    pose_sequence = []

    for frame in frames_data:
        frame_landmarks = []

        # Extract pose landmarks (33 points)
        pose_landmarks = frame.get('pose_landmarks', [])
        for landmark in pose_landmarks:
            if isinstance(landmark, dict):
                frame_landmarks.append([
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0),
                    landmark.get('visibility', 0.0)
                ])

        # Pad to 33 pose landmarks if needed
        while len(frame_landmarks) < 33:
            frame_landmarks.append([0.0, 0.0, 0.0, 0.0])

        # Extract left hand landmarks (21 points)
        left_hand = frame.get('left_hand_landmarks', [])
        for landmark in left_hand:
            if isinstance(landmark, dict):
                frame_landmarks.append([
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0),
                    landmark.get('visibility', 0.0)
                ])

        # Pad to 21 left hand landmarks if needed
        while len(frame_landmarks) < 54:  # 33 + 21
            frame_landmarks.append([0.0, 0.0, 0.0, 0.0])

        # Extract right hand landmarks (21 points)
        right_hand = frame.get('right_hand_landmarks', [])
        for landmark in right_hand:
            if isinstance(landmark, dict):
                frame_landmarks.append([
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0),
                    landmark.get('visibility', 0.0)
                ])

        # Pad to 21 right hand landmarks if needed
        while len(frame_landmarks) < 75:  # 33 + 21 + 21
            frame_landmarks.append([0.0, 0.0, 0.0, 0.0])

        pose_sequence.append(frame_landmarks)

    return pose_sequence


def calculate_hand_detection(pose_sequence):
    """Calculate has_left_hand and has_right_hand from pose sequence"""
    if not pose_sequence or len(pose_sequence) == 0:
        return False, False

    has_left = False
    has_right = False

    for frame in pose_sequence:
        if len(frame) > 33:
            # Check left hand landmarks (33-53)
            left_hand_visible = any(
                frame[i][3] > 0.5 for i in range(33, min(54, len(frame)))
            )
            if left_hand_visible:
                has_left = True

        if len(frame) > 54:
            # Check right hand landmarks (54-74)
            right_hand_visible = any(
                frame[i][3] > 0.5 for i in range(54, min(75, len(frame)))
            )
            if right_hand_visible:
                has_right = True

    return has_left, has_right


def calculate_fps(frames_data, num_frames, duration):
    """Calculate FPS from frames_data or fallback to num_frames/duration"""
    if frames_data and len(frames_data) > 0:
        # Try to calculate from timestamps
        first_frame = frames_data[0]
        last_frame = frames_data[-1]

        if 'timestamp' in first_frame and 'timestamp' in last_frame:
            time_diff = last_frame['timestamp'] - first_frame['timestamp']
            if time_diff > 0:
                return len(frames_data) / time_diff

    # Fallback to num_frames/duration
    if duration > 0:
        return num_frames / duration

    return 30.0  # Default FPS


def populate_pose_sequences():
    """Main migration function"""
    db: Session = next(get_db())

    try:
        # Get all contributions
        contributions = db.query(Contribution).all()
        print(f"Found {len(contributions)} contributions to process")

        updated_count = 0
        skipped_count = 0

        for contrib in contributions:
            # Skip if already has pose_sequence
            if contrib.pose_sequence is not None:
                print(f"  Skipping {contrib.id}: already has pose_sequence")
                skipped_count += 1
                continue

            # Skip if no frames_data
            if not contrib.frames_data:
                print(f"  Skipping {contrib.id}: no frames_data")
                skipped_count += 1
                continue

            print(f"  Processing contribution {contrib.id} ({contrib.word})...")

            # Transform frames_data to pose_sequence
            pose_sequence = transform_frames_data_to_pose_sequence(contrib.frames_data)

            if not pose_sequence:
                print(f"    Warning: Failed to transform frames_data")
                skipped_count += 1
                continue

            # Calculate derived fields
            has_left_hand, has_right_hand = calculate_hand_detection(pose_sequence)
            fps = calculate_fps(contrib.frames_data, contrib.num_frames, contrib.duration)
            data_points = len(pose_sequence) * 75  # frames * landmarks

            # Update contribution
            contrib.pose_sequence = pose_sequence
            contrib.fps = fps
            contrib.total_frames = len(pose_sequence)
            contrib.extracted_frames = len(pose_sequence)
            contrib.has_left_hand = has_left_hand
            contrib.has_right_hand = has_right_hand
            contrib.data_points = data_points

            print(f"    ✓ Updated: {len(pose_sequence)} frames, "
                  f"FPS: {fps:.1f}, "
                  f"Hands: L={has_left_hand} R={has_right_hand}")

            updated_count += 1

        # Commit all changes
        db.commit()
        print(f"\n✅ Migration complete!")
        print(f"   Updated: {updated_count}")
        print(f"   Skipped: {skipped_count}")
        print(f"   Total: {len(contributions)}")

    except Exception as e:
        db.rollback()
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Pose Sequence Population Migration")
    print("=" * 60)
    populate_pose_sequences()
