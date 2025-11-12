"""
Video-to-Pose Extraction Pipeline for User Contributions
Extracts MediaPipe Holistic landmarks from uploaded videos
Same pipeline as reference skeleton import (import_signtalk_skeletons.py)
"""
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic


def process_video_to_poses(video_path: Path) -> Dict:
    """
    Extract pose landmarks from video using MediaPipe Holistic.

    Returns full landmark sequence with 75 landmarks per frame:
    - 33 pose landmarks (body)
    - 21 left hand landmarks
    - 21 right hand landmarks

    Args:
        video_path: Path to uploaded video file

    Returns:
        {
            "pose_sequence": List[List[List[float]]],  # [frame][landmark][x,y,z,vis]
            "fps": float,
            "total_frames": int,
            "extracted_frames": int,
            "duration": float
        }

    Raises:
        ValueError: If video cannot be opened or processed
    """

    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get actual video duration from metadata (more reliable than frame count)
    actual_duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if cap.get(cv2.CAP_PROP_POS_MSEC) > 0 else (total_frames / fps)

    logger.info(f"Processing video: {video_path.name} ({total_frames} frames @ {fps} FPS, duration: {actual_duration:.2f}s)")

    pose_sequence = []

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as holistic:

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Convert BGR (OpenCV) to RGB (MediaPipe)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                results = holistic.process(rgb_frame)

                # Extract 75 landmarks (33 pose + 21 left hand + 21 right hand)
                frame_landmarks = []

                # Pose landmarks (33 points)
                if results.pose_landmarks:
                    for lm in results.pose_landmarks.landmark:
                        frame_landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    # Pad with zeros if no pose detected
                    frame_landmarks.extend([[0.0, 0.0, 0.0, 0.0]] * 33)

                # Left hand landmarks (21 points)
                if results.left_hand_landmarks:
                    for lm in results.left_hand_landmarks.landmark:
                        frame_landmarks.append([lm.x, lm.y, lm.z, 1.0])
                else:
                    frame_landmarks.extend([[0.0, 0.0, 0.0, 0.0]] * 21)

                # Right hand landmarks (21 points)
                if results.right_hand_landmarks:
                    for lm in results.right_hand_landmarks.landmark:
                        frame_landmarks.append([lm.x, lm.y, lm.z, 1.0])
                else:
                    frame_landmarks.extend([[0.0, 0.0, 0.0, 0.0]] * 21)

                pose_sequence.append(frame_landmarks)

    finally:
        cap.release()

    extracted_frames = len(pose_sequence)
    # Use actual video duration from metadata, not calculated from extracted frames
    # (WebM videos can extract duplicate frames causing incorrect duration calculation)
    duration = actual_duration

    logger.info(f" Extracted {extracted_frames} frames from {video_path.name}")

    if extracted_frames == 0:
        raise ValueError("No frames could be extracted from video")

    return {
        "pose_sequence": pose_sequence,
        "fps": fps,
        "total_frames": total_frames,
        "extracted_frames": extracted_frames,
        "duration": round(duration, 2)
    }


def auto_detect_sign_characteristics(pose_sequence: List[List[List[float]]]) -> Dict[str, str]:
    """
    Automatically detect sign characteristics from pose sequence.

    Args:
        pose_sequence: List of frames, each with 75 landmarks [x,y,z,vis]

    Returns:
        {
            "sign_type_movement": "static" or "dynamic",
            "sign_type_hands": "one-handed" or "two-handed"
        }
    """
    if not pose_sequence or len(pose_sequence) < 2:
        return {
            "sign_type_movement": "dynamic",
            "sign_type_hands": "two-handed"
        }

    # Detect hand usage (one-handed vs two-handed)
    left_hand_visible_count = 0
    right_hand_visible_count = 0

    for frame in pose_sequence:
        # Left hand: landmarks 33-53
        left_hand_vis = sum(lm[3] for lm in frame[33:54]) / 21 if len(frame) > 53 else 0
        # Right hand: landmarks 54-74
        right_hand_vis = sum(lm[3] for lm in frame[54:75]) / 21 if len(frame) > 74 else 0

        if left_hand_vis > 0.3:  # Hand is visible if avg visibility > 30%
            left_hand_visible_count += 1
        if right_hand_vis > 0.3:
            right_hand_visible_count += 1

    # If both hands visible in >40% of frames, it's two-handed
    total_frames = len(pose_sequence)
    both_hands_threshold = 0.4

    if (left_hand_visible_count / total_frames > both_hands_threshold and
        right_hand_visible_count / total_frames > both_hands_threshold):
        sign_type_hands = "two-handed"
    else:
        sign_type_hands = "one-handed"

    # Detect movement (static vs dynamic)
    # Calculate average motion across all landmarks
    total_motion = 0.0
    motion_frames = 0

    for i in range(1, len(pose_sequence)):
        prev_frame = pose_sequence[i-1]
        curr_frame = pose_sequence[i]

        frame_motion = 0.0
        visible_landmarks = 0

        for j in range(min(len(prev_frame), len(curr_frame))):
            # Only consider landmarks with good visibility
            if prev_frame[j][3] > 0.5 and curr_frame[j][3] > 0.5:
                # Euclidean distance in x,y space (ignore z for motion detection)
                dx = curr_frame[j][0] - prev_frame[j][0]
                dy = curr_frame[j][1] - prev_frame[j][1]
                motion = (dx**2 + dy**2) ** 0.5
                frame_motion += motion
                visible_landmarks += 1

        if visible_landmarks > 0:
            total_motion += frame_motion / visible_landmarks
            motion_frames += 1

    avg_motion = total_motion / motion_frames if motion_frames > 0 else 0.0

    # Threshold: if average motion per frame < 0.02 (2% of frame), it's static
    # This is a normalized value where 1.0 = full frame diagonal
    motion_threshold = 0.02
    sign_type_movement = "static" if avg_motion < motion_threshold else "dynamic"

    logger.info(f"Auto-detected: movement={sign_type_movement} (motion={avg_motion:.4f}), hands={sign_type_hands}")

    return {
        "sign_type_movement": sign_type_movement,
        "sign_type_hands": sign_type_hands
    }


def calculate_quality_metrics(pose_sequence: List[List[List[float]]],
                              sign_type_movement: Optional[str] = None,
                              sign_type_hands: Optional[str] = None) -> Dict:
    """
    Calculate quality metrics for extracted pose sequence.

    Same scoring as contribution_api.py:
    - Hand visibility (50% weight)
    - Motion smoothness (30% weight)
    - Frame completeness (20% weight)
    - Lighting quality (informational)

    Args:
        pose_sequence: List of frames, each with 75 landmarks [x,y,z,vis]
        sign_type_movement: 'static' or 'dynamic' (affects smoothness scoring)
        sign_type_hands: 'one-handed' or 'two-handed' (affects hand visibility scoring)

    Returns:
        {
            "overall_score": float (0.0-1.0),
            "hand_visibility": float (0.0-1.0),
            "motion_smoothness": float (0.0-1.0),
            "frame_completeness": float (0.0-1.0),
            "lighting_quality": float (0.0-1.0)
        }
    """

    if not pose_sequence or len(pose_sequence) == 0:
        return {
            "overall_score": 0.0,
            "hand_visibility": 0.0,
            "motion_smoothness": 0.0,
            "frame_completeness": 0.0,
            "lighting_quality": 0.0
        }

    # 1. Hand Visibility Score (weight 0.5)
    hand_vis_scores = []
    for frame in pose_sequence:
        if len(frame) < 75:
            hand_vis_scores.append(0.0)
            continue

        left_hand = frame[33:54]  # Indices 33-53 (21 landmarks)
        right_hand = frame[54:75]  # Indices 54-74 (21 landmarks)

        left_vis = sum(lm[3] for lm in left_hand) / len(left_hand) if left_hand else 0.0
        right_vis = sum(lm[3] for lm in right_hand) / len(right_hand) if right_hand else 0.0

        # Context-aware scoring based on sign type
        if sign_type_hands == 'one-handed':
            # For one-handed signs, score based on best hand
            hand_vis_scores.append(max(left_vis, right_vis))
        else:
            # For two-handed signs, expect both hands visible
            visible_hands = (1 if left_vis > 0.5 else 0) + (1 if right_vis > 0.5 else 0)
            hand_vis_scores.append(visible_hands / 2.0)

    hand_visibility = sum(hand_vis_scores) / len(hand_vis_scores)

    # 2. Motion Smoothness Score (weight 0.3)
    smoothness_scores = []
    for i in range(1, len(pose_sequence)):
        prev_frame = pose_sequence[i - 1]
        curr_frame = pose_sequence[i]

        if len(prev_frame) < 75 or len(curr_frame) < 75:
            continue

        # Calculate wrist movement (landmark indices 15 and 16 for left/right wrists)
        left_wrist_prev = prev_frame[15]
        left_wrist_curr = curr_frame[15]

        # Euclidean distance
        distance = (
            (left_wrist_curr[0] - left_wrist_prev[0]) ** 2 +
            (left_wrist_curr[1] - left_wrist_prev[1]) ** 2 +
            (left_wrist_curr[2] - left_wrist_prev[2]) ** 2
        ) ** 0.5

        # Context-aware smoothness
        if sign_type_movement == 'static':
            # For static signs, penalize large movements
            smoothness = 1.0 - min(distance * 5, 1.0)
        else:
            # For dynamic signs, moderate movement is good
            # Too much = jerky, too little = static
            ideal_distance = 0.05  # Ideal moderate movement
            deviation = abs(distance - ideal_distance)
            smoothness = 1.0 - min(deviation * 10, 1.0)

        smoothness_scores.append(smoothness)

    motion_smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 0.5

    # 3. Frame Completeness Score (weight 0.2)
    complete_frames = 0
    for frame in pose_sequence:
        if len(frame) < 75:
            continue

        # Frame is complete if average visibility > 0.5
        avg_vis = sum(lm[3] for lm in frame) / len(frame)
        if avg_vis > 0.5:
            complete_frames += 1

    frame_completeness = complete_frames / len(pose_sequence)

    # 4. Lighting Quality (informational - not weighted in overall score)
    lighting_scores = []
    for frame in pose_sequence:
        if len(frame) < 75:
            continue
        avg_vis = sum(lm[3] for lm in frame) / len(frame)
        lighting_scores.append(avg_vis)

    lighting_quality = sum(lighting_scores) / len(lighting_scores) if lighting_scores else 0.0

    # Overall Score (weighted average)
    overall_score = (
        hand_visibility * 0.5 +
        motion_smoothness * 0.3 +
        frame_completeness * 0.2
    )

    return {
        "overall_score": round(overall_score, 3),
        "hand_visibility": round(hand_visibility, 3),
        "motion_smoothness": round(motion_smoothness, 3),
        "frame_completeness": round(frame_completeness, 3),
        "lighting_quality": round(lighting_quality, 3)
    }


def get_quality_label(score: float) -> str:
    """Convert quality score to human-readable label."""
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.50:
        return "Acceptable"
    else:
        return "Poor"
