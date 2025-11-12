#!/usr/bin/env python3
"""
Community Contribution API for SignForge GSL Platform.

This API allows community members to contribute sign language recordings
by submitting MediaPipe landmark data extracted in-browser.

Endpoints:
- POST /api/contribute - Submit landmark data for a sign
- GET /api/contribution-stats/{word} - Get contribution statistics for a word
- GET /api/contribution-leaderboard - Get top contributors
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import uuid
from database import get_db, Contribution as DBContribution, Word, SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import func

# Create API router
router = APIRouter(prefix="/api", tags=["contributions"])

# Storage directory
CONTRIBUTIONS_DIR = Path(__file__).parent.parent.parent / "backend" / "contributions"
CONTRIBUTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================
# Pydantic Models
# ===========================

class Landmark(BaseModel):
    """Single 3D landmark point from MediaPipe."""
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate (0-1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate (0-1)")
    z: float = Field(..., description="Depth coordinate (relative to wrist)")
    visibility: float = Field(default=1.0, ge=0.0, le=1.0, description="Visibility confidence")


class Frame(BaseModel):
    """Single frame of landmark data."""
    frame_number: int = Field(..., ge=0, description="Frame index (0-based)")
    timestamp: float = Field(..., description="Timestamp in seconds from recording start")
    pose_landmarks: List[Landmark] = Field(..., min_items=33, max_items=33, description="33 pose landmarks")
    left_hand_landmarks: Optional[List[Landmark]] = Field(None, min_items=21, max_items=21, description="21 left hand landmarks")
    right_hand_landmarks: Optional[List[Landmark]] = Field(None, min_items=21, max_items=21, description="21 right hand landmarks")
    face_landmarks: Optional[List[Landmark]] = Field(None, description="468 face landmarks (optional)")

    @validator('pose_landmarks')
    def validate_pose(cls, v):
        if len(v) != 33:
            raise ValueError(f"Expected 33 pose landmarks, got {len(v)}")
        return v

    @validator('left_hand_landmarks', 'right_hand_landmarks')
    def validate_hands(cls, v):
        if v is not None and len(v) != 21:
            raise ValueError(f"Expected 21 hand landmarks, got {len(v)}")
        return v


class Contribution(BaseModel):
    """Complete contribution submission."""
    word: str = Field(..., min_length=1, max_length=100, description="GSL word being signed")
    user_id: str = Field(..., min_length=1, max_length=100, description="Anonymous user identifier")
    frames: List[Frame] = Field(..., min_items=30, max_items=150, description="30-150 frames (1-5 seconds @ 30fps)")
    duration: float = Field(..., gt=0.5, lt=5.0, description="Recording duration in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    # User's sign classification
    sign_type_movement: Optional[str] = Field(None, description="'static' or 'dynamic'")
    sign_type_hands: Optional[str] = Field(None, description="'one-handed' or 'two-handed'")

    # 3-attempt metadata (for averaged contributions)
    num_attempts: Optional[int] = Field(1, description="Number of attempts averaged (1 or 3)")
    individual_qualities: Optional[List[float]] = Field(None, description="Quality scores for each attempt")
    individual_durations: Optional[List[float]] = Field(None, description="Durations for each attempt")
    quality_variance: Optional[float] = Field(None, description="Variance of quality scores")
    improvement_trend: Optional[str] = Field(None, description="Trend across attempts")

    @validator('word')
    def normalize_word(cls, v):
        return v.upper().strip()

    @validator('frames')
    def validate_frame_count(cls, v):
        if not (30 <= len(v) <= 150):
            raise ValueError(f"Expected 30-150 frames, got {len(v)}")
        return v


class QualityBreakdown(BaseModel):
    """Detailed quality score breakdown."""
    overall_score: float
    hand_visibility: float
    motion_smoothness: float
    frame_completeness: float
    lighting_quality: float
    components: Dict[str, str]  # Human-readable labels
    recommendations: List[str]  # Actionable advice


class ContributionResponse(BaseModel):
    """Response after successful contribution submission."""
    contribution_id: str
    word: str
    quality_score: float
    quality_breakdown: Optional[QualityBreakdown] = None
    total_contributions: int
    progress_percentage: float
    message: str


class ContributionStats(BaseModel):
    """Statistics for contributions to a specific word."""
    word: str
    total_contributions: int
    unique_contributors: int
    average_quality: float
    ready_for_training: bool
    contributions_needed: int


# ===========================
# Quality Scoring
# ===========================

def calculate_lighting_quality(frames: List[Frame]) -> float:
    """
    Calculate lighting quality based on landmark visibility.

    Good lighting = high average visibility across all landmarks.
    Poor lighting = low visibility, detected hands/pose flickering.

    Args:
        frames: List of landmark frames

    Returns:
        Lighting quality score (0.0 - 1.0)
    """
    if not frames:
        return 0.0

    visibility_scores = []
    for frame in frames:
        frame_visibility = []

        # Pose visibility
        if frame.pose_landmarks:
            pose_vis = [lm.visibility for lm in frame.pose_landmarks]
            frame_visibility.extend(pose_vis)

        # Hand visibility
        if frame.left_hand_landmarks:
            left_vis = [lm.visibility for lm in frame.left_hand_landmarks]
            frame_visibility.extend(left_vis)

        if frame.right_hand_landmarks:
            right_vis = [lm.visibility for lm in frame.right_hand_landmarks]
            frame_visibility.extend(right_vis)

        if frame_visibility:
            visibility_scores.append(sum(frame_visibility) / len(frame_visibility))

    if not visibility_scores:
        return 0.0

    avg_visibility = sum(visibility_scores) / len(visibility_scores)
    return round(avg_visibility, 3)


def calculate_quality_score(frames: List[Frame], sign_type_movement: str = None, sign_type_hands: str = None) -> float:
    """
    Calculate quality score for a contribution (0.0 - 1.0).

    Context-aware scoring based on user classification:
    - Static signs: Prioritize stability over smoothness
    - Dynamic signs: Prioritize motion smoothness
    - One-handed signs: Only score the active hand
    - Two-handed signs: Score both hands

    Factors:
    1. Hand visibility (50% weight) - adjusted based on one/two-handed
    2. Motion smoothness (30% weight) - adjusted based on static/dynamic
    3. Frame completeness (20% weight)

    Args:
        frames: List of landmark frames
        sign_type_movement: 'static' or 'dynamic' (from user classification)
        sign_type_hands: 'one-handed' or 'two-handed' (from user classification)

    Returns:
        Quality score between 0.0 and 1.0
    """
    if not frames:
        return 0.0

    # 1. Hand Visibility Score (50%)
    # Adjust expectation based on one-handed vs two-handed
    hand_visibility_scores = []
    for frame in frames:
        visible_hands = 0
        left_vis = 0.0
        right_vis = 0.0

        if frame.left_hand_landmarks and len(frame.left_hand_landmarks) > 0:
            left_vis = sum(lm.visibility for lm in frame.left_hand_landmarks) / len(frame.left_hand_landmarks)
            if left_vis > 0.5:
                visible_hands += 1

        if frame.right_hand_landmarks and len(frame.right_hand_landmarks) > 0:
            right_vis = sum(lm.visibility for lm in frame.right_hand_landmarks) / len(frame.right_hand_landmarks)
            if right_vis > 0.5:
                visible_hands += 1

        # Context-aware hand visibility scoring
        if sign_type_hands == 'one-handed':
            # For one-handed signs, at least ONE hand should be visible
            # Score based on best hand visibility
            hand_visibility_scores.append(max(left_vis, right_vis))
        else:
            # For two-handed signs (or unknown), expect both hands
            # Default behavior: normalize to 0-1 based on 2 hands
            hand_visibility_scores.append(visible_hands / 2.0)

    hand_visibility = sum(hand_visibility_scores) / len(hand_visibility_scores) if hand_visibility_scores else 0.0

    # 2. Motion Smoothness Score (30%)
    # Context-aware: For static signs, penalize movement. For dynamic signs, reward smooth motion.
    smoothness_scores = []
    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        # Compare wrist positions (landmark index 15 and 16 for left/right wrists)
        if prev_frame.pose_landmarks and curr_frame.pose_landmarks:
            left_wrist_prev = prev_frame.pose_landmarks[15]
            left_wrist_curr = curr_frame.pose_landmarks[15]

            # Calculate Euclidean distance
            distance = (
                (left_wrist_curr.x - left_wrist_prev.x) ** 2 +
                (left_wrist_curr.y - left_wrist_prev.y) ** 2 +
                (left_wrist_curr.z - left_wrist_prev.z) ** 2
            ) ** 0.5

            if sign_type_movement == 'static':
                # For static signs: Reward minimal movement (stability)
                # Small movements are good, large movements are bad
                smoothness = max(0.0, 1.0 - (distance / 0.05))  # Tighter threshold for static
            else:
                # For dynamic signs: Penalize large jumps but allow movement
                # Default behavior: smooth motion is good, jitter is bad
                smoothness = max(0.0, 1.0 - (distance / 0.1))

            smoothness_scores.append(smoothness)

    smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 1.0

    # 3. Frame Completeness Score (20%)
    completeness_scores = []
    for frame in frames:
        has_pose = frame.pose_landmarks is not None and len(frame.pose_landmarks) == 33
        has_left_hand = frame.left_hand_landmarks is not None and len(frame.left_hand_landmarks) == 21
        has_right_hand = frame.right_hand_landmarks is not None and len(frame.right_hand_landmarks) == 21

        completeness = (int(has_pose) + int(has_left_hand) + int(has_right_hand)) / 3.0
        completeness_scores.append(completeness)

    completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

    # Weighted average
    quality_score = (
        hand_visibility * 0.50 +
        smoothness * 0.30 +
        completeness * 0.20
    )

    return round(quality_score, 3)


def calculate_quality_breakdown(
    frames: List[Frame],
    sign_type_movement: str = None,
    sign_type_hands: str = None
) -> Dict[str, Any]:
    """
    Calculate detailed quality breakdown with component scores and recommendations.

    Args:
        frames: List of landmark frames
        sign_type_movement: 'static' or 'dynamic'
        sign_type_hands: 'one-handed' or 'two-handed'

    Returns:
        Dictionary with detailed quality metrics and actionable recommendations
    """
    if not frames:
        return {
            "overall_score": 0.0,
            "hand_visibility": 0.0,
            "motion_smoothness": 0.0,
            "frame_completeness": 0.0,
            "lighting_quality": 0.0,
            "components": {},
            "recommendations": ["No frames detected"]
        }

    # Calculate individual components
    hand_visibility_scores = []
    for frame in frames:
        visible_hands = 0
        left_vis = 0.0
        right_vis = 0.0

        if frame.left_hand_landmarks and len(frame.left_hand_landmarks) > 0:
            left_vis = sum(lm.visibility for lm in frame.left_hand_landmarks) / len(frame.left_hand_landmarks)
            if left_vis > 0.5:
                visible_hands += 1

        if frame.right_hand_landmarks and len(frame.right_hand_landmarks) > 0:
            right_vis = sum(lm.visibility for lm in frame.right_hand_landmarks) / len(frame.right_hand_landmarks)
            if right_vis > 0.5:
                visible_hands += 1

        if sign_type_hands == 'one-handed':
            hand_visibility_scores.append(max(left_vis, right_vis))
        else:
            hand_visibility_scores.append(visible_hands / 2.0)

    hand_visibility = sum(hand_visibility_scores) / len(hand_visibility_scores) if hand_visibility_scores else 0.0

    # Motion smoothness
    smoothness_scores = []
    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        if prev_frame.pose_landmarks and curr_frame.pose_landmarks:
            left_wrist_prev = prev_frame.pose_landmarks[15]
            left_wrist_curr = curr_frame.pose_landmarks[15]

            distance = (
                (left_wrist_curr.x - left_wrist_prev.x) ** 2 +
                (left_wrist_curr.y - left_wrist_prev.y) ** 2 +
                (left_wrist_curr.z - left_wrist_prev.z) ** 2
            ) ** 0.5

            if sign_type_movement == 'static':
                smoothness = max(0.0, 1.0 - (distance / 0.05))
            else:
                smoothness = max(0.0, 1.0 - (distance / 0.1))

            smoothness_scores.append(smoothness)

    motion_smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 1.0

    # Frame completeness
    completeness_scores = []
    for frame in frames:
        has_pose = frame.pose_landmarks is not None and len(frame.pose_landmarks) == 33
        has_left_hand = frame.left_hand_landmarks is not None and len(frame.left_hand_landmarks) == 21
        has_right_hand = frame.right_hand_landmarks is not None and len(frame.right_hand_landmarks) == 21

        completeness = (int(has_pose) + int(has_left_hand) + int(has_right_hand)) / 3.0
        completeness_scores.append(completeness)

    frame_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

    # Lighting quality
    lighting_quality = calculate_lighting_quality(frames)

    # Overall score
    overall_score = (
        hand_visibility * 0.50 +
        motion_smoothness * 0.30 +
        frame_completeness * 0.20
    )

    # Generate human-readable labels
    def get_label(score: float) -> str:
        if score >= 0.85:
            return "Excellent ✓"
        elif score >= 0.70:
            return "Good ✓"
        elif score >= 0.55:
            return "Acceptable ⚠️"
        elif score >= 0.40:
            return "Poor ⚠️"
        else:
            return "Very Poor ✗"

    def get_lighting_label(score: float) -> str:
        if score >= 0.85:
            return "Excellent (☀️)"
        elif score >= 0.70:
            return "Good (💡)"
        elif score >= 0.55:
            return "Acceptable (⚠️)"
        elif score >= 0.40:
            return "Poor (🌙)"
        else:
            return "Very Poor (🌑)"

    components = {
        "overall": get_label(overall_score),
        "hand_visibility": get_label(hand_visibility),
        "motion_smoothness": get_label(motion_smoothness),
        "frame_completeness": get_label(frame_completeness),
        "lighting": get_lighting_label(lighting_quality)
    }

    # Generate recommendations
    recommendations = []
    if hand_visibility < 0.5:
        recommendations.append("Keep both hands clearly visible in frame throughout recording")
    elif hand_visibility < 0.7:
        recommendations.append("Ensure hands remain in frame during the entire sign")

    if lighting_quality < 0.5:
        recommendations.append("Move to better lighting or turn on more lights")
    elif lighting_quality < 0.7:
        recommendations.append("Consider improving lighting for better quality")

    if motion_smoothness < 0.5:
        recommendations.append("Perform sign more smoothly - avoid jerky movements")
    elif motion_smoothness < 0.7:
        recommendations.append("Try to move more fluidly during the sign")

    if frame_completeness < 0.5:
        recommendations.append("Position camera to show full upper body (waist up)")
    elif frame_completeness < 0.7:
        recommendations.append("Ensure full upper body is visible throughout")

    if overall_score >= 0.7 and not recommendations:
        recommendations.append("Great job! Your recording quality is excellent")

    return {
        "overall_score": round(overall_score, 3),
        "hand_visibility": round(hand_visibility, 3),
        "motion_smoothness": round(motion_smoothness, 3),
        "frame_completeness": round(frame_completeness, 3),
        "lighting_quality": round(lighting_quality, 3),
        "components": components,
        "recommendations": recommendations
    }


# ===========================
# Storage Functions
# ===========================

def save_contribution(contribution: Contribution, contribution_id: str, quality_score: float) -> str:
    """
    Save contribution to DATABASE ONLY (no more JSON files).

    Args:
        contribution: Contribution data
        contribution_id: Unique contribution ID
        quality_score: Calculated quality score

    Returns:
        Contribution ID

    Raises:
        HTTPException: If quality score is below minimum threshold (60%)
    """
    # Quality threshold check - reject contributions below 60%
    MIN_QUALITY_THRESHOLD = 0.60
    if quality_score < MIN_QUALITY_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Quality too low",
                "message": f"Contribution quality ({quality_score:.1%}) is below minimum acceptable threshold ({MIN_QUALITY_THRESHOLD:.0%}). Please try again with better lighting, hand positioning, and steadier movement.",
                "quality_score": round(quality_score, 3),
                "minimum_required": MIN_QUALITY_THRESHOLD,
                "tips": [
                    "Ensure good lighting on your hands",
                    "Keep hands clearly visible in frame",
                    "Make smooth, deliberate movements",
                    "Avoid quick or jittery motions",
                    "Use a clear background"
                ]
            }
        )

    # Prepare frames data for storage
    frames_data = [frame.dict() for frame in contribution.frames]

    # Save to database
    db = SessionLocal()

    try:
        # Insert contribution record
        db_contribution = DBContribution(
            contribution_id=contribution_id,
            word=contribution.word.upper(),
            user_id=contribution.user_id,
            num_frames=len(contribution.frames),
            duration=contribution.duration,
            quality_score=quality_score,
            file_path=None,  # No file path since we're not saving JSON files
            frames_data=frames_data,  # Store frames in JSON column

            # Sign classification (from user input)
            sign_type_movement=contribution.sign_type_movement,
            sign_type_hands=contribution.sign_type_hands,

            # 3-attempt metadata
            num_attempts=contribution.num_attempts or 1,
            individual_qualities=contribution.individual_qualities,
            individual_durations=contribution.individual_durations,
            quality_variance=contribution.quality_variance,
            improvement_trend=contribution.improvement_trend
        )
        db.add(db_contribution)

        # Update or create Word record
        word_upper = contribution.word.upper()
        word_record = db.query(Word).filter(Word.word == word_upper).first()

        if word_record:
            # Increment contribution count
            word_record.contributions_count += 1

            # Recalculate average quality score (need to flush to include the new contribution)
            db.flush()  # Make the new contribution visible to the query
            all_contributions = db.query(DBContribution).filter(
                DBContribution.word == word_upper
            ).all()
            if all_contributions:
                avg_quality = sum(c.quality_score for c in all_contributions) / len(all_contributions)
                word_record.quality_score = round(avg_quality, 3)
            else:
                word_record.quality_score = quality_score

            # Update sign classification consensus (if user provided classification)
            if contribution.sign_type_movement:
                # Count votes for static vs dynamic
                static_count = sum(1 for c in all_contributions
                                  if c.sign_type_movement == 'static')
                dynamic_count = sum(1 for c in all_contributions
                                   if c.sign_type_movement == 'dynamic')

                word_record.static_votes = static_count
                word_record.dynamic_votes = dynamic_count

                total_votes = static_count + dynamic_count
                if total_votes >= 10:  # Need at least 10 votes for consensus
                    majority = max(static_count, dynamic_count)
                    confidence = majority / total_votes

                    if confidence >= 0.7:  # 70% agreement = consensus
                        word_record.sign_type_consensus = 'static' if static_count > dynamic_count else 'dynamic'
                        word_record.consensus_confidence = round(confidence, 3)
                    else:
                        word_record.sign_type_consensus = 'unknown'
                        word_record.consensus_confidence = round(confidence, 3)

            # Check if complete (50 contributions needed)
            if word_record.contributions_count >= word_record.contributions_needed:
                word_record.is_complete = True
                # Automatically close word when it reaches 50 contributions
                word_record.is_open_for_contribution = False
                print(f"✓ Auto-closed {word_upper} - reached {word_record.contributions_count} contributions")
        else:
            # Create new word record if it doesn't exist
            word_record = Word(
                word=word_upper,
                contributions_count=1,
                contributions_needed=50,
                is_complete=False,
                quality_score=quality_score
            )
            db.add(word_record)

        db.commit()
        print(f"✓ Saved contribution {contribution_id} for {word_upper} to database")
        return contribution_id

    except Exception as e:
        db.rollback()
        print(f"✗ Failed to save to database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save contribution to database: {str(e)}"
        )
    finally:
        db.close()


def load_contributions(word: str) -> List[Dict[str, Any]]:
    """
    Load all contributions for a specific word.

    Args:
        word: GSL word (case-insensitive)

    Returns:
        List of contribution dictionaries
    """
    word_dir = CONTRIBUTIONS_DIR / word.upper()
    if not word_dir.exists():
        return []

    contributions = []
    for filepath in word_dir.glob("*.json"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                contributions.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    return contributions


def count_contributions(word: str) -> Dict[str, Any]:
    """
    Count contributions and calculate statistics for a word from DATABASE.

    Args:
        word: GSL word (case-insensitive)

    Returns:
        Dictionary with statistics
    """
    try:
        db = SessionLocal()
        word_upper = word.upper()

        # Get all contributions for this word from database
        contributions = db.query(DBContribution).filter(
            DBContribution.word == word_upper
        ).all()

        if not contributions:
            db.close()
            return {
                "total_contributions": 0,
                "unique_contributors": 0,
                "average_quality": 0.0,
                "ready_for_training": False
            }

        unique_users = set(c.user_id for c in contributions if c.user_id)
        quality_scores = [c.quality_score for c in contributions]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        result = {
            "total_contributions": len(contributions),
            "unique_contributors": len(unique_users),
            "average_quality": round(avg_quality, 3),
            "ready_for_training": len(contributions) >= 10 and avg_quality >= 0.6
        }

        db.close()
        return result

    except Exception as e:
        print(f"Error counting contributions from database: {e}")
        return {
            "total_contributions": 0,
            "unique_contributors": 0,
            "average_quality": 0.0,
            "ready_for_training": False
        }


# ===========================
# API Endpoints
# ===========================

class EnvironmentCheckRequest(BaseModel):
    """Request model for environment quality check."""
    frames: List[Frame] = Field(..., min_items=10, max_items=30, description="10-30 sample frames for environment check")


class EnvironmentCheckResponse(BaseModel):
    """Response model for environment quality check."""
    lighting_quality: float
    hand_visibility: float
    frame_completeness: float
    can_proceed: bool
    lighting_label: str
    recommendations: List[str]


@router.post("/check-environment", response_model=EnvironmentCheckResponse)
async def check_recording_environment(request: EnvironmentCheckRequest):
    """
    Check recording environment quality BEFORE starting actual recording.

    Analyzes a short sample of frames (10-30) to determine:
    - Lighting quality
    - Hand visibility
    - Frame completeness

    Blocks recording if conditions are too poor (lighting < 25%).
    Returns recommendations for improvement.

    Args:
        request: Environment check request with sample frames

    Returns:
        EnvironmentCheckResponse with quality metrics and recommendations
    """
    frames = request.frames

    # Calculate lighting quality
    lighting_quality = calculate_lighting_quality(frames)

    # Calculate hand visibility
    hand_visibility_scores = []
    for frame in frames:
        left_vis = 0.0
        right_vis = 0.0

        if frame.left_hand_landmarks and len(frame.left_hand_landmarks) > 0:
            left_vis = sum(lm.visibility for lm in frame.left_hand_landmarks) / len(frame.left_hand_landmarks)

        if frame.right_hand_landmarks and len(frame.right_hand_landmarks) > 0:
            right_vis = sum(lm.visibility for lm in frame.right_hand_landmarks) / len(frame.right_hand_landmarks)

        # Use best hand visibility
        hand_visibility_scores.append(max(left_vis, right_vis))

    hand_visibility = sum(hand_visibility_scores) / len(hand_visibility_scores) if hand_visibility_scores else 0.0

    # Calculate frame completeness
    completeness_scores = []
    for frame in frames:
        has_pose = frame.pose_landmarks is not None and len(frame.pose_landmarks) == 33
        has_left_hand = frame.left_hand_landmarks is not None and len(frame.left_hand_landmarks) == 21
        has_right_hand = frame.right_hand_landmarks is not None and len(frame.right_hand_landmarks) == 21

        completeness = (int(has_pose) + int(has_left_hand) + int(has_right_hand)) / 3.0
        completeness_scores.append(completeness)

    frame_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

    # Determine if can proceed (lighting must be at least 25%)
    can_proceed = lighting_quality >= 0.25

    # Generate lighting label
    def get_lighting_label(score: float) -> str:
        if score >= 0.85:
            return "Excellent (☀️)"
        elif score >= 0.70:
            return "Good (💡)"
        elif score >= 0.55:
            return "Acceptable (⚠️)"
        elif score >= 0.40:
            return "Poor (🌙)"
        elif score >= 0.25:
            return "Very Poor (🌑)"
        else:
            return "Too Dark (✗)"

    lighting_label = get_lighting_label(lighting_quality)

    # Generate recommendations
    recommendations = []
    if lighting_quality < 0.25:
        recommendations.append("⚠️ Lighting: Too Dark - Cannot start recording")
        recommendations.append("Move to better lighting or turn on more lights")
    elif lighting_quality < 0.55:
        recommendations.append("Lighting could be improved - consider moving to better lighting")

    if hand_visibility < 0.5:
        recommendations.append("Ensure both hands are clearly visible in frame")

    if frame_completeness < 0.5:
        recommendations.append("Position camera to show full upper body (waist up)")

    if can_proceed and not recommendations:
        recommendations.append("✓ Environment looks good! Ready to record")

    return EnvironmentCheckResponse(
        lighting_quality=round(lighting_quality, 3),
        hand_visibility=round(hand_visibility, 3),
        frame_completeness=round(frame_completeness, 3),
        can_proceed=can_proceed,
        lighting_label=lighting_label,
        recommendations=recommendations
    )


@router.post("/contribute", response_model=ContributionResponse, status_code=status.HTTP_201_CREATED)
async def receive_contribution(contribution: Contribution):
    """
    Receive and validate a sign language contribution.

    This endpoint:
    1. Validates landmark data structure
    2. Calculates quality score with detailed breakdown
    3. Rejects low-quality contributions (< 0.5)
    4. Saves to database
    5. Returns contribution status with quality feedback

    Args:
        contribution: Contribution data with landmark frames

    Returns:
        ContributionResponse with contribution ID, statistics, and quality breakdown
    """
    # Calculate quality score (context-aware based on user classification)
    quality_score = calculate_quality_score(
        contribution.frames,
        sign_type_movement=contribution.sign_type_movement,
        sign_type_hands=contribution.sign_type_hands
    )

    # Calculate detailed quality breakdown
    breakdown_data = calculate_quality_breakdown(
        contribution.frames,
        sign_type_movement=contribution.sign_type_movement,
        sign_type_hands=contribution.sign_type_hands
    )

    # Reject low-quality contributions
    if quality_score < 0.5:
        # Include breakdown in error response
        detail_message = f"Contribution quality too low: {quality_score:.2f} (minimum 50% required).\n\n"
        detail_message += "Quality Breakdown:\n"
        detail_message += f"- Hand Visibility: {breakdown_data['hand_visibility']*100:.0f}% ({breakdown_data['components']['hand_visibility']})\n"
        detail_message += f"- Motion Smoothness: {breakdown_data['motion_smoothness']*100:.0f}% ({breakdown_data['components']['motion_smoothness']})\n"
        detail_message += f"- Frame Completeness: {breakdown_data['frame_completeness']*100:.0f}% ({breakdown_data['components']['frame_completeness']})\n"
        detail_message += f"- Lighting: {breakdown_data['lighting_quality']*100:.0f}% ({breakdown_data['components']['lighting']})\n\n"
        detail_message += "Recommendations:\n"
        for rec in breakdown_data['recommendations']:
            detail_message += f"• {rec}\n"

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail_message
        )

    # Check if word is open for contribution
    from database import get_db, Word
    db = next(get_db())
    word_upper = contribution.word.upper()
    word_record = db.query(Word).filter(Word.word == word_upper).first()

    if not word_record:
        db.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Word '{word_upper}' not found in the dictionary. Please select a valid sign."
        )

    if not word_record.is_open_for_contribution:
        db.close()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This sign is currently closed for contributions. Please choose a different sign that is open for data collection."
        )

    db.close()

    # Check if word has already reached the contribution limit
    stats = count_contributions(contribution.word)
    target_contributions = 50

    if stats["total_contributions"] >= target_contributions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This sign has already reached the maximum of {target_contributions} contributions. Please choose a different sign to contribute to."
        )

    # Generate unique contribution ID
    contribution_id = str(uuid.uuid4())[:8]

    # Save contribution
    try:
        filepath = save_contribution(contribution, contribution_id, quality_score)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save contribution: {str(e)}"
        )

    # Get updated statistics (after saving)
    stats = count_contributions(contribution.word)

    # Calculate progress (target: 50 contributions per sign)
    progress = min(100.0, (stats["total_contributions"] / target_contributions) * 100)

    # Create quality breakdown response
    quality_breakdown = QualityBreakdown(
        overall_score=breakdown_data["overall_score"],
        hand_visibility=breakdown_data["hand_visibility"],
        motion_smoothness=breakdown_data["motion_smoothness"],
        frame_completeness=breakdown_data["frame_completeness"],
        lighting_quality=breakdown_data["lighting_quality"],
        components=breakdown_data["components"],
        recommendations=breakdown_data["recommendations"]
    )

    return ContributionResponse(
        contribution_id=contribution_id,
        word=contribution.word,
        quality_score=quality_score,
        quality_breakdown=quality_breakdown,
        total_contributions=stats["total_contributions"],
        progress_percentage=round(progress, 1),
        message=f"Thank you! Your contribution has been saved with quality score {quality_score:.2f}"
    )


@router.get("/contribution-stats/{word}", response_model=ContributionStats)
async def get_contribution_stats(word: str):
    """
    Get contribution statistics for a specific word.

    Args:
        word: GSL word (case-insensitive)

    Returns:
        ContributionStats with detailed statistics
    """
    word = word.upper().strip()
    stats = count_contributions(word)

    target_contributions = 10
    contributions_needed = max(0, target_contributions - stats["total_contributions"])

    return ContributionStats(
        word=word,
        total_contributions=stats["total_contributions"],
        unique_contributors=stats["unique_contributors"],
        average_quality=stats["average_quality"],
        ready_for_training=stats["ready_for_training"],
        contributions_needed=contributions_needed
    )


class WordClassificationResponse(BaseModel):
    """Response with word's consensus classification data."""
    word: str
    sign_type_consensus: Optional[str] = None  # 'static', 'dynamic', or 'unknown'
    consensus_confidence: Optional[float] = None
    static_votes: int = 0
    dynamic_votes: int = 0
    total_votes: int = 0
    has_consensus: bool = False


@router.get("/words/{word}/classification", response_model=WordClassificationResponse)
async def get_word_classification(word: str):
    """
    Get the crowdsourced classification consensus for a word.

    This endpoint retrieves existing classification data from the database,
    which can be used to pre-fill classification forms or display consensus.

    Args:
        word: GSL word (case-insensitive)

    Returns:
        WordClassificationResponse with consensus data
    """
    db = SessionLocal()
    try:
        word_upper = word.upper().strip()
        word_record = db.query(Word).filter(Word.word == word_upper).first()

        if not word_record:
            # Word doesn't exist yet, return empty classification
            return WordClassificationResponse(
                word=word_upper,
                sign_type_consensus=None,
                consensus_confidence=None,
                static_votes=0,
                dynamic_votes=0,
                total_votes=0,
                has_consensus=False
            )

        total_votes = word_record.static_votes + word_record.dynamic_votes
        has_consensus = (
            word_record.sign_type_consensus is not None
            and word_record.sign_type_consensus != 'unknown'
            and word_record.consensus_confidence is not None
            and word_record.consensus_confidence >= 0.7
        )

        return WordClassificationResponse(
            word=word_upper,
            sign_type_consensus=word_record.sign_type_consensus,
            consensus_confidence=word_record.consensus_confidence,
            static_votes=word_record.static_votes,
            dynamic_votes=word_record.dynamic_votes,
            total_votes=total_votes,
            has_consensus=has_consensus
        )
    finally:
        db.close()


@router.get("/contribution-leaderboard")
async def get_contribution_leaderboard(limit: int = 10):
    """
    Get leaderboard of words by contribution count.

    Args:
        limit: Maximum number of results to return

    Returns:
        List of words with contribution statistics, sorted by count
    """
    # Scan all word directories
    leaderboard = []

    if not CONTRIBUTIONS_DIR.exists():
        return []

    for word_dir in CONTRIBUTIONS_DIR.iterdir():
        if word_dir.is_dir():
            word = word_dir.name
            stats = count_contributions(word)

            if stats["total_contributions"] > 0:
                leaderboard.append({
                    "word": word,
                    "total_contributions": stats["total_contributions"],
                    "unique_contributors": stats["unique_contributors"],
                    "average_quality": stats["average_quality"],
                    "ready_for_training": stats["ready_for_training"]
                })

    # Sort by contribution count (descending)
    leaderboard.sort(key=lambda x: x["total_contributions"], reverse=True)

    return leaderboard[:limit]


# ===========================
# Health Check
# ===========================

@router.get("/contribution-health")
async def contribution_health():
    """Health check endpoint for contribution system."""
    return {
        "status": "healthy",
        "contributions_dir": str(CONTRIBUTIONS_DIR),
        "total_words": len(list(CONTRIBUTIONS_DIR.iterdir())) if CONTRIBUTIONS_DIR.exists() else 0
    }


@router.get("/contributions/{word}")
async def get_contributions_for_word(word: str, limit: int = 10):
    """
    Get all contributions for a specific word from the database.
    
    Args:
        word: The sign word
        limit: Maximum number of contributions to return (default: 10)
    
    Returns:
        List of contributions with frames data for skeleton preview
    """
    word = word.upper().strip()
    
    try:
        db = SessionLocal()
        contributions = db.query(DBContribution).filter(
            DBContribution.word == word
        ).order_by(
            DBContribution.quality_score.desc(),  # Best quality first
            DBContribution.created_at.desc()      # Most recent first
        ).limit(limit).all()
        
        result = []
        for contrib in contributions:
            result.append({
                "contribution_id": contrib.contribution_id,
                "user_id": contrib.user_id,
                "duration": contrib.duration,
                "quality_score": contrib.quality_score,
                "num_frames": contrib.num_frames,
                "frames": contrib.frames_data,  # Include frames for preview
                "created_at": contrib.created_at.isoformat()
            })
        
        db.close()
        return {"word": word, "contributions": result, "total": len(result)}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch contributions: {str(e)}"
        )


@router.get("/contributions/{word}/average")
async def get_average_skeleton(word: str):
    """
    Calculate the average skeleton from all contributions for a word.
    This represents the "ground truth" or community consensus.
    
    Args:
        word: The sign word
    
    Returns:
        Averaged frames representing the ideal sign
    """
    word = word.upper().strip()
    
    try:
        db = SessionLocal()
        contributions = db.query(DBContribution).filter(
            DBContribution.word == word,
            DBContribution.frames_data != None
        ).all()
        
        if not contributions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No contributions found for word: {word}"
            )
        
        # Calculate average frames
        # 1. Normalize all contributions to same frame count
        # 2. Average each landmark position across all contributions
        
        # Find the median frame count to use as target
        frame_counts = [c.num_frames for c in contributions]
        target_frames = sorted(frame_counts)[len(frame_counts) // 2]
        
        averaged_frames = []
        
        for frame_idx in range(target_frames):
            # Collect landmarks from all contributions for this frame index
            frame_landmarks = {
                "pose_landmarks": [],
                "left_hand_landmarks": [],
                "right_hand_landmarks": []
            }
            
            for contrib in contributions:
                frames_data = contrib.frames_data
                if not frames_data or frame_idx >= len(frames_data):
                    continue
                
                frame = frames_data[frame_idx]
                if "pose_landmarks" in frame:
                    frame_landmarks["pose_landmarks"].append(frame["pose_landmarks"])
                if "left_hand_landmarks" in frame and frame["left_hand_landmarks"]:
                    frame_landmarks["left_hand_landmarks"].append(frame["left_hand_landmarks"])
                if "right_hand_landmarks" in frame and frame["right_hand_landmarks"]:
                    frame_landmarks["right_hand_landmarks"].append(frame["right_hand_landmarks"])
            
            # Average the landmarks
            avg_frame = {
                "frame_number": frame_idx,
                "timestamp": frame_idx / 30.0,  # Assuming 30fps
                "pose_landmarks": average_landmarks(frame_landmarks["pose_landmarks"], 33),
                "left_hand_landmarks": average_landmarks(frame_landmarks["left_hand_landmarks"], 21) if frame_landmarks["left_hand_landmarks"] else None,
                "right_hand_landmarks": average_landmarks(frame_landmarks["right_hand_landmarks"], 21) if frame_landmarks["right_hand_landmarks"] else None,
                "face_landmarks": None
            }
            
            averaged_frames.append(avg_frame)
        
        db.close()
        
        return {
            "word": word,
            "num_contributions": len(contributions),
            "avg_quality_score": sum(c.quality_score for c in contributions) / len(contributions),
            "frames": averaged_frames
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate average: {str(e)}"
        )


def average_landmarks(landmarks_list, expected_count):
    """
    Average a list of landmark arrays.
    
    Args:
        landmarks_list: List of landmark arrays from different contributions
        expected_count: Expected number of landmarks (e.g., 33 for pose, 21 for hands)
    
    Returns:
        Averaged landmark array
    """
    if not landmarks_list:
        return None
    
    # Initialize averages
    averaged = []
    
    for idx in range(expected_count):
        x_sum, y_sum, z_sum, vis_sum, count = 0, 0, 0, 0, 0
        
        for landmarks in landmarks_list:
            if idx < len(landmarks) and landmarks[idx]:
                lm = landmarks[idx]
                x_sum += lm.get("x", 0)
                y_sum += lm.get("y", 0)
                z_sum += lm.get("z", 0)
                vis_sum += lm.get("visibility", 1.0)
                count += 1
        
        if count > 0:
            averaged.append({
                "x": x_sum / count,
                "y": y_sum / count,
                "z": z_sum / count,
                "visibility": vis_sum / count
            })
        else:
            # No data for this landmark
            averaged.append({
                "x": 0, "y": 0, "z": 0, "visibility": 0
            })
    
    return averaged
