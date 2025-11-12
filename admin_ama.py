#!/usr/bin/env python3
"""
Admin AMA (Ask Me Anything) API - Comprehensive Admin Dashboard Endpoint
Provides bird's eye view and full CRUD operations for SignForge platform.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("admin_ama")

try:
    from database import get_db, Word, Contribution
    DB_AVAILABLE = True
    logger.info("✓ Database modules imported successfully")
except ImportError as e:
    DB_AVAILABLE = False
    logger.warning(f"⚠ Database import failed: {e}")
    # Define dummy dependencies
    def get_db():
        raise HTTPException(status_code=503, detail="Database not available")
    Word = Contribution = None

router = APIRouter(prefix="/api/ama", tags=["admin"])


# ===========================
# Pydantic Models
# ===========================

class SystemOverview(BaseModel):
    """Complete system overview"""
    total_words: int
    total_contributions: int
    unique_contributors: int
    average_quality_score: float
    words_ready_for_training: int
    total_frames_collected: int
    total_data_points: int
    storage_size_mb: float
    uptime_hours: float
    last_contribution_time: Optional[datetime]
    system_health: str


class WordStats(BaseModel):
    """Detailed statistics for a word"""
    word: str
    total_contributions: int
    unique_contributors: int
    average_quality: float
    min_quality: float
    max_quality: float
    average_duration: float
    average_fps: float
    total_frames: int
    ready_for_training: bool
    contributions_needed: int
    last_contribution: Optional[datetime]
    is_open_for_contribution: Optional[bool] = False


class ContributionDetail(BaseModel):
    """Detailed contribution information"""
    id: int
    word: str
    user_id: str
    duration: float
    quality_score: float
    num_frames: int
    fps: float
    data_points: int
    has_left_hand: bool
    has_right_hand: bool
    created_at: datetime
    metadata: Optional[Dict[str, Any]]


class ContributionFullDetail(BaseModel):
    """Full contribution information with pose sequence data"""
    id: int
    word: str
    user_id: str
    duration: float
    quality_score: float
    num_frames: int
    fps: float
    data_points: int
    has_left_hand: bool
    has_right_hand: bool
    created_at: datetime
    metadata: Optional[Dict[str, Any]]
    pose_sequence: List[Dict[str, Any]]  # frames_data format: list of frame objects
    quality_breakdown: Optional[Dict[str, Any]]


class ContributionUpdateRequest(BaseModel):
    """Request model for updating contribution"""
    word: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DataQualityReport(BaseModel):
    """Data quality analysis report"""
    total_contributions: int
    high_quality_count: int  # >= 0.8
    medium_quality_count: int  # 0.6-0.8
    low_quality_count: int  # < 0.6
    average_quality: float
    quality_distribution: Dict[str, int]
    fps_distribution: Dict[str, int]
    hand_detection_rate: float
    average_frames_per_contribution: float
    data_completeness_score: float


class UserActivity(BaseModel):
    """User activity statistics"""
    user_id: str
    total_contributions: int
    words_contributed: List[str]
    average_quality: float
    total_frames: int
    first_contribution: datetime
    last_contribution: datetime
    consistency_score: float


# ===========================
# Helper Functions
# ===========================

def calculate_fps(contribution) -> float:
    """Calculate FPS from contribution frames"""
    try:
        frames = json.loads(contribution.frames_data) if isinstance(contribution.frames_data, str) else contribution.frames_data
        if not frames or len(frames) < 2:
            return 0.0

        total_duration = frames[-1]['timestamp'] - frames[0]['timestamp']
        return len(frames) / total_duration if total_duration > 0 else 0.0
    except:
        return 0.0


def calculate_data_points(contribution) -> int:
    """Calculate total data points in contribution"""
    try:
        frames = json.loads(contribution.frames_data) if isinstance(contribution.frames_data, str) else contribution.frames_data
        if not frames:
            return 0

        # Assume 300 data points per frame (75 landmarks × 4 values)
        return len(frames) * 300
    except:
        return 0


def check_hand_presence(contribution) -> tuple:
    """Check if left/right hands are present"""
    try:
        frames = json.loads(contribution.frames_data) if isinstance(contribution.frames_data, str) else contribution.frames_data
        if not frames:
            return False, False

        has_left = any(f.get('left_hand_landmarks') is not None for f in frames)
        has_right = any(f.get('right_hand_landmarks') is not None for f in frames)
        return has_left, has_right
    except:
        return False, False


# ===========================
# Admin Endpoints
# ===========================

@router.get("/overview", response_model=SystemOverview)
async def get_system_overview(request: Request, db: Session = Depends(get_db)):
    """
    🎯 BIRD'S EYE VIEW - Complete system overview

    Returns comprehensive statistics about the entire platform:
    - Total words, contributions, corrections
    - Quality metrics
    - Data collection progress
    - System health status
    """
    logger.info(f"📊 Admin Overview requested from {request.client.host}")

    if not DB_AVAILABLE:
        logger.error("Database not available for overview request")
        raise HTTPException(status_code=503, detail="Database not available")

    # Total counts
    total_words = db.query(func.count(Word.id)).scalar() or 0
    total_contributions = db.query(func.count(Contribution.id)).scalar() or 0

    # Unique contributors
    unique_contributors = db.query(func.count(func.distinct(Contribution.user_id))).scalar() or 0

    # Quality metrics
    avg_quality = db.query(func.avg(Contribution.quality_score)).scalar() or 0.0

    # Words ready for training (>= 10 contributions with avg quality >= 0.6)
    words_ready = db.query(func.count(func.distinct(Contribution.word)))\
        .group_by(Contribution.word)\
        .having(func.count(Contribution.id) >= 10)\
        .having(func.avg(Contribution.quality_score) >= 0.6)\
        .count()

    # Total frames collected
    total_frames = db.query(func.sum(Contribution.num_frames)).scalar() or 0

    # Total data points (assuming 300 per frame)
    total_data_points = total_frames * 300

    # Storage estimate (rough calculation)
    # Each contribution ~50KB average
    storage_size_mb = (total_contributions * 50) / 1024

    # Last contribution time
    last_contribution = db.query(Contribution.created_at)\
        .order_by(desc(Contribution.created_at))\
        .first()

    last_contrib_time = last_contribution[0] if last_contribution else None

    # Calculate uptime (time since first contribution)
    first_contribution = db.query(Contribution.created_at)\
        .order_by(Contribution.created_at)\
        .first()

    uptime_hours = 0.0
    if first_contribution and first_contribution[0]:
        uptime_delta = datetime.utcnow() - first_contribution[0]
        uptime_hours = uptime_delta.total_seconds() / 3600

    # System health determination
    if avg_quality >= 0.7 and total_contributions > 50:
        system_health = "EXCELLENT"
    elif avg_quality >= 0.6 and total_contributions > 20:
        system_health = "GOOD"
    elif total_contributions > 0:
        system_health = "FAIR"
    else:
        system_health = "NEW"

    return SystemOverview(
        total_words=total_words,
        total_contributions=total_contributions,
        unique_contributors=unique_contributors,
        average_quality_score=round(float(avg_quality), 3),
        words_ready_for_training=words_ready,
        total_frames_collected=total_frames,
        total_data_points=total_data_points,
        storage_size_mb=round(storage_size_mb, 2),
        uptime_hours=round(uptime_hours, 2),
        last_contribution_time=last_contrib_time,
        system_health=system_health
    )


@router.get("/words", response_model=List[WordStats])
async def get_all_words_stats(
    min_quality: float = Query(0.0, ge=0.0, le=1.0),
    ready_only: bool = Query(False),
    limit: int = Query(100, le=2000),
    db: Session = Depends(get_db)
):
    """
    📊 WORD ANALYTICS - Detailed statistics for all words

    Query params:
    - min_quality: Filter words with average quality >= this value
    - ready_only: Show only words ready for training
    - limit: Maximum number of words to return
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get all words from the words table
    all_words = db.query(Word).all()

    stats_list = []
    for word_record in all_words:
        # Get contributions for this word
        contributions = db.query(Contribution).filter(Contribution.word == word_record.word).all()

        if contributions:
            avg_quality = sum(c.quality_score for c in contributions) / len(contributions)

            # Apply filters
            if avg_quality < min_quality:
                continue

            ready = len(contributions) >= 50 and avg_quality >= 0.6
            if ready_only and not ready:
                continue

            # Calculate detailed stats
            unique_users = len(set(c.user_id for c in contributions))
            qualities = [c.quality_score for c in contributions]
            durations = [c.duration for c in contributions]
            total_frames = sum(c.num_frames for c in contributions)

            # Calculate average FPS
            fps_values = [calculate_fps(c) for c in contributions]
            avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0

            last_contribution = max(c.created_at for c in contributions)

            stats_list.append(WordStats(
                word=word_record.word,
                total_contributions=len(contributions),
                unique_contributors=unique_users,
                average_quality=round(avg_quality, 3),
                min_quality=round(min(qualities), 3),
                max_quality=round(max(qualities), 3),
                average_duration=round(sum(durations) / len(durations), 2),
                average_fps=round(avg_fps, 2),
                total_frames=total_frames,
                ready_for_training=ready,
                contributions_needed=max(0, 50 - len(contributions)),
                last_contribution=last_contribution,
                is_open_for_contribution=word_record.is_open_for_contribution
            ))
        else:
            # Word with no contributions yet
            if ready_only:
                continue

            stats_list.append(WordStats(
                word=word_record.word,
                total_contributions=0,
                unique_contributors=0,
                average_quality=0.0,
                min_quality=0.0,
                max_quality=0.0,
                average_duration=0.0,
                average_fps=0.0,
                total_frames=0,
                ready_for_training=False,
                contributions_needed=50,
                last_contribution=None,
                is_open_for_contribution=word_record.is_open_for_contribution
            ))

    # Sort by total contributions (descending), then by word name
    stats_list.sort(key=lambda x: (x.total_contributions, x.word), reverse=True)

    return stats_list[:limit]


@router.get("/contributions", response_model=List[ContributionDetail])
async def get_all_contributions(
    word: Optional[str] = None,
    user_id: Optional[str] = None,
    min_quality: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    📝 CONTRIBUTION BROWSER - Browse all contributions with filters

    Query params:
    - word: Filter by specific word
    - user_id: Filter by specific user
    - min_quality: Minimum quality score
    - limit/offset: Pagination
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    query = db.query(Contribution)

    # Apply filters
    if word:
        query = query.filter(Contribution.word == word.upper())
    if user_id:
        query = query.filter(Contribution.user_id == user_id)
    if min_quality > 0:
        query = query.filter(Contribution.quality_score >= min_quality)

    # Order by most recent first
    query = query.order_by(desc(Contribution.created_at))

    # Pagination
    contributions = query.offset(offset).limit(limit).all()

    # Convert to detailed response
    details = []
    for c in contributions:
        fps = calculate_fps(c)
        data_points = calculate_data_points(c)
        has_left, has_right = check_hand_presence(c)

        # Handle metadata - it might be JSON string, dict, or SQLAlchemy object
        try:
            if c.metadata is None:
                metadata_dict = None
            elif isinstance(c.metadata, str):
                metadata_dict = json.loads(c.metadata)
            elif isinstance(c.metadata, dict):
                metadata_dict = c.metadata
            else:
                # Try to convert to dict (for SQLAlchemy objects)
                metadata_dict = dict(c.metadata) if c.metadata else None
        except:
            metadata_dict = None

        details.append(ContributionDetail(
            id=c.id,
            word=c.word,
            user_id=c.user_id,
            duration=c.duration,
            quality_score=c.quality_score,
            num_frames=c.num_frames,
            fps=round(fps, 2),
            data_points=data_points,
            has_left_hand=has_left,
            has_right_hand=has_right,
            created_at=c.created_at,
            metadata=metadata_dict
        ))

    return details


@router.get("/quality-report", response_model=DataQualityReport)
async def get_data_quality_report(db: Session = Depends(get_db)):
    """
    🔍 DATA QUALITY REPORT - Comprehensive data quality analysis

    Analyzes:
    - Quality score distribution
    - FPS distribution
    - Hand detection rates
    - Data completeness
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    contributions = db.query(Contribution).all()

    if not contributions:
        # Return empty report when no contributions exist
        return DataQualityReport(
            total_contributions=0,
            high_quality_count=0,
            medium_quality_count=0,
            low_quality_count=0,
            average_quality=0.0,
            quality_distribution={
                "0-20%": 0,
                "20-40%": 0,
                "40-60%": 0,
                "60-80%": 0,
                "80-100%": 0,
            },
            fps_distribution={
                "0-15 FPS": 0,
                "15-25 FPS": 0,
                "25-35 FPS": 0,
                "35+ FPS": 0,
            },
            hand_detection_rate=0.0,
            average_frames_per_contribution=0.0,
            data_completeness_score=0.0
        )

    # Quality distribution
    high_quality = sum(1 for c in contributions if c.quality_score >= 0.8)
    medium_quality = sum(1 for c in contributions if 0.6 <= c.quality_score < 0.8)
    low_quality = sum(1 for c in contributions if c.quality_score < 0.6)

    avg_quality = sum(c.quality_score for c in contributions) / len(contributions)

    # Quality buckets (0-20%, 20-40%, etc.)
    quality_dist = {
        "0-20%": sum(1 for c in contributions if c.quality_score < 0.2),
        "20-40%": sum(1 for c in contributions if 0.2 <= c.quality_score < 0.4),
        "40-60%": sum(1 for c in contributions if 0.4 <= c.quality_score < 0.6),
        "60-80%": sum(1 for c in contributions if 0.6 <= c.quality_score < 0.8),
        "80-100%": sum(1 for c in contributions if c.quality_score >= 0.8),
    }

    # FPS distribution
    fps_values = [calculate_fps(c) for c in contributions]
    fps_dist = {
        "0-15 FPS": sum(1 for fps in fps_values if fps < 15),
        "15-25 FPS": sum(1 for fps in fps_values if 15 <= fps < 25),
        "25-35 FPS": sum(1 for fps in fps_values if 25 <= fps < 35),
        "35+ FPS": sum(1 for fps in fps_values if fps >= 35),
    }

    # Hand detection rate
    hand_detections = [check_hand_presence(c) for c in contributions]
    has_any_hand = sum(1 for left, right in hand_detections if left or right)
    hand_detection_rate = has_any_hand / len(contributions) if contributions else 0.0

    # Average frames
    avg_frames = sum(c.num_frames for c in contributions) / len(contributions)

    # Data completeness (percentage of contributions with high quality + good hand detection)
    complete_data = sum(1 for c, (left, right) in zip(contributions, hand_detections)
                       if c.quality_score >= 0.7 and (left or right))
    completeness_score = complete_data / len(contributions) if contributions else 0.0

    return DataQualityReport(
        total_contributions=len(contributions),
        high_quality_count=high_quality,
        medium_quality_count=medium_quality,
        low_quality_count=low_quality,
        average_quality=round(avg_quality, 3),
        quality_distribution=quality_dist,
        fps_distribution=fps_dist,
        hand_detection_rate=round(hand_detection_rate, 3),
        average_frames_per_contribution=round(avg_frames, 2),
        data_completeness_score=round(completeness_score, 3)
    )


@router.get("/users", response_model=List[UserActivity])
async def get_user_activity(
    min_contributions: int = Query(1, ge=1),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """
    👥 USER ANALYTICS - Track user contributions and activity

    Shows:
    - User contribution counts
    - Words each user has contributed
    - Quality consistency
    - Activity timeline
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get all unique users
    users = db.query(Contribution.user_id).distinct().all()

    activities = []
    for (user_id,) in users:
        contributions = db.query(Contribution).filter(Contribution.user_id == user_id).all()

        if len(contributions) < min_contributions:
            continue

        # Calculate stats
        words = list(set(c.word for c in contributions))
        avg_quality = sum(c.quality_score for c in contributions) / len(contributions)
        total_frames = sum(c.num_frames for c in contributions)

        # Quality consistency (standard deviation)
        qualities = [c.quality_score for c in contributions]
        variance = sum((q - avg_quality) ** 2 for q in qualities) / len(qualities)
        std_dev = variance ** 0.5
        consistency_score = 1.0 - min(std_dev, 1.0)  # Higher = more consistent

        first_contrib = min(c.created_at for c in contributions)
        last_contrib = max(c.created_at for c in contributions)

        activities.append(UserActivity(
            user_id=user_id,
            total_contributions=len(contributions),
            words_contributed=words,
            average_quality=round(avg_quality, 3),
            total_frames=total_frames,
            first_contribution=first_contrib,
            last_contribution=last_contrib,
            consistency_score=round(consistency_score, 3)
        ))

    # Sort by total contributions
    activities.sort(key=lambda x: x.total_contributions, reverse=True)

    return activities[:limit]


@router.delete("/contributions/{contribution_id}")
async def delete_contribution(contribution_id: int, db: Session = Depends(get_db)):
    """
    🗑️ DELETE CONTRIBUTION - Remove a specific contribution

    Use cases:
    - Remove low-quality data
    - Delete duplicate submissions
    - Clean up test data
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    contribution = db.query(Contribution).filter(Contribution.id == contribution_id).first()

    if not contribution:
        raise HTTPException(status_code=404, detail=f"Contribution {contribution_id} not found")

    word = contribution.word
    user_id = contribution.user_id

    db.delete(contribution)
    db.commit()

    return {
        "message": f"Contribution {contribution_id} deleted successfully",
        "word": word,
        "user_id": user_id
    }


@router.delete("/contributions")
async def delete_all_contributions(confirm: bool = False, db: Session = Depends(get_db)):
    """
    🗑️ DELETE ALL CONTRIBUTIONS - Remove ALL contributions from the database

    ⚠️ WARNING: This is a destructive operation and cannot be undone!

    Query Parameters:
    - confirm: Must be set to true to execute deletion

    Use cases:
    - Reset database for fresh start
    - Clean up all test data
    - Prepare for new collection phase

    Returns:
    - Count of deleted contributions
    - Words affected with updated counts
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true query parameter to delete all contributions. This action cannot be undone!"
        )

    # Get count before deletion
    total_contributions = db.query(Contribution).count()

    if total_contributions == 0:
        return {
            "message": "No contributions to delete",
            "deleted_count": 0,
            "words_affected": []
        }

    # Get affected words before deletion
    words_affected = db.query(Contribution.word).distinct().all()
    word_list = [w[0] for w in words_affected]

    # Delete all contributions
    db.query(Contribution).delete()

    # Update word contribution counts to 0
    db.query(Word).update({
        "contributions_count": 0,
        "is_complete": False
    })

    db.commit()

    return {
        "message": f"Successfully deleted all {total_contributions} contributions",
        "deleted_count": total_contributions,
        "words_affected": word_list,
        "warning": "All contribution data has been permanently removed"
    }


@router.get("/contributions/{contribution_id}", response_model=ContributionFullDetail)
async def get_contribution_detail(contribution_id: int, db: Session = Depends(get_db)):
    """
    📋 GET CONTRIBUTION DETAIL - Get full contribution data including pose sequence

    Returns complete contribution information with pose sequence data for preview/analysis.

    Use cases:
    - Preview contribution before deletion
    - Analyze pose quality
    - Export for training
    - Detailed inspection
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    contribution = db.query(Contribution).filter(Contribution.id == contribution_id).first()

    if not contribution:
        raise HTTPException(status_code=404, detail=f"Contribution {contribution_id} not found")

    # Calculate metrics
    fps = calculate_fps(contribution)
    data_points = calculate_data_points(contribution)
    has_left, has_right = check_hand_presence(contribution)

    # Parse metadata
    try:
        if contribution.metadata is None:
            metadata_dict = None
        elif isinstance(contribution.metadata, str):
            metadata_dict = json.loads(contribution.metadata)
        elif isinstance(contribution.metadata, dict):
            metadata_dict = contribution.metadata
        else:
            metadata_dict = dict(contribution.metadata) if contribution.metadata else None
    except:
        metadata_dict = None

    # Use frames_data directly as pose_sequence (already in correct format)
    pose_sequence = contribution.frames_data if hasattr(contribution, 'frames_data') and contribution.frames_data else []

    # Get quality breakdown from frames_data if available
    quality_breakdown = None
    if hasattr(contribution, 'individual_qualities') and contribution.individual_qualities:
        quality_breakdown = contribution.individual_qualities

    return ContributionFullDetail(
        id=contribution.id,
        word=contribution.word,
        user_id=contribution.user_id,
        duration=contribution.duration,
        quality_score=contribution.quality_score,
        num_frames=contribution.num_frames,
        fps=round(fps, 2),
        data_points=data_points,
        has_left_hand=has_left,
        has_right_hand=has_right,
        created_at=contribution.created_at,
        metadata=metadata_dict,
        pose_sequence=pose_sequence,
        quality_breakdown=quality_breakdown
    )


@router.patch("/contributions/{contribution_id}", response_model=ContributionDetail)
async def update_contribution(
    contribution_id: int,
    update_data: ContributionUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    ✏️ UPDATE CONTRIBUTION - Update contribution word or metadata

    Allows admins to:
    - Correct word labels
    - Add/update metadata
    - Fix categorization errors

    Query Parameters:
    - word: New word label (optional)
    - metadata: New metadata dictionary (optional)

    Returns updated contribution details
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    contribution = db.query(Contribution).filter(Contribution.id == contribution_id).first()

    if not contribution:
        raise HTTPException(status_code=404, detail=f"Contribution {contribution_id} not found")

    # Track changes
    changes = []

    # Update word if provided
    if update_data.word is not None:
        old_word = contribution.word
        contribution.word = update_data.word.upper()
        changes.append(f"word: {old_word} → {contribution.word}")
        logger.info(f"Updated contribution {contribution_id} word: {old_word} → {contribution.word}")

    # Update metadata if provided
    if update_data.metadata is not None:
        contribution.metadata = update_data.metadata
        changes.append("metadata updated")
        logger.info(f"Updated contribution {contribution_id} metadata")

    if not changes:
        raise HTTPException(status_code=400, detail="No updates provided")

    db.commit()
    db.refresh(contribution)

    # Calculate metrics for response
    fps = calculate_fps(contribution)
    data_points = calculate_data_points(contribution)
    has_left, has_right = check_hand_presence(contribution)

    # Parse metadata for response
    try:
        if contribution.metadata is None:
            metadata_dict = None
        elif isinstance(contribution.metadata, str):
            metadata_dict = json.loads(contribution.metadata)
        elif isinstance(contribution.metadata, dict):
            metadata_dict = contribution.metadata
        else:
            metadata_dict = dict(contribution.metadata) if contribution.metadata else None
    except:
        metadata_dict = None

    logger.info(f"Contribution {contribution_id} updated successfully: {', '.join(changes)}")

    return ContributionDetail(
        id=contribution.id,
        word=contribution.word,
        user_id=contribution.user_id,
        duration=contribution.duration,
        quality_score=contribution.quality_score,
        num_frames=contribution.num_frames,
        fps=round(fps, 2),
        data_points=data_points,
        has_left_hand=has_left,
        has_right_hand=has_right,
        created_at=contribution.created_at,
        metadata=metadata_dict
    )


@router.post("/contributions/{contribution_id}/approve")
async def approve_contribution(contribution_id: int, db: Session = Depends(get_db)):
    """
    ✅ APPROVE CONTRIBUTION - Mark contribution as verified/approved

    Adds metadata flag for approved contributions
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    contribution = db.query(Contribution).filter(Contribution.id == contribution_id).first()

    if not contribution:
        raise HTTPException(status_code=404, detail=f"Contribution {contribution_id} not found")

    # Update metadata
    metadata = json.loads(contribution.metadata) if contribution.metadata else {}
    metadata['approved'] = True
    metadata['approved_at'] = datetime.utcnow().isoformat()
    contribution.metadata = json.dumps(metadata)

    db.commit()

    return {
        "message": f"Contribution {contribution_id} approved",
        "word": contribution.word,
        "quality_score": contribution.quality_score
    }


@router.get("/analytics/trends")
async def get_contribution_trends(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """
    📈 TREND ANALYSIS - Time-series data for contributions

    Shows contribution trends over time:
    - Daily contribution counts
    - Average quality over time
    - Word diversity trends
    - User growth metrics
    """
    logger.info(f"📈 Trend analysis requested for last {days} days")

    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get contributions from last N days
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    contributions = db.query(Contribution)\
        .filter(Contribution.created_at >= cutoff_date)\
        .order_by(Contribution.created_at)\
        .all()

    if not contributions:
        return {
            "period_days": days,
            "daily_stats": [],
            "total_contributions": 0,
            "trend": "NO_DATA"
        }

    # Group by date
    daily_data = {}
    for c in contributions:
        date_key = c.created_at.date().isoformat()
        if date_key not in daily_data:
            daily_data[date_key] = {
                "date": date_key,
                "count": 0,
                "qualities": [],
                "words": set(),
                "users": set()
            }
        daily_data[date_key]["count"] += 1
        daily_data[date_key]["qualities"].append(c.quality_score)
        daily_data[date_key]["words"].add(c.word)
        daily_data[date_key]["users"].add(c.user_id)

    # Calculate daily stats
    daily_stats = []
    for date_key in sorted(daily_data.keys()):
        data = daily_data[date_key]
        avg_quality = sum(data["qualities"]) / len(data["qualities"])

        daily_stats.append({
            "date": data["date"],
            "contributions": data["count"],
            "average_quality": round(avg_quality, 3),
            "unique_words": len(data["words"]),
            "unique_users": len(data["users"])
        })

    # Determine trend
    if len(daily_stats) >= 2:
        first_half = sum(d["contributions"] for d in daily_stats[:len(daily_stats)//2])
        second_half = sum(d["contributions"] for d in daily_stats[len(daily_stats)//2:])
        trend = "GROWING" if second_half > first_half else "DECLINING" if second_half < first_half else "STABLE"
    else:
        trend = "INSUFFICIENT_DATA"

    logger.info(f"Trend analysis complete: {len(daily_stats)} days, trend={trend}")

    return {
        "period_days": days,
        "daily_stats": daily_stats,
        "total_contributions": len(contributions),
        "trend": trend
    }


@router.get("/analytics/leaderboard")
async def get_contributor_leaderboard(
    metric: str = Query("contributions", regex="^(contributions|quality|frames|consistency)$"),
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db)
):
    """
    🏆 LEADERBOARD - Top contributors by various metrics

    Metrics:
    - contributions: Most contributions
    - quality: Highest average quality
    - frames: Most frames contributed
    - consistency: Most consistent quality
    """
    logger.info(f"🏆 Leaderboard requested: metric={metric}, limit={limit}")

    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get all users
    users = db.query(Contribution.user_id).distinct().all()

    leaderboard = []
    for (user_id,) in users:
        contributions = db.query(Contribution).filter(Contribution.user_id == user_id).all()

        if not contributions:
            continue

        avg_quality = sum(c.quality_score for c in contributions) / len(contributions)
        total_frames = sum(c.num_frames for c in contributions)

        # Calculate consistency (inverse of standard deviation)
        qualities = [c.quality_score for c in contributions]
        variance = sum((q - avg_quality) ** 2 for q in qualities) / len(qualities)
        std_dev = variance ** 0.5
        consistency = 1.0 - min(std_dev, 1.0)

        leaderboard.append({
            "user_id": user_id,
            "total_contributions": len(contributions),
            "average_quality": round(avg_quality, 3),
            "total_frames": total_frames,
            "consistency_score": round(consistency, 3),
            "words": list(set(c.word for c in contributions))
        })

    # Sort by requested metric
    if metric == "contributions":
        leaderboard.sort(key=lambda x: x["total_contributions"], reverse=True)
    elif metric == "quality":
        leaderboard.sort(key=lambda x: x["average_quality"], reverse=True)
    elif metric == "frames":
        leaderboard.sort(key=lambda x: x["total_frames"], reverse=True)
    elif metric == "consistency":
        leaderboard.sort(key=lambda x: x["consistency_score"], reverse=True)

    logger.info(f"Leaderboard generated: {len(leaderboard)} users")

    return {
        "metric": metric,
        "leaderboard": leaderboard[:limit]
    }


@router.get("/analytics/word-coverage")
async def get_word_coverage(db: Session = Depends(get_db)):
    """
    🎯 WORD COVERAGE - Analyze which words need more contributions

    Categories:
    - Ready: >= 10 contributions, quality >= 0.6
    - Almost Ready: 5-9 contributions
    - Needs Work: 1-4 contributions
    - Missing: Words in dictionary but no contributions
    """
    logger.info("🎯 Word coverage analysis requested")

    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get all words with contributions
    words_with_contributions = db.query(Contribution.word).distinct().all()

    ready = []
    almost_ready = []
    needs_work = []

    for (word,) in words_with_contributions:
        contributions = db.query(Contribution).filter(Contribution.word == word).all()
        count = len(contributions)
        avg_quality = sum(c.quality_score for c in contributions) / count

        word_data = {
            "word": word,
            "contributions": count,
            "average_quality": round(avg_quality, 3)
        }

        if count >= 10 and avg_quality >= 0.6:
            ready.append(word_data)
        elif count >= 5:
            almost_ready.append(word_data)
        else:
            needs_work.append(word_data)

    # Get total words in dictionary
    total_words_in_dict = db.query(func.count(Word.id)).scalar() or 0
    words_with_data = len(words_with_contributions)
    missing_words = total_words_in_dict - words_with_data

    logger.info(f"Coverage analysis: {len(ready)} ready, {len(almost_ready)} almost ready, {len(needs_work)} need work")

    return {
        "ready_for_training": {
            "count": len(ready),
            "words": ready
        },
        "almost_ready": {
            "count": len(almost_ready),
            "words": almost_ready
        },
        "needs_work": {
            "count": len(needs_work),
            "words": needs_work
        },
        "missing_data": {
            "count": missing_words
        },
        "coverage_percentage": round((words_with_data / total_words_in_dict * 100) if total_words_in_dict > 0 else 0, 1)
    }


@router.get("/analytics/quality-insights")
async def get_quality_insights(db: Session = Depends(get_db)):
    """
    💎 QUALITY INSIGHTS - Deep dive into data quality patterns

    Analyzes:
    - Quality vs FPS correlation
    - Quality vs duration correlation
    - Hand detection impact on quality
    - Common quality issues
    """
    logger.info("💎 Quality insights analysis requested")

    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    contributions = db.query(Contribution).all()

    if not contributions:
        raise HTTPException(status_code=404, detail="No contributions found")

    # Analyze correlations
    quality_fps_pairs = []
    quality_duration_pairs = []
    hand_detection_impact = {"with_hands": [], "without_hands": []}

    for c in contributions:
        fps = calculate_fps(c)
        quality_fps_pairs.append((c.quality_score, fps))
        quality_duration_pairs.append((c.quality_score, c.duration))

        has_left, has_right = check_hand_presence(c)
        if has_left or has_right:
            hand_detection_impact["with_hands"].append(c.quality_score)
        else:
            hand_detection_impact["without_hands"].append(c.quality_score)

    # Calculate correlations (simplified)
    avg_quality_high_fps = sum(q for q, fps in quality_fps_pairs if fps >= 25) / max(len([q for q, fps in quality_fps_pairs if fps >= 25]), 1)
    avg_quality_low_fps = sum(q for q, fps in quality_fps_pairs if fps < 25) / max(len([q for q, fps in quality_fps_pairs if fps < 25]), 1)

    avg_quality_with_hands = sum(hand_detection_impact["with_hands"]) / max(len(hand_detection_impact["with_hands"]), 1)
    avg_quality_without_hands = sum(hand_detection_impact["without_hands"]) / max(len(hand_detection_impact["without_hands"]), 1)

    # Identify common issues
    issues = []
    if avg_quality_low_fps < 0.6:
        issues.append("Low FPS recordings have poor quality - encourage users to use better devices")
    if avg_quality_without_hands < 0.7:
        issues.append("Contributions without hand detection have lower quality - improve lighting guidance")

    low_quality_count = sum(1 for c in contributions if c.quality_score < 0.6)
    if low_quality_count > len(contributions) * 0.3:
        issues.append(f"{low_quality_count} contributions below quality threshold - review user guidance")

    logger.info(f"Quality insights complete: {len(issues)} issues identified")

    return {
        "fps_impact": {
            "high_fps_quality": round(avg_quality_high_fps, 3),
            "low_fps_quality": round(avg_quality_low_fps, 3),
            "recommendation": "Higher FPS improves quality" if avg_quality_high_fps > avg_quality_low_fps else "FPS has minimal impact"
        },
        "hand_detection_impact": {
            "with_hands": round(avg_quality_with_hands, 3),
            "without_hands": round(avg_quality_without_hands, 3),
            "recommendation": "Hand detection significantly improves quality" if avg_quality_with_hands > avg_quality_without_hands + 0.1 else "Hand detection has minimal impact"
        },
        "common_issues": issues,
        "total_analyzed": len(contributions)
    }


@router.get("/export/contributions")
async def export_contributions_csv(
    word: Optional[str] = None,
    format: str = Query("json", regex="^(json|csv)$"),
    db: Session = Depends(get_db)
):
    """
    💾 EXPORT DATA - Export contributions for analysis

    Formats:
    - json: Structured JSON data
    - csv: Comma-separated values (metadata only, not frames)
    """
    logger.info(f"💾 Data export requested: word={word}, format={format}")

    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    query = db.query(Contribution)
    if word:
        query = query.filter(Contribution.word == word.upper())

    contributions = query.all()

    if format == "csv":
        # CSV export (metadata only)
        csv_data = "id,word,user_id,duration,quality_score,num_frames,created_at\n"
        for c in contributions:
            csv_data += f"{c.id},{c.word},{c.user_id},{c.duration},{c.quality_score},{c.num_frames},{c.created_at}\n"

        logger.info(f"CSV export complete: {len(contributions)} rows")
        return {"format": "csv", "data": csv_data, "count": len(contributions)}

    else:
        # JSON export
        export_data = []
        for c in contributions:
            export_data.append({
                "id": c.id,
                "word": c.word,
                "user_id": c.user_id,
                "duration": c.duration,
                "quality_score": c.quality_score,
                "num_frames": c.num_frames,
                "created_at": c.created_at.isoformat(),
                "fps": round(calculate_fps(c), 2)
            })

        logger.info(f"JSON export complete: {len(export_data)} records")
        return {"format": "json", "data": export_data, "count": len(export_data)}


# ============================================================================
# 🔐 WORD GATING MANAGEMENT
# ============================================================================

@router.post("/words/{word}/open")
async def open_word_for_contribution(word: str, db: Session = Depends(get_db)):
    """
    🔓 OPEN WORD FOR CONTRIBUTION

    Enable data collection for a specific word.
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    word_upper = word.upper()
    word_record = db.query(Word).filter(Word.word == word_upper).first()

    if not word_record:
        raise HTTPException(status_code=404, detail=f"Word '{word_upper}' not found")

    if word_record.is_open_for_contribution:
        return {
            "message": f"Word '{word_upper}' is already open for contributions",
            "word": word_upper,
            "is_open": True
        }

    word_record.is_open_for_contribution = True
    db.commit()
    logger.info(f"✓ Opened word for contribution: {word_upper}")

    return {
        "message": f"Word '{word_upper}' is now open for contributions",
        "word": word_upper,
        "is_open": True
    }


@router.post("/words/{word}/close")
async def close_word_for_contribution(word: str, db: Session = Depends(get_db)):
    """
    🔒 CLOSE WORD FOR CONTRIBUTION

    Disable data collection for a specific word.
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    word_upper = word.upper()
    word_record = db.query(Word).filter(Word.word == word_upper).first()

    if not word_record:
        raise HTTPException(status_code=404, detail=f"Word '{word_upper}' not found")

    if not word_record.is_open_for_contribution:
        return {
            "message": f"Word '{word_upper}' is already closed for contributions",
            "word": word_upper,
            "is_open": False
        }

    word_record.is_open_for_contribution = False
    db.commit()
    logger.info(f"✓ Closed word for contribution: {word_upper}")

    return {
        "message": f"Word '{word_upper}' is now closed for contributions",
        "word": word_upper,
        "is_open": False
    }


@router.post("/words/bulk-open")
async def bulk_open_words(
    words: List[str],
    db: Session = Depends(get_db)
):
    """
    🔓 BULK OPEN WORDS

    Open multiple words for contribution at once.
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    words_upper = [w.upper() for w in words]
    updated_count = 0
    not_found = []

    for word in words_upper:
        word_record = db.query(Word).filter(Word.word == word).first()
        if word_record:
            if not word_record.is_open_for_contribution:
                word_record.is_open_for_contribution = True
                updated_count += 1
        else:
            not_found.append(word)

    db.commit()
    logger.info(f"✓ Bulk opened {updated_count} words")

    return {
        "message": f"Opened {updated_count} words for contributions",
        "updated_count": updated_count,
        "not_found": not_found if not_found else []
    }


@router.get("/health")
async def admin_health_check():
    """🏥 HEALTH CHECK - Verify admin API is operational"""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "endpoint": "/api/ama",
        "database_available": DB_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }
