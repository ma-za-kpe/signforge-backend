"""
Reference Skeletons API
SignTalk-GH dataset skeleton viewer with fast pagination and lazy loading
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from sqlalchemy.orm import Session
from database import get_db, ReferenceSkeleton
import random

router = APIRouter(prefix="/api/skeletons", tags=["reference-skeletons"])


# Pydantic Models (Response schemas)
class SkeletonListItem(BaseModel):
    """Lightweight skeleton info for list view"""
    id: int
    video_filename: str
    sentence_id: int
    variation: str
    sentence_text: str
    category: str
    extracted_frames: int
    duration: Optional[float]
    pose_quality_score: Optional[float]
    
    class Config:
        from_attributes = True


class SkeletonPreview(BaseModel):
    """Skeleton with only first 30 frames for fast preview"""
    id: int
    video_filename: str
    sentence_id: int
    variation: str
    sentence_text: str
    category: str
    fps: float
    total_frames: int
    extracted_frames: int
    duration: Optional[float]
    pose_sequence: List[List[List[float]]] = Field(..., description="First 30 frames only")
    pose_quality_score: Optional[float]
    hand_visibility_score: Optional[float]
    
    class Config:
        from_attributes = True


class SkeletonFull(BaseModel):
    """Complete skeleton data"""
    id: int
    video_filename: str
    sentence_id: int
    variation: str
    sentence_text: str
    category: str
    fps: float
    total_frames: int
    extracted_frames: int
    duration: Optional[float]
    pose_sequence: List[List[List[float]]] = Field(..., description="All frames with 75 landmarks each")
    pose_quality_score: Optional[float]
    hand_visibility_score: Optional[float]
    processed_at: Optional[str]
    
    class Config:
        from_attributes = True


class PaginatedSkeletons(BaseModel):
    """Paginated list response"""
    items: List[SkeletonListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# API Endpoints
@router.get("/", response_model=PaginatedSkeletons)
async def list_skeletons(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    sentence_id: Optional[int] = Query(None, description="Filter by sentence ID"),
    db: Session = Depends(get_db)
):
    """
    List skeletons with pagination and filters
    
    Fast endpoint returning only metadata, no pose data
    """
    query = db.query(ReferenceSkeleton)
    
    # Apply filters
    if category:
        query = query.filter(ReferenceSkeleton.category == category)
    if sentence_id is not None:
        query = query.filter(ReferenceSkeleton.sentence_id == sentence_id)
    
    # Get total count
    total = query.count()
    
    # Paginate
    offset = (page - 1) * page_size
    items = query.offset(offset).limit(page_size).all()
    
    total_pages = (total + page_size - 1) // page_size
    
    return PaginatedSkeletons(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.get("/categories", response_model=List[str])
async def get_categories(db: Session = Depends(get_db)):
    """Get list of all available categories"""
    categories = db.query(ReferenceSkeleton.category).distinct().all()
    return [cat[0] for cat in categories]


@router.get("/random", response_model=SkeletonPreview)
async def get_random_skeleton(
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db)
):
    """Get a random skeleton (preview mode)"""
    query = db.query(ReferenceSkeleton)
    
    if category:
        query = query.filter(ReferenceSkeleton.category == category)
    
    count = query.count()
    if count == 0:
        raise HTTPException(status_code=404, detail="No skeletons found")
    
    # Random offset
    random_offset = random.randint(0, count - 1)
    skeleton = query.offset(random_offset).first()
    
    # Return only first 30 frames
    result = skeleton.__dict__.copy()
    result['pose_sequence'] = skeleton.pose_sequence[:30]
    
    return result


@router.get("/{skeleton_id}/preview", response_model=SkeletonPreview)
async def get_skeleton_preview(skeleton_id: int, db: Session = Depends(get_db)):
    """
    Get skeleton preview (first 30 frames only)
    
    Fast endpoint for initial preview - loads ~50KB instead of 1.3MB
    """
    skeleton = db.query(ReferenceSkeleton).filter(ReferenceSkeleton.id == skeleton_id).first()
    
    if not skeleton:
        raise HTTPException(status_code=404, detail=f"Skeleton {skeleton_id} not found")
    
    # Return only first 30 frames for fast preview
    result = skeleton.__dict__.copy()
    result['pose_sequence'] = skeleton.pose_sequence[:30]
    
    return result


@router.get("/{skeleton_id}", response_model=SkeletonFull)
async def get_skeleton_full(skeleton_id: int, db: Session = Depends(get_db)):
    """
    Get complete skeleton data
    
    Returns all frames - use /preview endpoint for faster initial load
    """
    skeleton = db.query(ReferenceSkeleton).filter(ReferenceSkeleton.id == skeleton_id).first()
    
    if not skeleton:
        raise HTTPException(status_code=404, detail=f"Skeleton {skeleton_id} not found")
    
    return skeleton


@router.get("/{skeleton_id}/frames", response_model=dict)
async def get_skeleton_frames(
    skeleton_id: int,
    start: int = Query(0, ge=0, description="Start frame index"),
    end: int = Query(100, ge=1, description="End frame index"),
    db: Session = Depends(get_db)
):
    """
    Get specific frame range from skeleton
    
    Useful for lazy loading long sequences
    """
    skeleton = db.query(ReferenceSkeleton).filter(ReferenceSkeleton.id == skeleton_id).first()
    
    if not skeleton:
        raise HTTPException(status_code=404, detail=f"Skeleton {skeleton_id} not found")
    
    # Validate range
    if start >= len(skeleton.pose_sequence):
        raise HTTPException(status_code=400, detail="Start index out of range")
    
    end = min(end, len(skeleton.pose_sequence))
    
    return {
        "id": skeleton.id,
        "start": start,
        "end": end,
        "total_frames": len(skeleton.pose_sequence),
        "frames": skeleton.pose_sequence[start:end]
    }
