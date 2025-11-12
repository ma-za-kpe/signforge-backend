#!/usr/bin/env python3
"""
Skeleton Preview API

Endpoints to preview extracted pose skeletons from processed videos.
Shows MediaPipe 33-point pose landmarks in real-time.

Author: SignForge Team
Date: 2025-01-11
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skeleton-preview", tags=["skeleton-preview"])


@router.get("/available")
async def get_available_skeletons():
    """
    Get list of available skeleton visualizations

    Returns list of processed videos with their pose data
    """
    pose_dir = Path('data/processed_poses')

    if not pose_dir.exists():
        return {"skeletons": [], "total": 0}

    skeletons = []

    for pose_file in sorted(pose_dir.glob('*_poses.json'))[:50]:  # Limit to 50 for preview
        try:
            with open(pose_file) as f:
                data = json.load(f)

            skeletons.append({
                'id': pose_file.stem,
                'filename': pose_file.name,
                'word': data.get('word', 'Unknown'),
                'frames': len(data.get('pose_sequence', [])),
                'video_filename': data.get('video_filename', ''),
                'fps': data.get('fps', 30)
            })
        except Exception as e:
            logger.error(f"Error reading {pose_file}: {e}")
            continue

    return {
        "skeletons": skeletons,
        "total": len(skeletons)
    }


@router.get("/pose/{skeleton_id}")
async def get_skeleton_pose_data(skeleton_id: str):
    """
    Get full pose data for a specific skeleton

    Returns all pose frames with 33 landmarks each
    """
    pose_file = Path(f'data/processed_poses/{skeleton_id}.json')

    if not pose_file.exists():
        raise HTTPException(status_code=404, detail=f"Skeleton '{skeleton_id}' not found")

    try:
        with open(pose_file) as f:
            data = json.load(f)

        return {
            'id': skeleton_id,
            'word': data.get('word', 'Unknown'),
            'video_filename': data.get('video_filename', ''),
            'video_path': data.get('video_path', ''),
            'fps': data.get('fps', 30),
            'total_frames': data.get('total_frames', 0),
            'extracted_frames': len(data.get('pose_sequence', [])),
            'pose_sequence': data.get('pose_sequence', []),
            'processed_at': data.get('processed_at', '')
        }

    except Exception as e:
        logger.error(f"Error loading skeleton {skeleton_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/random")
async def get_random_skeleton():
    """
    Get a random skeleton for quick preview
    """
    pose_dir = Path('data/processed_poses')

    if not pose_dir.exists():
        raise HTTPException(status_code=404, detail="No skeletons available yet")

    pose_files = list(pose_dir.glob('*_poses.json'))

    if not pose_files:
        raise HTTPException(status_code=404, detail="No skeletons available yet")

    # Get random file
    import random
    pose_file = random.choice(pose_files)

    try:
        with open(pose_file) as f:
            data = json.load(f)

        return {
            'id': pose_file.stem,
            'word': data.get('word', 'Unknown'),
            'video_filename': data.get('video_filename', ''),
            'fps': data.get('fps', 30),
            'extracted_frames': len(data.get('pose_sequence', [])),
            'pose_sequence': data.get('pose_sequence', [])
        }

    except Exception as e:
        logger.error(f"Error loading random skeleton: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_skeleton_stats():
    """
    Get statistics about processed skeletons
    """
    pose_dir = Path('data/processed_poses')

    if not pose_dir.exists():
        return {
            "total_skeletons": 0,
            "total_frames": 0,
            "avg_frames_per_skeleton": 0,
            "words": []
        }

    pose_files = list(pose_dir.glob('*_poses.json'))
    total_frames = 0
    words = set()

    for pose_file in pose_files:
        try:
            with open(pose_file) as f:
                data = json.load(f)
                total_frames += len(data.get('pose_sequence', []))
                words.add(data.get('word', 'Unknown'))
        except:
            continue

    return {
        "total_skeletons": len(pose_files),
        "total_frames": total_frames,
        "avg_frames_per_skeleton": total_frames / len(pose_files) if pose_files else 0,
        "unique_words": len(words),
        "words": sorted(list(words))[:20]  # First 20 words
    }
