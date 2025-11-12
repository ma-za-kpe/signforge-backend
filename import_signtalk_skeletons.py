"""
Import SignTalk-GH skeletons into database
Loads pose JSONs and enriches with Metadata.xlsx
"""
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from database import SessionLocal, ReferenceSkeleton
import re
from typing import Dict, Optional

# Paths
# Try container paths first, then fall back to local paths
if Path("/app/data/processed_poses").exists():
    POSES_DIR = Path("/app/data/processed_poses")
    METADATA_PATH = Path("/app/data/signtalk-gsl/SignTalk-GH/Metadata.xlsx")
else:
    BASE_DIR = Path(__file__).parent.parent
    POSES_DIR = BASE_DIR / "data" / "processed_poses"
    METADATA_PATH = BASE_DIR / "data" / "signtalk-gsl" / "SignTalk-GH" / "Metadata.xlsx"


def load_metadata() -> Dict[int, Dict[str, str]]:
    """Load Metadata.xlsx into memory for fast lookups"""
    print(f"📖 Loading metadata from: {METADATA_PATH}")

    df = pd.read_excel(METADATA_PATH)

    # Create lookup dict: {sentence_id: {sentence_text, category}}
    metadata_lookup = {}
    for _, row in df.iterrows():
        sentence_id = int(row['Sentence ID'])
        metadata_lookup[sentence_id] = {
            'sentence_text': row['Sentence Text'],
            'category': row['Category']
        }

    print(f"✅ Loaded metadata for {len(metadata_lookup)} unique sentences")
    return metadata_lookup


def parse_video_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    Parse video filename to extract sentence_id and variation

    Examples:
    - 1000A_poses.json → {sentence_id: 1000, variation: 'A', video_filename: '1000A.mp4'}
    - 2315D_poses.json → {sentence_id: 2315, variation: 'D', video_filename: '2315D.mp4'}
    """
    # Match pattern: digits + letter + _poses.json (or just .json for compatibility)
    match = re.match(r'^(\d+)([A-Z])(?:_poses)?\.json$', filename)

    if not match:
        return None

    sentence_id = int(match.group(1))
    variation = match.group(2)

    return {
        'sentence_id': sentence_id,
        'variation': variation,
        'video_filename': f"{sentence_id}{variation}.mp4"
    }


def calculate_quality_scores(pose_sequence: list) -> Dict[str, float]:
    """
    Calculate quality metrics for pose sequence

    Returns:
    - pose_quality_score: Average visibility across all landmarks and frames
    - hand_visibility_score: % of frames where both hands are visible
    """
    if not pose_sequence or len(pose_sequence) == 0:
        return {'pose_quality_score': 0.0, 'hand_visibility_score': 0.0}

    total_visibility = 0
    total_landmarks = 0
    frames_with_both_hands = 0

    for frame in pose_sequence:
        if not frame or len(frame) < 75:
            continue

        # Calculate average visibility for this frame
        frame_visibility = sum(landmark[3] for landmark in frame if len(landmark) >= 4)
        total_visibility += frame_visibility
        total_landmarks += len(frame)

        # Check if both hands visible (visibility > 0.3)
        left_hand_visible = any(
            frame[i][3] > 0.3 for i in range(33, 54) if i < len(frame) and len(frame[i]) >= 4
        )
        right_hand_visible = any(
            frame[i][3] > 0.3 for i in range(54, 75) if i < len(frame) and len(frame[i]) >= 4
        )

        if left_hand_visible and right_hand_visible:
            frames_with_both_hands += 1

    pose_quality_score = total_visibility / total_landmarks if total_landmarks > 0 else 0.0
    hand_visibility_score = frames_with_both_hands / len(pose_sequence) if pose_sequence else 0.0

    return {
        'pose_quality_score': round(pose_quality_score, 4),
        'hand_visibility_score': round(hand_visibility_score, 4)
    }


def import_skeleton(json_path: Path, metadata_lookup: Dict, db: Session) -> bool:
    """Import a single skeleton JSON file into database"""

    # Parse filename
    parsed = parse_video_filename(json_path.name)
    if not parsed:
        print(f"⚠️  Skipping invalid filename: {json_path.name}")
        return False

    sentence_id = parsed['sentence_id']
    variation = parsed['variation']
    video_filename = parsed['video_filename']

    # Get metadata
    if sentence_id not in metadata_lookup:
        print(f"⚠️  No metadata for sentence {sentence_id}, skipping {video_filename}")
        return False

    metadata = metadata_lookup[sentence_id]

    # Load pose JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {json_path.name}: {e}")
        return False

    # Extract pose data
    pose_sequence = data.get('landmarks', [])
    fps = data.get('fps', 30.0)
    total_frames = data.get('total_frames', len(pose_sequence))
    extracted_frames = len(pose_sequence)
    duration = extracted_frames / fps if fps > 0 else None

    # Calculate quality scores
    quality = calculate_quality_scores(pose_sequence)

    # Get file size
    file_size_bytes = json_path.stat().st_size

    # Check if already exists
    existing = db.query(ReferenceSkeleton).filter(
        ReferenceSkeleton.video_filename == video_filename
    ).first()

    if existing:
        print(f"⏭️  Skipping existing: {video_filename}")
        return False

    # Create database record
    skeleton = ReferenceSkeleton(
        video_filename=video_filename,
        sentence_id=sentence_id,
        variation=variation,
        sentence_text=metadata['sentence_text'],
        category=metadata['category'],
        fps=fps,
        total_frames=total_frames,
        extracted_frames=extracted_frames,
        duration=duration,
        pose_sequence=pose_sequence,
        pose_quality_score=quality['pose_quality_score'],
        hand_visibility_score=quality['hand_visibility_score'],
        processed_at=datetime.utcnow(),
        file_size_bytes=file_size_bytes
    )

    try:
        db.add(skeleton)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving {video_filename}: {e}")
        return False


def main():
    """Main import function"""
    print("🚀 Starting SignTalk-GH skeleton import")
    print(f"📂 Poses directory: {POSES_DIR}")
    print(f"📖 Metadata file: {METADATA_PATH}")

    # Verify paths exist
    if not POSES_DIR.exists():
        print(f"❌ Poses directory not found: {POSES_DIR}")
        return

    if not METADATA_PATH.exists():
        print(f"❌ Metadata file not found: {METADATA_PATH}")
        return

    # Load metadata
    metadata_lookup = load_metadata()

    # Get all JSON files
    json_files = list(POSES_DIR.glob("*.json"))
    total_files = len(json_files)
    print(f"\n📊 Found {total_files} pose JSON files")

    if total_files == 0:
        print("⚠️  No JSON files found to import")
        return

    # Import skeletons
    db = SessionLocal()
    imported = 0
    skipped = 0
    errors = 0

    try:
        for i, json_path in enumerate(json_files, 1):
            if i % 100 == 0:
                print(f"Progress: {i}/{total_files} ({i/total_files*100:.1f}%)")

            result = import_skeleton(json_path, metadata_lookup, db)

            if result:
                imported += 1
            elif result is False:
                skipped += 1
            else:
                errors += 1

        print(f"\n✅ Import complete!")
        print(f"   Imported: {imported}")
        print(f"   Skipped:  {skipped}")
        print(f"   Errors:   {errors}")
        print(f"   Total:    {total_files}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
