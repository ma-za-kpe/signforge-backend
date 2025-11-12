"""
PostgreSQL database models and session management
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghsl:ghsl_dev_pass@localhost:5432/ghsl_contributions")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Word(Base):
    """Dictionary word model"""
    __tablename__ = "words"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String(255), unique=True, index=True, nullable=False)
    contributions_count = Column(Integer, default=0)
    contributions_needed = Column(Integer, default=50)
    is_complete = Column(Boolean, default=False)
    trained_model_path = Column(String(500), nullable=True)
    quality_score = Column(Float, nullable=True)
    is_open_for_contribution = Column(Boolean, default=True, nullable=False)

    # Sign classification metadata (crowdsourced)
    static_votes = Column(Integer, default=0)
    dynamic_votes = Column(Integer, default=0)
    sign_type_consensus = Column(String(20), nullable=True)  # 'static', 'dynamic', or 'unknown'
    consensus_confidence = Column(Float, nullable=True)  # 0.0-1.0

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Contribution(Base):
    """User contribution model"""
    __tablename__ = "contributions"

    id = Column(Integer, primary_key=True, index=True)
    contribution_id = Column(String(100), unique=True, index=True, nullable=False)
    word = Column(String(255), index=True, nullable=False)
    user_id = Column(String(100), nullable=True)
    num_frames = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=False)
    file_path = Column(String(500), nullable=True)  # Nullable since we store frames in JSON column
    frames_data = Column(JSON, nullable=True)  # Store actual frame landmarks for preview

    # User's sign classification (from classification questions)
    sign_type_movement = Column(String(20), nullable=True)  # 'static' or 'dynamic'
    sign_type_hands = Column(String(20), nullable=True)  # 'one-handed' or 'two-handed'

    # 3-attempt metadata (for averaged contributions)
    num_attempts = Column(Integer, default=1)  # Number of attempts averaged (1 for single, 3 for averaged)
    individual_qualities = Column(JSON, nullable=True)  # [0.78, 0.82, 0.75] for 3 attempts
    individual_durations = Column(JSON, nullable=True)  # [1.9, 2.1, 2.0] for 3 attempts
    quality_variance = Column(Float, nullable=True)  # Variance of quality scores across attempts
    improvement_trend = Column(String(100), nullable=True)  # "Consistently improving", etc.

    created_at = Column(DateTime, default=datetime.utcnow)


class ReferenceSkeleton(Base):
    """SignTalk-GH reference skeleton data model"""
    __tablename__ = "reference_skeletons"

    id = Column(Integer, primary_key=True, index=True)

    # Video identification
    video_filename = Column(String(100), unique=True, index=True, nullable=False)  # e.g., "1000A.mp4"
    sentence_id = Column(Integer, index=True, nullable=False)  # e.g., 1000
    variation = Column(String(1), index=True, nullable=False)  # e.g., "A", "B", "D"

    # Metadata from Metadata.xlsx
    sentence_text = Column(Text, nullable=False)  # "Where is the hospital entrance?"
    category = Column(String(100), index=True, nullable=False)  # "General hospital interactions"

    # Video processing metadata
    fps = Column(Float, nullable=False)
    total_frames = Column(Integer, nullable=False)
    extracted_frames = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)  # calculated from extracted_frames/fps

    # Pose data (JSON for efficient storage and queries)
    pose_sequence = Column(JSON, nullable=False)  # Array of frames with 75 landmarks each

    # Quality metrics
    pose_quality_score = Column(Float, nullable=True)  # Average visibility across all frames
    hand_visibility_score = Column(Float, nullable=True)  # % of frames with both hands visible

    # Processing metadata
    processed_at = Column(DateTime, default=datetime.utcnow)
    file_size_bytes = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
