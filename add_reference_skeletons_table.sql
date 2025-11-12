-- Migration: Add reference_skeletons table for SignTalk-GH dataset
-- Date: 2025-01-11
-- Description: Store processed pose skeletons with metadata from Metadata.xlsx

CREATE TABLE IF NOT EXISTS reference_skeletons (
    id SERIAL PRIMARY KEY,
    
    -- Video identification
    video_filename VARCHAR(100) UNIQUE NOT NULL,
    sentence_id INTEGER NOT NULL,
    variation VARCHAR(1) NOT NULL,
    
    -- Metadata from Metadata.xlsx
    sentence_text TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    
    -- Video processing metadata
    fps FLOAT NOT NULL,
    total_frames INTEGER NOT NULL,
    extracted_frames INTEGER NOT NULL,
    duration FLOAT,
    
    -- Pose data (JSONB for efficient queries)
    pose_sequence JSONB NOT NULL,
    
    -- Quality metrics
    pose_quality_score FLOAT,
    hand_visibility_score FLOAT,
    
    -- Processing metadata
    processed_at TIMESTAMP DEFAULT NOW(),
    file_size_bytes BIGINT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_sentence_id ON reference_skeletons(sentence_id);
CREATE INDEX IF NOT EXISTS idx_category ON reference_skeletons(category);
CREATE INDEX IF NOT EXISTS idx_variation ON reference_skeletons(variation);
CREATE INDEX IF NOT EXISTS idx_video_filename ON reference_skeletons(video_filename);

-- Add comments
COMMENT ON TABLE reference_skeletons IS 'Reference pose skeletons from SignTalk-GH dataset';
COMMENT ON COLUMN reference_skeletons.video_filename IS 'Original video filename (e.g., 1000A.mp4)';
COMMENT ON COLUMN reference_skeletons.sentence_id IS 'Sentence ID from Metadata.xlsx (e.g., 1000)';
COMMENT ON COLUMN reference_skeletons.variation IS 'Video variation letter (A, B, C, D, E)';
COMMENT ON COLUMN reference_skeletons.pose_sequence IS 'Array of pose frames, each with 75 landmarks (33 pose + 21 left hand + 21 right hand)';
