-- Migration: Add pose_sequence and related columns to contributions table
-- Date: 2025-01-12
-- Purpose: Enable skeleton visualization and enhanced contribution metadata

BEGIN;

-- Add pose_sequence column (JSON array of frames with landmarks)
ALTER TABLE contributions
ADD COLUMN IF NOT EXISTS pose_sequence JSON;

-- Add metadata column (JSON for flexible additional data)
ALTER TABLE contributions
ADD COLUMN IF NOT EXISTS metadata JSON;

-- Add FPS and frame tracking columns
ALTER TABLE contributions
ADD COLUMN IF NOT EXISTS fps DOUBLE PRECISION;

ALTER TABLE contributions
ADD COLUMN IF NOT EXISTS total_frames INTEGER;

ALTER TABLE contributions
ADD COLUMN IF NOT EXISTS extracted_frames INTEGER;

-- Add hand detection columns
ALTER TABLE contributions
ADD COLUMN IF NOT EXISTS has_left_hand BOOLEAN DEFAULT FALSE;

ALTER TABLE contributions
ADD COLUMN IF NOT EXISTS has_right_hand BOOLEAN DEFAULT FALSE;

-- Add data_points column for quick stats
ALTER TABLE contributions
ADD COLUMN IF NOT EXISTS data_points INTEGER;

COMMIT;

-- Verify the migration
\d contributions;
