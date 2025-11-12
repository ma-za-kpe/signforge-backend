-- Migration: Populate pose_sequence from frames_data using SQL
-- This extracts landmarks from the old frames_data format

BEGIN;

-- First, let's create a temporary function to extract pose landmarks
-- We'll use a simpler approach: just extract the pose_landmarks array from each frame

-- Update each contribution one by one
-- For contribution 38
UPDATE contributions
SET
  pose_sequence = (
    SELECT json_agg(
      (
        SELECT json_agg(
          json_build_array(
            landmark->>'x',
            landmark->>'y',
            landmark->>'z',
            landmark->>'visibility'
          )
        )
        FROM json_array_elements(frame->'pose_landmarks') AS landmark
      )
    )
    FROM json_array_elements(frames_data) AS frame
  ),
  fps = CASE
    WHEN duration > 0 THEN num_frames / duration
    ELSE 30.0
  END,
  total_frames = num_frames,
  extracted_frames = num_frames,
  has_left_hand = EXISTS (
    SELECT 1
    FROM json_array_elements(frames_data) AS frame
    WHERE frame->'left_hand_landmarks' IS NOT NULL
      AND json_array_length(frame->'left_hand_landmarks') > 0
  ),
  has_right_hand = EXISTS (
    SELECT 1
    FROM json_array_elements(frames_data) AS frame
    WHERE frame->'right_hand_landmarks' IS NOT NULL
      AND json_array_length(frame->'right_hand_landmarks') > 0
  ),
  data_points = num_frames * 75
WHERE frames_data IS NOT NULL
  AND pose_sequence IS NULL;

COMMIT;

-- Verify the results
SELECT
  id,
  word,
  pose_sequence IS NOT NULL AS has_pose_seq,
  json_array_length(pose_sequence) AS num_frames_in_pose,
  fps,
  has_left_hand,
  has_right_hand
FROM contributions
WHERE frames_data IS NOT NULL
LIMIT 5;
