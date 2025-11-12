-- Add is_open_for_contribution column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'words' AND column_name = 'is_open_for_contribution'
    ) THEN
        ALTER TABLE words ADD COLUMN is_open_for_contribution BOOLEAN DEFAULT TRUE NOT NULL;
        RAISE NOTICE 'Column is_open_for_contribution added successfully';
    ELSE
        RAISE NOTICE 'Column is_open_for_contribution already exists';
    END IF;
END $$;
