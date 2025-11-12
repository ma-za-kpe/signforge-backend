"""
Migration script to add sign classification and 3-attempt metadata fields
Run this to update existing database schema
"""
import os
from sqlalchemy import create_engine, text
from database import DATABASE_URL

def migrate():
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        print("🔧 Starting migration: Adding classification and 3-attempt fields...")

        # Add fields to Word table
        try:
            conn.execute(text("""
                ALTER TABLE words
                ADD COLUMN IF NOT EXISTS static_votes INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS dynamic_votes INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS sign_type_consensus VARCHAR(20),
                ADD COLUMN IF NOT EXISTS consensus_confidence FLOAT;
            """))
            conn.commit()
            print("✅ Added classification fields to Word table")
        except Exception as e:
            print(f"⚠️  Word table fields might already exist: {e}")

        # Add fields to Contribution table
        try:
            conn.execute(text("""
                ALTER TABLE contributions
                ADD COLUMN IF NOT EXISTS sign_type_movement VARCHAR(20),
                ADD COLUMN IF NOT EXISTS sign_type_hands VARCHAR(20),
                ADD COLUMN IF NOT EXISTS num_attempts INTEGER DEFAULT 1,
                ADD COLUMN IF NOT EXISTS individual_qualities JSON,
                ADD COLUMN IF NOT EXISTS individual_durations JSON,
                ADD COLUMN IF NOT EXISTS quality_variance FLOAT,
                ADD COLUMN IF NOT EXISTS improvement_trend VARCHAR(100);
            """))
            conn.commit()
            print("✅ Added classification and 3-attempt fields to Contribution table")
        except Exception as e:
            print(f"⚠️  Contribution table fields might already exist: {e}")

        print("✅ Migration complete!")

if __name__ == "__main__":
    migrate()
