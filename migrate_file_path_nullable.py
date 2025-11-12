#!/usr/bin/env python3
"""
Migration: Make file_path column nullable in contributions table
"""
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghsl:ghsl_dev_pass@localhost:5432/ghsl_contributions")

def migrate():
    """Alter contributions.file_path to be nullable"""
    engine = create_engine(DATABASE_URL)

    try:
        with engine.connect() as conn:
            # Make file_path nullable
            conn.execute(text("""
                ALTER TABLE contributions
                ALTER COLUMN file_path DROP NOT NULL
            """))
            conn.commit()
            print("✓ Migration successful: file_path is now nullable")
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate()
