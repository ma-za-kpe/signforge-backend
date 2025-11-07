#!/usr/bin/env python3
"""
Migration script to add is_open_for_contribution column to words table.

This adds the word gating feature to existing Railway databases.
Run this once on Railway to enable admin control of which words users can contribute to.
"""

import os
import sys
from sqlalchemy import create_engine, text

def add_word_gating_column():
    """Add is_open_for_contribution column to words table if it doesn't exist"""

    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    print(f"Connecting to database...")
    engine = create_engine(database_url)

    try:
        with engine.connect() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='words'
                AND column_name='is_open_for_contribution'
            """))

            if result.fetchone():
                print("✓ Column 'is_open_for_contribution' already exists. No migration needed.")
                return

            print("Adding 'is_open_for_contribution' column to words table...")

            # Add the column (default FALSE - all words closed by default)
            conn.execute(text("""
                ALTER TABLE words
                ADD COLUMN is_open_for_contribution BOOLEAN DEFAULT FALSE NOT NULL
            """))
            conn.commit()

            print("✓ Successfully added 'is_open_for_contribution' column")
            print("  All words are now closed by default (is_open_for_contribution = FALSE)")
            print("  Admins can open specific words via the /ama dashboard")

    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        sys.exit(1)
    finally:
        engine.dispose()

if __name__ == "__main__":
    print("=" * 60)
    print("Word Gating Migration")
    print("=" * 60)
    add_word_gating_column()
    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)
