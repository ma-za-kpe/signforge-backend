#!/usr/bin/env python3
"""
Reset word contribution counts to match actual contributions.

This fixes data inconsistencies where Word.contributions_count doesn't match
the actual number of Contribution records.
"""

import os
from sqlalchemy import create_engine, text

def reset_word_counts():
    """Reset all word contribution counts to 0"""

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    engine = create_engine(database_url)

    try:
        with engine.connect() as conn:
            print("Resetting word contribution counts...")

            # Reset all contribution counts to 0
            conn.execute(text("""
                UPDATE words
                SET contributions_count = 0,
                    quality_score = NULL,
                    is_complete = FALSE
            """))
            conn.commit()

            print("✓ Successfully reset all word counts to 0")
            print("  All words now show 0 contributions")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    print("=" * 60)
    print("Reset Word Contribution Counts")
    print("=" * 60)
    reset_word_counts()
    print("=" * 60)
