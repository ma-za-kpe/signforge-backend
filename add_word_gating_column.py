"""
Add is_open_for_contribution column to words table in production database.

This migration adds the word gating feature by adding a boolean column
that controls which words are shown to contributors.
"""

import os
from sqlalchemy import create_engine, text

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL environment variable not set")
    exit(1)

print(f"🔗 Connecting to database...")
engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        # Check if column already exists
        result = connection.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='words' AND column_name='is_open_for_contribution'
        """))

        if result.fetchone():
            print("✅ Column 'is_open_for_contribution' already exists!")
        else:
            print("📝 Adding 'is_open_for_contribution' column...")

            # Add the column with default value TRUE (all words open by default)
            connection.execute(text("""
                ALTER TABLE words
                ADD COLUMN is_open_for_contribution BOOLEAN DEFAULT TRUE NOT NULL
            """))

            connection.commit()
            print("✅ Column added successfully!")
            print("📊 All existing words are now open for contribution by default")

except Exception as e:
    print(f"❌ ERROR: {e}")
    exit(1)

print("🎉 Migration completed!")
