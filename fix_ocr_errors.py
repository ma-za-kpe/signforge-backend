#!/usr/bin/env python3
"""
FIX OCR MISIDENTIFICATIONS
Rename files with numbered suffixes (SEE_1, SEER_2, etc.)
by extracting the correct word from their descriptions
"""
import json
import re
from pathlib import Path
from typing import Dict, Tuple

class OCRErrorFixer:
    def __init__(self, metadata_file, images_dir):
        self.metadata_file = Path(metadata_file)
        self.images_dir = Path(images_dir)

        # Common words to skip when extracting actual word from description
        self.skip_words = {
            'THE', 'A', 'AN', 'AND', 'OR', 'OF', 'IN', 'ON', 'AT',
            'TO', 'FOR', 'WITH', 'BY', 'FROM', 'UP', 'DOWN', 'OUT',
            'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING',
            'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID',
            'WILL', 'WOULD', 'SHOULD', 'COULD', 'MAY', 'MIGHT',
            'MAKE', 'NAKE', 'PULL', 'PUSH', 'MOVE', 'PLACE', 'PUT',
            'HAND', 'HANDS', 'FINGER', 'FINGERS', 'ARM', 'ARMS',
            'BOTH', 'ONE', 'TWO', 'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD',
            'EERE', 'NITH', 'FRON'  # Common OCR artifacts
        }

    def has_numbered_suffix(self, filename: str) -> bool:
        """Check if filename has numbered suffix like SEE_1.png"""
        return bool(re.match(r'^[A-Z]+_\d+\.png$', filename))

    def extract_actual_word(self, description: str) -> str:
        """Extract the actual word from description"""
        if not description:
            return None

        # Clean description
        description = description.upper().strip()
        words = description.split()

        # Try to find the actual word (first meaningful words)
        result_words = []

        for word in words:
            # Skip common/filler words
            if word in self.skip_words:
                continue

            # Skip very short words
            if len(word) <= 2:
                continue

            # Add this word
            result_words.append(word)

            # If we have 1-3 meaningful words, check if it looks complete
            if len(result_words) >= 1:
                candidate = ' '.join(result_words)

                # If next few words are skip words, we probably have the title
                next_idx = words.index(word) + 1
                if next_idx < len(words):
                    next_words = words[next_idx:next_idx+3]
                    if any(w in self.skip_words for w in next_words):
                        return candidate

                # Stop after 3 words max
                if len(result_words) >= 3:
                    return candidate

        # Return what we have
        if result_words:
            return ' '.join(result_words)

        # Fallback: return first word that's not in skip list
        for word in words:
            if word not in self.skip_words and len(word) > 2:
                return word

        return None

    def sanitize_filename(self, word: str) -> str:
        """Convert word to safe filename"""
        # Replace spaces with underscores
        filename = word.replace(' ', '_')

        # Remove invalid characters
        filename = re.sub(r'[^\w\-]', '', filename)

        return filename.upper()

    def fix_metadata(self) -> Tuple[Dict, int]:
        """Fix metadata and return corrected data"""
        print("🔄 FIXING OCR MISIDENTIFICATIONS\n")

        # Load metadata
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"📊 Loaded {len(metadata)} entries\n")

        fixed_count = 0
        corrections = []

        for entry in metadata:
            current_file = entry.get('new_file') or entry.get('image')

            # Check if this file has numbered suffix
            if not self.has_numbered_suffix(current_file):
                continue

            description = entry.get('description', '')

            # Extract actual word from description
            actual_word = self.extract_actual_word(description)

            if not actual_word:
                print(f"⚠️  Could not extract word from: {current_file}")
                print(f"   Description: {description[:100]}...")
                continue

            # Create new filename
            new_filename = f"{self.sanitize_filename(actual_word)}.png"

            # Check if this would create a duplicate
            existing_entry = next((e for e in metadata if (e.get('new_file') or e.get('image')) == new_filename and e != entry), None)

            if existing_entry:
                # Add suffix to make unique
                counter = 1
                base_name = self.sanitize_filename(actual_word)
                while True:
                    new_filename = f"{base_name}_{counter}.png"
                    if not any((e.get('new_file') or e.get('image')) == new_filename for e in metadata):
                        break
                    counter += 1

            # Record correction
            corrections.append({
                'old_file': current_file,
                'new_file': new_filename,
                'word': actual_word,
                'description': description[:80] + '...' if len(description) > 80 else description
            })

            # Update entry
            entry['word'] = actual_word
            entry['new_file'] = new_filename

            fixed_count += 1

        print(f"\n📝 CORRECTIONS TO BE MADE ({fixed_count} files):\n")
        for i, corr in enumerate(corrections, 1):
            print(f"{i:3d}. {corr['old_file']:25s} → {corr['new_file']:25s}")
            print(f"     Word: {corr['word']}")
            print(f"     Desc: {corr['description']}\n")

        return metadata, corrections

    def rename_files(self, corrections):
        """Rename the actual image files"""
        print(f"\n🔄 RENAMING {len(corrections)} FILES...\n")

        renamed = 0
        failed = 0

        for corr in corrections:
            old_path = self.images_dir / corr['old_file']
            new_path = self.images_dir / corr['new_file']

            if not old_path.exists():
                print(f"   ⚠️  Source not found: {corr['old_file']}")
                failed += 1
                continue

            try:
                old_path.rename(new_path)
                renamed += 1

                if renamed % 10 == 0:
                    print(f"   Progress: {renamed}/{len(corrections)}")

            except Exception as e:
                print(f"   ❌ Failed to rename {corr['old_file']}: {e}")
                failed += 1

        print(f"\n✅ FILES RENAMED:")
        print(f"   Renamed: {renamed}")
        print(f"   Failed: {failed}")

    def save_corrected_metadata(self, metadata):
        """Save corrected metadata"""
        print(f"\n💾 SAVING CORRECTED METADATA...")

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✅ Saved: {self.metadata_file}")

    def run(self):
        """Run the complete fix process"""
        print("="*70)
        print("FIX OCR MISIDENTIFICATIONS")
        print("="*70 + "\n")

        # Fix metadata and get corrections
        corrected_metadata, corrections = self.fix_metadata()

        if not corrections:
            print("\n✅ No corrections needed!")
            return

        # Rename files
        self.rename_files(corrections)

        # Save corrected metadata
        self.save_corrected_metadata(corrected_metadata)

        print("\n" + "="*70)
        print("✅ OCR ERROR FIXING COMPLETE!")
        print("="*70)
        print(f"\nNext step: Rebuild FAISS brain with corrected labels")
        print("Run: python build_brain_from_ocr.py")

if __name__ == "__main__":
    fixer = OCRErrorFixer(
        metadata_file="ghsl_brain/ocr_processed/ocr_metadata.json",
        images_dir="ghsl_brain/final_signs"
    )

    fixer.run()
