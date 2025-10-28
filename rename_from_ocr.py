#!/usr/bin/env python3
"""
RENAME FILES FROM OCR DATA
Rename page_XXX_box_Y.png files to their actual word names
"""
import json
import shutil
from pathlib import Path

class FileRenamer:
    def __init__(self, source_dir, ocr_metadata_file, output_dir):
        self.source_dir = Path(source_dir)
        self.ocr_metadata_file = Path(ocr_metadata_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def rename_files(self):
        """Rename files based on OCR metadata"""
        print("🔄 RENAMING FILES FROM OCR DATA\n")

        # Load OCR metadata
        with open(self.ocr_metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"📊 Loaded {len(metadata)} OCR entries")

        renamed = 0
        failed = 0

        for entry in metadata:
            original_file = entry['original_file']
            new_name = entry['new_file']

            source_path = self.source_dir / original_file
            dest_path = self.output_dir / new_name

            if not source_path.exists():
                print(f"   ⚠️  Source not found: {original_file}")
                failed += 1
                continue

            try:
                # Move (rename) the file
                source_path.rename(dest_path)
                renamed += 1

                if renamed % 100 == 0:
                    print(f"   Progress: {renamed}/{len(metadata)}")

            except Exception as e:
                print(f"   ❌ Failed to rename {original_file}: {e}")
                failed += 1

        print(f"\n✅ RENAMING COMPLETE!")
        print(f"   Renamed: {renamed}")
        print(f"   Failed: {failed}")
        print(f"   Output: {self.output_dir}")

if __name__ == "__main__":
    renamer = FileRenamer(
        source_dir="ghsl_brain/perfect_extraction",
        ocr_metadata_file="ghsl_brain/ocr_processed/ocr_metadata.json",
        output_dir="ghsl_brain/perfect_extraction"  # Overwrite in same folder
    )

    renamer.rename_files()
