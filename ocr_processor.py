#!/usr/bin/env python3
"""
OCR PROCESSOR
Extract text from sign images and rename files with correct labels

Strategy:
1. Split each image into title zone (top) and description zone (bottom)
2. Run OCR on each zone separately
3. Clean and normalize the text
4. Rename files: page_XXX_box_Y.png -> WORD.png
5. Save metadata mapping
"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
import json
from pathlib import Path
import re
from typing import Dict, Tuple, Optional

class OCRProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metadata = []

        # OCR config for better accuracy
        self.title_config = '--psm 7 --oem 3'  # Single line
        self.desc_config = '--psm 6 --oem 3'   # Block of text

    def extract_text_zone(self, img: Image.Image) -> Image.Image:
        """Extract the text zone (bottom 25% containing title + description)"""
        width, height = img.size

        # The text zone is the bottom 25% of the image
        # This contains both the bold title and description
        text_zone = img.crop((0, int(height * 0.75), width, height))

        return text_zone

    def preprocess_for_ocr(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Increase contrast
        img_cv = cv2.convertScaleAbs(img_cv, alpha=1.5, beta=0)

        # Threshold to get black text on white background
        _, img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return Image.fromarray(img_cv)

    def clean_text(self, text: str) -> str:
        """Clean OCR output"""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove special characters except common punctuation
        text = re.sub(r'[^\w\s\-\'\,\.]', '', text)

        # Uppercase for consistency
        text = text.upper().strip()

        return text

    def extract_text_from_zone(self, text_zone: Image.Image) -> Tuple[str, str]:
        """Extract both title and description from text zone"""
        # Preprocess
        processed = self.preprocess_for_ocr(text_zone)

        # Run OCR to get all text
        text = pytesseract.image_to_string(processed, config='--psm 6 --oem 3')

        # Clean
        text = self.clean_text(text)

        if not text:
            return "", ""

        words = text.split()

        if not words:
            return "", ""

        # Common words to skip (not titles)
        skip_words = {'A', 'AN', 'THE', 'AND', 'OR', 'OF', 'IN', 'ON', 'AT',
                     'TO', 'FOR', 'WITH', 'BY', 'FROM', 'UP', 'DOWN', 'OUT'}

        # Strategy: Find the first meaningful word (not a skip word, not 1-2 chars)
        # The title is typically 1-3 words, so we check combinations

        for i in range(len(words)):
            word = words[i]

            # Skip common words and single/double character artifacts
            if word in skip_words or len(word) <= 2:
                continue

            # Found a potential title word
            # Check if it's a 1-3 word title
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                potential_title = ' '.join(words[i:j])
                remaining = ' '.join(words[j:]) if j < len(words) else ""

                # If we have a meaningful title (3-15 chars) and remaining text
                if 3 <= len(potential_title) <= 15 and len(remaining) > 10:
                    return potential_title, remaining

            # Fallback: just this word
            remaining = ' '.join(words[i+1:])
            return word, remaining

        # Last resort: first word is title
        if len(words) >= 2:
            return words[0], ' '.join(words[1:])

        return text, ""

    def sanitize_filename(self, word: str) -> str:
        """Convert word to valid filename"""
        # Remove invalid filename characters
        word = re.sub(r'[<>:"/\\|?*]', '', word)

        # Replace spaces with underscores
        word = word.replace(' ', '_')

        # Limit length
        word = word[:100]

        return word or "UNKNOWN"

    def process_image(self, img_path: Path) -> Optional[Dict]:
        """Process single image: extract text and prepare for renaming"""
        try:
            img = Image.open(img_path)

            # Extract text zone (bottom 25%)
            text_zone = self.extract_text_zone(img)

            # Extract both title and description
            title, description = self.extract_text_from_zone(text_zone)

            if not title:
                print(f"   ⚠️  {img_path.name}: No title extracted")
                title = "UNKNOWN"

            # Create sanitized filename
            new_filename = f"{self.sanitize_filename(title)}.png"

            # Copy image to output with new name
            output_path = self.output_dir / new_filename

            # Handle duplicates
            if output_path.exists():
                counter = 1
                while output_path.exists():
                    new_filename = f"{self.sanitize_filename(title)}_{counter}.png"
                    output_path = self.output_dir / new_filename
                    counter += 1

            # Copy image
            img.save(output_path, quality=95)

            return {
                'original_file': img_path.name,
                'new_file': new_filename,
                'word': title,
                'description': description,
                'status': 'success'
            }

        except Exception as e:
            print(f"   ❌ {img_path.name}: {e}")
            return {
                'original_file': img_path.name,
                'error': str(e),
                'status': 'failed'
            }

    def process_batch(self, pattern='page_*.png', limit=None):
        """Process all dictionary images"""
        print(f"🔍 OCR PROCESSING")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Pattern: {pattern}\n")

        # Get all matching files
        files = sorted(self.input_dir.glob(pattern))

        if limit:
            files = files[:limit]

        print(f"📊 Found {len(files)} images to process\n")

        processed = 0
        success = 0
        failed = 0

        for img_path in files:
            result = self.process_image(img_path)

            if result:
                self.metadata.append(result)

                if result['status'] == 'success':
                    success += 1
                    print(f"   ✅ {result['original_file']} → {result['new_file']}")
                    print(f"      Word: {result['word']}")
                    if result['description']:
                        desc_preview = result['description'][:60] + '...' if len(result['description']) > 60 else result['description']
                        print(f"      Desc: {desc_preview}")
                else:
                    failed += 1

            processed += 1

            if processed % 100 == 0:
                print(f"\n   Progress: {processed}/{len(files)} ({success} success, {failed} failed)\n")

        # Save metadata
        metadata_file = self.output_dir / 'ocr_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ OCR PROCESSING COMPLETE!")
        print(f"   Total: {processed}")
        print(f"   Success: {success}")
        print(f"   Failed: {failed}")
        print(f"   Metadata: {metadata_file}")
        print(f"   Output: {self.output_dir}")

if __name__ == "__main__":
    import sys

    # Test on a few images first
    processor = OCRProcessor(
        input_dir="ghsl_brain/perfect_extraction",
        output_dir="ghsl_brain/ocr_processed"
    )

    # Get limit from command line or default to 10 for testing
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if limit == -1:
        # Process all
        processor.process_batch(limit=None)
    else:
        # Test mode
        print(f"🧪 TEST MODE: Processing {limit} images\n")
        processor.process_batch(limit=limit)
