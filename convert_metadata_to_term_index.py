#!/usr/bin/env python3
"""
CONVERT NEW BRAIN METADATA TO OLD FORMAT
Convert ghsl_brain/new_brain/metadata.json to term_index.json format
so the existing backend can use the corrected OCR brain
"""
import json
from pathlib import Path

class MetadataConverter:
    def __init__(self, metadata_file, output_dir):
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)

    def convert(self):
        """Convert metadata.json to term_index.json"""
        print("🔄 CONVERTING METADATA TO TERM_INDEX FORMAT\n")

        # Load new metadata
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"📊 Loaded {len(metadata)} entries from metadata.json\n")

        # Convert to term_index format
        term_index = {}

        for idx, entry in enumerate(metadata):
            word = entry.get('word', 'UNKNOWN')
            description = entry.get('description', '')
            image = entry.get('image', '')
            entry_type = entry.get('type', 'dictionary')

            # Determine category based on type
            if entry_type == 'alphabet':
                category = 'ALPHABET'
            elif entry_type == 'numeral':
                category = 'NUMERALS'
            else:
                # Try to infer category from description or default to GENERAL
                category = 'GENERAL'
                # Could add smarter categorization based on keywords in description

            # Create term_index entry (matches old format)
            term_index[str(idx)] = {
                'word': word,
                'category': category,
                'image': image,  # Just filename, not path
                'page': idx + 1,  # Use index as page number
                'sign_image': f'/sign_images/{image}',  # Full path for API response
                'sign_id': idx,
                'metadata': {
                    'source': 'Ghana Sign Language Dictionary 3rd Edition',
                    'matched_word': word,
                    'description': description,
                    'type': entry_type
                }
            }

        # Save term_index.json
        output_file = self.output_dir / 'term_index.json'
        with open(output_file, 'w') as f:
            json.dump(term_index, f, indent=2)

        print(f"✅ Saved term_index.json: {output_file}")
        print(f"   Total entries: {len(term_index)}\n")

        # Also save a simplified terms.json for compatibility
        terms_data = {}
        for idx, entry in enumerate(metadata):
            terms_data[str(idx)] = {
                'word': entry['word'],
                'description': entry.get('description', ''),
                'image': entry.get('image', '')
            }

        terms_file = self.output_dir / 'terms.json'
        with open(terms_file, 'w') as f:
            json.dump(terms_data, f, indent=2)

        print(f"✅ Saved terms.json: {terms_file}")
        print(f"   Total entries: {len(terms_data)}\n")

        print("="*70)
        print("✅ CONVERSION COMPLETE!")
        print("="*70)
        print("\nBackend will now use the corrected OCR brain with:")
        print(f"  - {len(term_index)} sign entries")
        print(f"  - Corrected labels from OCR + manual fixes")
        print(f"  - Images in ghsl_brain/final_signs/")

if __name__ == "__main__":
    converter = MetadataConverter(
        metadata_file="ghsl_brain/brain_metadata.json",
        output_dir="ghsl_brain"
    )

    converter.convert()
