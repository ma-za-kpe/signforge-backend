#!/usr/bin/env python3
"""
BUILD FAISS BRAIN FROM OCR DATA
Create a new FAISS vector database using the OCR-labeled images

Uses:
- OCR-labeled images from ghsl_brain/ocr_processed/
- Metadata from ocr_metadata.json
- Includes alphabet (A-Z) and numerals (0-1M) from perfect_extraction/
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from PIL import Image

class BrainBuilder:
    def __init__(self, ocr_dir, perfect_extraction_dir, output_dir):
        self.ocr_dir = Path(ocr_dir)
        self.perfect_extraction_dir = Path(perfect_extraction_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load sentence transformer for text embeddings
        print("🤖 Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ Model loaded\n")

        self.brain_data = []

    def load_ocr_data(self):
        """Load OCR metadata"""
        metadata_file = self.ocr_dir / 'ocr_metadata.json'

        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return []

        with open(metadata_file, 'r') as f:
            data = json.load(f)

        print(f"📊 Loaded {len(data)} OCR entries")
        return data

    def load_alphabet_numerals(self):
        """Load alphabet and numeral signs"""
        signs = []

        # Alphabet (A-Z)
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            img_path = self.perfect_extraction_dir / f"{letter}.png"
            if img_path.exists():
                signs.append({
                    'word': letter,
                    'description': f'The letter {letter}',
                    'image': f"{letter}.png",
                    'type': 'alphabet'
                })

        # Numerals
        numerals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                   '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                   '30', '40', '50', '60', '70', '80', '90', '100', '1000', '1 MILLION']

        for num in numerals:
            img_path = self.perfect_extraction_dir / f"{num}.png"
            if img_path.exists():
                signs.append({
                    'word': num,
                    'description': f'The number {num}',
                    'image': f"{num}.png",
                    'type': 'numeral'
                })

        print(f"📊 Loaded {len(signs)} alphabet/numeral signs")
        return signs

    def create_embeddings(self, data):
        """Create text embeddings for search"""
        embeddings = []
        metadata = []

        print(f"\n🧠 Creating embeddings for {len(data)} signs...")

        for i, item in enumerate(data):
            word = item.get('word', 'UNKNOWN')
            desc = item.get('description', '')

            # Combine word and description for better search
            text = f"{word}. {desc}".strip()

            # Create embedding
            embedding = self.model.encode(text)
            embeddings.append(embedding)

            # Store metadata
            metadata.append({
                'word': word,
                'description': desc,
                'image': item.get('new_file') or item.get('image'),
                'type': item.get('type', 'dictionary'),
                'original_file': item.get('original_file', '')
            })

            if (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(data)}")

        print(f"   ✅ Created {len(embeddings)} embeddings\n")

        return np.array(embeddings).astype('float32'), metadata

    def build_faiss_index(self, embeddings):
        """Build FAISS index"""
        print("🔨 Building FAISS index...")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        print(f"   ✅ Index built with {index.ntotal} vectors")
        print(f"   Dimension: {dimension}\n")

        return index

    def save_brain(self, index, metadata):
        """Save FAISS index and metadata"""
        print("💾 Saving brain files...")

        # Save FAISS index
        index_file = self.output_dir / 'ghsl.index'
        faiss.write_index(index, str(index_file))
        print(f"   ✅ FAISS index: {index_file}")

        # Save metadata
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadata: {metadata_file}")

        # Save stats
        stats = {
            'total_signs': len(metadata),
            'dictionary_signs': len([m for m in metadata if m['type'] == 'dictionary']),
            'alphabet_signs': len([m for m in metadata if m['type'] == 'alphabet']),
            'numeral_signs': len([m for m in metadata if m['type'] == 'numeral']),
            'embedding_dimension': index.d,
            'model': 'all-MiniLM-L6-v2'
        }

        stats_file = self.output_dir / 'brain_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   ✅ Stats: {stats_file}\n")

        print("="*60)
        print("📊 BRAIN STATISTICS:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

    def build(self):
        """Build the complete brain"""
        print("🎯 BUILDING FAISS BRAIN FROM OCR DATA")
        print("="*60 + "\n")

        # Load OCR data
        ocr_data = self.load_ocr_data()

        # Load alphabet/numerals
        alphabet_numerals = self.load_alphabet_numerals()

        # Combine all data
        all_data = ocr_data + alphabet_numerals

        print(f"\n📊 Total signs to process: {len(all_data)}")
        print(f"   Dictionary: {len(ocr_data)}")
        print(f"   Alphabet/Numerals: {len(alphabet_numerals)}\n")

        # Create embeddings
        embeddings, metadata = self.create_embeddings(all_data)

        # Build FAISS index
        index = self.build_faiss_index(embeddings)

        # Save everything
        self.save_brain(index, metadata)

        print("\n✅ BRAIN BUILD COMPLETE!")
        print(f"   Ready to use in: {self.output_dir}")

if __name__ == "__main__":
    builder = BrainBuilder(
        ocr_dir="ghsl_brain/ocr_processed",
        perfect_extraction_dir="ghsl_brain/perfect_extraction",
        output_dir="ghsl_brain/new_brain"
    )

    builder.build()
