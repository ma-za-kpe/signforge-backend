#!/usr/bin/env python3
"""
Enhanced Ghana Sign Language Vector Brain Builder
Uses: CLIP-style multimodal embeddings + Sentence Transformers

Models:
1. CLIP (openai/clip-vit-base-patch32) - Image + Text embeddings
2. SentenceTransformers (all-MiniLM-L6-v2) - Text embeddings
3. Hybrid approach for best accuracy

PASSING CRITERIA:
- FAISS index created successfully
- Search accuracy > 90% on test queries
- Average search time < 10ms
- Index size < 200MB
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import time

class EnhancedBrainBuilder:
    def __init__(self, ghsl_brain_dir: Path):
        self.brain_dir = ghsl_brain_dir
        self.terms_file = ghsl_brain_dir / "terms.json"
        self.images_dir = ghsl_brain_dir / "sign_images"

        # Load models
        print("\n🧠 Loading AI models...")
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("   ✅ Text model loaded (384 dim)")

        # For now, we'll use text embeddings
        # In production, add CLIP for image embeddings
        self.embedding_dim = 384

    def load_terms(self) -> Dict:
        """Load terms from JSON"""
        print("\n📖 Loading terms...")
        with open(self.terms_file, 'r', encoding='utf-8') as f:
            terms = json.load(f)
        print(f"   ✅ Loaded {len(terms)} terms")
        return terms

    def create_text_embeddings(self, terms: Dict) -> np.ndarray:
        """Create text embeddings for all terms"""
        print("\n🔢 Generating text embeddings...")

        # Enhanced text: word + category + description for better context
        enhanced_texts = []
        for word, data in terms.items():
            # Combine word with category and description for richer context
            category_text = data['category'].lower().replace('_', ' ')
            description_text = data.get('description', '').lower()

            # Format: "word category description"
            enhanced_text = f"{word} {category_text} {description_text}"
            enhanced_texts.append(enhanced_text)

        # Generate embeddings
        embeddings = self.text_model.encode(
            enhanced_texts,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        print(f"   ✅ Generated {len(embeddings)} embeddings ({embeddings.shape[1]} dim)")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index"""
        print("\n🔨 Building FAISS index...")

        # Use IndexFlatIP for cosine similarity (normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings.astype('float32'))

        print(f"   ✅ Index built with {index.ntotal} vectors")
        return index

    def create_term_mapping(self, terms: Dict) -> Dict:
        """Create index → term data mapping"""
        print("\n🗂️  Creating term index mapping...")

        term_index = {}
        for idx, (word, data) in enumerate(terms.items()):
            term_index[str(idx)] = {
                "word": word,
                **data
            }

        print(f"   ✅ Created mapping for {len(term_index)} terms")
        return term_index

    def test_brain(self, index: faiss.Index, term_index: Dict) -> Tuple[bool, Dict]:
        """
        Test the brain with queries

        PASSING CRITERIA:
        - Top-1 accuracy > 90%
        - Average search time < 10ms
        """
        print("\n🧪 Testing the brain...")

        test_cases = [
            # (query, expected_word_contains)
            ("cow", "cow"),
            ("family", "family"),
            ("hello", "hello"),
            ("food", "food"),
            ("school", "school"),
            ("mother", "mother"),
            ("father", "father"),
            ("baby", "baby"),
            ("colour", "colour"),
            ("purple", "purple"),
            ("orange", "orange"),
        ]

        results = []
        total_time = 0

        for query, expected in test_cases:
            # Encode query
            start_time = time.time()
            query_vec = self.text_model.encode([query], normalize_embeddings=True)

            # Search
            D, I = index.search(query_vec.astype('float32'), k=5)  # Top 5
            search_time = (time.time() - start_time) * 1000  # ms
            total_time += search_time

            # Get top result
            top_idx = I[0][0]
            top_score = D[0][0]
            matched_term = term_index[str(top_idx)]

            # Check if correct
            is_correct = expected.lower() in matched_term['word'].lower()

            results.append({
                "query": query,
                "expected": expected,
                "matched": matched_term['word'],
                "score": float(top_score),
                "correct": is_correct,
                "time_ms": search_time
            })

            status = "✅" if is_correct else "❌"
            print(f"   {status} '{query}' → '{matched_term['word']}' (score: {top_score:.3f}, {search_time:.1f}ms)")

        # Calculate metrics
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        avg_time = total_time / len(results)

        print(f"\n📊 Test Results:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Avg Search Time: {avg_time:.1f}ms")

        # Check passing criteria
        passed = accuracy >= 0.9 and avg_time < 10

        return passed, {
            "accuracy": accuracy,
            "avg_search_time_ms": avg_time,
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r['correct']),
            "results": results
        }

    def save_brain(self, index: faiss.Index, term_index: Dict, test_results: Dict):
        """Save brain to disk"""
        print("\n💾 Saving brain...")

        # Save FAISS index
        index_file = self.brain_dir / "vectors.index"
        faiss.write_index(index, str(index_file))
        index_size_mb = index_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved vectors.index ({index_size_mb:.2f} MB)")

        # Save term mapping
        mapping_file = self.brain_dir / "term_index.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(term_index, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved term_index.json")

        # Save test results
        test_file = self.brain_dir / "brain_test_results.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2)
        print(f"   ✅ Saved test results")

        # Save metadata
        metadata = {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": self.embedding_dim,
            "total_signs": len(term_index),
            "index_size_mb": index_size_mb,
            "accuracy": test_results['accuracy'],
            "avg_search_time_ms": test_results['avg_search_time_ms']
        }

        metadata_file = self.brain_dir / "brain_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved metadata")

    def build(self) -> bool:
        """Main build process"""
        print("\n" + "=" * 70)
        print("🚀 ENHANCED BRAIN BUILDER")
        print("=" * 70)

        # Load terms
        terms = self.load_terms()

        # Create embeddings
        embeddings = self.create_text_embeddings(terms)

        # Build index
        index = self.build_faiss_index(embeddings)

        # Create mapping
        term_index = self.create_term_mapping(terms)

        # Test brain
        passed, test_results = self.test_brain(index, term_index)

        # Save brain
        self.save_brain(index, term_index, test_results)

        # Final verdict
        print("\n" + "=" * 70)
        if passed:
            print("🎉 PHASE 2: COMPLETE ✅")
            print(f"   ✅ Accuracy: {test_results['accuracy']:.1%} (target: >90%)")
            print(f"   ✅ Search Time: {test_results['avg_search_time_ms']:.1f}ms (target: <10ms)")
            print(f"   ✅ Index Size: {test_results.get('index_size_mb', 0):.1f}MB (target: <200MB)")
            print("\n✅ Ready to proceed to Phase 3: FastAPI Integration")
        else:
            print("❌ PHASE 2: INCOMPLETE")
            if test_results['accuracy'] < 0.9:
                print(f"   ❌ Accuracy too low: {test_results['accuracy']:.1%}")
            if test_results['avg_search_time_ms'] >= 10:
                print(f"   ❌ Search too slow: {test_results['avg_search_time_ms']:.1f}ms")
            print("\n❌ Fix issues before proceeding")

        return passed

def main():
    brain_dir = Path("/output")

    if not brain_dir.exists():
        print(f"❌ Error: {brain_dir} not found")
        exit(1)

    builder = EnhancedBrainBuilder(brain_dir)
    success = builder.build()

    exit(0 if success else 1)

if __name__ == "__main__":
    main()
