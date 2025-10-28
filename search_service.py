"""
FAISS Search Service
Handles vector brain operations
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer

class SearchService:
    """FAISS-powered sign language search"""

    def __init__(self, brain_dir: Path):
        self.brain_dir = brain_dir
        self.index: Optional[faiss.Index] = None
        self.term_index: Optional[Dict] = None
        self.model: Optional[SentenceTransformer] = None
        self.loaded = False

    def load_brain(self):
        """Load FAISS index and model"""
        print("🧠 Loading brain...")

        # Load FAISS index
        index_file = self.brain_dir / "vectors.index"
        if not index_file.exists():
            raise FileNotFoundError(f"Brain not found: {index_file}")

        self.index = faiss.read_index(str(index_file))
        print(f"   ✅ Loaded FAISS index ({self.index.ntotal} vectors)")

        # Load term mapping
        mapping_file = self.brain_dir / "term_index.json"
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.term_index = json.load(f)
        print(f"   ✅ Loaded term index ({len(self.term_index)} terms)")

        # Load model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print(f"   ✅ Loaded embedding model")

        self.loaded = True

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for signs by text query

        Returns:
            List of {word, category, image, page, score} dicts
        """
        if not self.loaded:
            self.load_brain()

        # Encode query
        query_vec = self.model.encode([query], normalize_embeddings=True)

        # Search FAISS
        distances, indices = self.index.search(
            query_vec.astype('float32'),
            k=top_k
        )

        # Build results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            term_data = self.term_index[str(idx)]

            # Convert distance to confidence (0-1)
            # For cosine similarity (IndexFlatIP), higher is better
            confidence = float(distance)

            results.append({
                "rank": i + 1,
                "word": term_data["word"],
                "category": term_data["category"],
                "image": term_data["image"],
                "page": term_data["page"],
                "score": confidence,
                "confidence": confidence
            })

        return results

    def get_stats(self) -> Dict:
        """Get brain statistics"""
        if not self.loaded:
            self.load_brain()

        # Load metadata if exists
        metadata_file = self.brain_dir / "brain_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        return {
            "total_signs": self.index.ntotal if self.index else 0,
            "embedding_dim": self.index.d if self.index else 0,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "loaded": self.loaded,
            **metadata
        }

    def get_sign_by_word(self, word: str) -> Optional[Dict]:
        """Get exact sign by word"""
        if not self.loaded:
            self.load_brain()

        # Search term index
        for idx, term_data in self.term_index.items():
            if term_data["word"].lower() == word.lower():
                return term_data

        return None

# Global instance
_search_service: Optional[SearchService] = None

def get_search_service(brain_dir: Path) -> SearchService:
    """Get or create search service singleton"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService(brain_dir)
        _search_service.load_brain()
    return _search_service
