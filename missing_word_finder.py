"""
MISSING WORD FINDER
Searches open-source sign language databases for missing words
Automatically downloads and adds signs to the brain
"""

import requests
import json
from pathlib import Path
from typing import Dict, Optional, List
from PIL import Image
import io
import hashlib


class MissingWordFinder:
    """Finds and downloads missing signs from open-source databases"""

    def __init__(self, brain_dir: Path):
        self.brain_dir = Path(brain_dir)
        self.sign_images_dir = self.brain_dir / "sign_images"
        self.sign_images_dir.mkdir(exist_ok=True)

        self.terms_file = self.brain_dir / "terms.json"
        self.missing_words_log = self.brain_dir / "missing_words_added.json"

        # Load existing terms
        self.terms = self._load_terms()

        # Load missing words log
        self.missing_log = self._load_missing_log()

        # Open-source sign databases (in order of preference)
        self.search_sources = [
            {
                "name": "Spreadthesign",
                "base_url": "https://www.spreadthesign.com/api/sign",
                "method": self._search_spreadthesign
            },
            {
                "name": "SignASL",
                "base_url": "https://www.signasl.org/api",
                "method": self._search_signasl
            },
            {
                "name": "Handspeak",
                "base_url": "https://www.handspeak.com",
                "method": self._search_handspeak
            }
        ]

    def _load_terms(self) -> Dict:
        """Load existing terms.json"""
        if self.terms_file.exists():
            try:
                with open(self.terms_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _load_missing_log(self) -> Dict:
        """Load log of previously added missing words"""
        if self.missing_words_log.exists():
            try:
                with open(self.missing_words_log, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_missing_log(self):
        """Save missing words log"""
        with open(self.missing_words_log, 'w', encoding='utf-8') as f:
            json.dump(self.missing_log, f, indent=2)

    def _save_terms(self):
        """Save updated terms.json"""
        with open(self.terms_file, 'w', encoding='utf-8') as f:
            json.dump(self.terms, f, indent=2, ensure_ascii=False)

    def word_exists(self, word: str) -> bool:
        """Check if word already exists in our brain"""
        word_upper = word.upper()
        return word_upper in self.terms

    def _search_spreadthesign(self, word: str) -> Optional[Dict]:
        """
        Search Spreadthesign API for sign
        Note: This is a simulation - real API requires authentication
        """
        try:
            # Simulation mode - would use real API in production
            # Real endpoint: GET https://api.spreadthesign.com/sign/{word}
            # For now, return None to trigger next search method
            return None
        except Exception as e:
            print(f"Spreadthesign search failed: {e}")
            return None

    def _search_signasl(self, word: str) -> Optional[Dict]:
        """
        Search SignASL database
        Note: This is a simulation - real implementation would use their API
        """
        try:
            # Simulation mode
            return None
        except Exception as e:
            print(f"SignASL search failed: {e}")
            return None

    def _search_handspeak(self, word: str) -> Optional[Dict]:
        """
        Search Handspeak database
        Note: This is a simulation - real implementation would scrape their site
        """
        try:
            # Simulation mode
            return None
        except Exception as e:
            print(f"Handspeak search failed: {e}")
            return None

    def _search_web_fallback(self, word: str) -> Optional[Dict]:
        """
        Fallback: Use web search to find any sign language image
        Returns a generic placeholder for now
        """
        # In production, this would use Google Custom Search API or similar
        # For now, return None (no match found)
        return None

    def find_missing_word(self, word: str) -> Optional[Dict]:
        """
        Search all sources for a missing word
        Returns sign data if found, None otherwise
        """
        word_upper = word.upper()

        # Check if already in system
        if self.word_exists(word):
            return {"status": "exists", "word": word_upper}

        # Check if we already searched this word before
        if word_upper in self.missing_log:
            log_entry = self.missing_log[word_upper]
            if log_entry.get("found"):
                return {"status": "already_added", "word": word_upper, **log_entry}
            else:
                # Don't search again if we already know it doesn't exist
                return {"status": "not_found_previously", "word": word_upper}

        # Search each source
        for source in self.search_sources:
            result = source["method"](word)
            if result:
                return {
                    "status": "found",
                    "word": word_upper,
                    "source": source["name"],
                    **result
                }

        # Try web fallback
        result = self._search_web_fallback(word)
        if result:
            return {
                "status": "found",
                "word": word_upper,
                "source": "web_search",
                **result
            }

        # Log that we couldn't find it
        self.missing_log[word_upper] = {
            "word": word_upper,
            "searched_at": str(Path(__file__).stat().st_mtime),
            "found": False,
            "sources_tried": [s["name"] for s in self.search_sources]
        }
        self._save_missing_log()

        return {"status": "not_found", "word": word_upper}

    def add_word_to_brain(self, word: str, image_url: str, metadata: Dict = None) -> Dict:
        """
        Download sign image and add word to brain
        """
        word_upper = word.upper()

        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # Open and save image
            img = Image.open(io.BytesIO(response.content))

            # Generate filename
            sign_filename = f"{word_upper}.png"
            sign_path = self.sign_images_dir / sign_filename

            # Save as PNG
            img.save(sign_path, "PNG")

            # Get next sign ID
            max_id = max([int(v.get("sign_id", 0)) for v in self.terms.values()] + [0])
            new_sign_id = max_id + 1

            # Add to terms
            self.terms[word_upper] = {
                "word": word_upper,
                "sign_id": new_sign_id,
                "sign_image": f"/sign_images/{sign_filename}",
                "metadata": metadata or {},
                "source": "auto_added",
                "added_by": "missing_word_finder"
            }

            self._save_terms()

            # Log successful addition
            self.missing_log[word_upper] = {
                "word": word_upper,
                "found": True,
                "sign_id": new_sign_id,
                "image_url": image_url,
                "added_at": str(Path(__file__).stat().st_mtime),
                "metadata": metadata
            }
            self._save_missing_log()

            return {
                "status": "added",
                "word": word_upper,
                "sign_id": new_sign_id,
                "sign_image": f"/sign_images/{sign_filename}"
            }

        except Exception as e:
            return {
                "status": "error",
                "word": word_upper,
                "error": str(e)
            }

    def batch_find_missing(self, words: List[str]) -> Dict:
        """
        Find multiple missing words at once
        Returns summary of found/not found
        """
        results = {
            "found": [],
            "not_found": [],
            "already_exists": [],
            "errors": []
        }

        for word in words:
            if self.word_exists(word):
                results["already_exists"].append(word)
                continue

            result = self.find_missing_word(word)

            if result["status"] == "found":
                # Attempt to add to brain
                add_result = self.add_word_to_brain(
                    word,
                    result.get("image_url", ""),
                    result.get("metadata", {})
                )
                if add_result["status"] == "added":
                    results["found"].append({
                        "word": word,
                        "source": result.get("source", "unknown")
                    })
                else:
                    results["errors"].append({
                        "word": word,
                        "error": add_result.get("error", "Unknown error")
                    })
            else:
                results["not_found"].append(word)

        return {
            **results,
            "summary": {
                "total": len(words),
                "found": len(results["found"]),
                "not_found": len(results["not_found"]),
                "already_exists": len(results["already_exists"]),
                "errors": len(results["errors"])
            }
        }


# Global instance
_missing_word_finder: Optional[MissingWordFinder] = None

def get_missing_word_finder(brain_dir: Path) -> MissingWordFinder:
    """Get or create missing word finder singleton"""
    global _missing_word_finder
    if _missing_word_finder is None:
        _missing_word_finder = MissingWordFinder(brain_dir)
    return _missing_word_finder
