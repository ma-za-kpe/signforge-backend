"""
Auto-Dictionary Fixer
Automatically detects and fixes missing dictionary entries using user feedback
"""
import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
from correction_service import get_correction_service


class AutoDictionaryFixer:
    """
    Analyzes user corrections to identify missing dictionary entries
    and suggests fixes
    """

    def __init__(self, brain_dir: Path):
        self.brain_dir = brain_dir
        self.terms_file = brain_dir / "terms.json"
        self.missing_entries_file = brain_dir / "missing_entries.json"
        self.correction_service = get_correction_service(brain_dir)

    def analyze_missing_entries(self) -> Dict:
        """
        Analyze user corrections to find words that should be in dictionary
        but are missing
        """
        # Get all corrections
        corrections = self.correction_service.corrections

        # Load existing dictionary words
        existing_words = self._get_existing_words()

        # Find missing words from user corrections
        missing_words = {}

        for correction in corrections:
            if not correction.get("is_correct") and correction.get("correct_word"):
                correct_word = correction["correct_word"].upper()
                query = correction["query"].lower()

                # Check if the corrected word exists in dictionary
                if correct_word not in existing_words:
                    if correct_word not in missing_words:
                        missing_words[correct_word] = {
                            "word": correct_word,
                            "queries": [],
                            "correction_count": 0,
                            "first_reported": correction["timestamp"],
                            "last_reported": correction["timestamp"],
                            "user_comments": [],
                        }

                    missing_words[correct_word]["queries"].append(query)
                    missing_words[correct_word]["correction_count"] += 1
                    missing_words[correct_word]["last_reported"] = correction["timestamp"]

                    if correction.get("user_comment"):
                        missing_words[correct_word]["user_comments"].append(
                            correction["user_comment"]
                        )

        # Sort by correction count (most requested first)
        sorted_missing = sorted(
            missing_words.values(),
            key=lambda x: x["correction_count"],
            reverse=True
        )

        report = {
            "total_missing": len(sorted_missing),
            "missing_entries": sorted_missing,
            "analysis_date": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations(sorted_missing)
        }

        # Save report
        with open(self.missing_entries_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _get_existing_words(self) -> Set[str]:
        """Get all words that exist in the dictionary"""
        if not self.terms_file.exists():
            return set()

        with open(self.terms_file, 'r', encoding='utf-8') as f:
            terms = json.load(f)

        return {term["word"].upper() for term in terms}

    def _generate_recommendations(self, missing_entries: List[Dict]) -> List[Dict]:
        """
        Generate actionable recommendations for fixing missing entries
        """
        recommendations = []

        for entry in missing_entries[:10]:  # Top 10 most requested
            word = entry["word"]
            count = entry["correction_count"]

            rec = {
                "priority": "HIGH" if count >= 5 else "MEDIUM" if count >= 2 else "LOW",
                "word": word,
                "correction_count": count,
                "action": f"Add '{word}' to dictionary",
                "steps": [
                    f"1. Locate '{word}' in the Ghana Sign Language Dictionary PDF",
                    f"2. Extract the sign image for '{word}'",
                    f"3. Add to ghsl_brain/sign_images/",
                    f"4. Update terms.json with metadata",
                    f"5. Rebuild vector index",
                    f"6. Test search for: {', '.join(entry['queries'][:3])}"
                ]
            }

            recommendations.append(rec)

        return recommendations

    def suggest_similar_entries(self, missing_word: str) -> List[Dict]:
        """
        Find similar words in dictionary that might be the same as missing word

        This helps identify cases where:
        - Word exists but with different spelling (e.g., "ORGANIZATION" vs "ORGANISATION")
        - Word exists but under synonym (e.g., "COW" vs "CATTLE")
        """
        from difflib import SequenceMatcher

        existing_words = self._get_existing_words()
        similarities = []

        for existing_word in existing_words:
            ratio = SequenceMatcher(None, missing_word.upper(), existing_word.upper()).ratio()

            if ratio > 0.7:  # 70% similar
                similarities.append({
                    "existing_word": existing_word,
                    "missing_word": missing_word,
                    "similarity": ratio,
                    "suggestion": f"Link '{missing_word}' to existing '{existing_word}'"
                })

        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    def auto_create_synonym_mapping(self) -> Dict[str, str]:
        """
        Automatically create synonym mappings from user corrections

        Example: If users consistently correct "cow" to "CATTLE" (which exists),
        create a mapping: "cow" → "CATTLE"
        """
        corrections = self.correction_service.corrections
        existing_words = self._get_existing_words()

        synonym_map = {}

        for correction in corrections:
            if not correction.get("is_correct") and correction.get("correct_word"):
                correct_word = correction["correct_word"].upper()
                query = correction["query"].lower()

                # If the corrected word EXISTS in dictionary
                if correct_word in existing_words:
                    # Create synonym mapping
                    if query not in synonym_map:
                        synonym_map[query] = {
                            "synonym": correct_word,
                            "correction_count": 0
                        }
                    synonym_map[query]["correction_count"] += 1

        # Save synonym mappings
        synonym_file = self.brain_dir / "auto_synonyms.json"
        with open(synonym_file, 'w', encoding='utf-8') as f:
            json.dump({
                "synonyms": synonym_map,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)

        return synonym_map

    def generate_extraction_targets(self) -> List[Dict]:
        """
        Generate a list of words that should be extracted from PDF
        based on user feedback
        """
        report = self.analyze_missing_entries()

        targets = []
        for entry in report["missing_entries"]:
            if entry["correction_count"] >= 2:  # At least 2 users requested it
                targets.append({
                    "word": entry["word"],
                    "priority": entry["correction_count"],
                    "search_queries": entry["queries"],
                    "extraction_command": f"python extract_individual_signs.py --word '{entry['word']}'"
                })

        # Save extraction targets
        targets_file = self.brain_dir / "extraction_targets.json"
        with open(targets_file, 'w', encoding='utf-8') as f:
            json.dump({
                "targets": targets,
                "total_targets": len(targets),
                "generated_at": datetime.now().isoformat()
            }, f, indent=2)

        return targets


def get_auto_fixer(brain_dir: Path) -> AutoDictionaryFixer:
    """Get auto-fixer instance"""
    return AutoDictionaryFixer(brain_dir)


# CLI for manual analysis
if __name__ == "__main__":
    import sys
    from pathlib import Path

    brain_dir = Path(__file__).parent.parent / "ghsl_brain"

    if not brain_dir.exists():
        print(f"Error: Brain directory not found at {brain_dir}")
        sys.exit(1)

    fixer = AutoDictionaryFixer(brain_dir)

    print("🔍 Analyzing user corrections for missing entries...")
    report = fixer.analyze_missing_entries()

    print(f"\n📊 Analysis Results:")
    print(f"   Total missing entries: {report['total_missing']}")

    if report['missing_entries']:
        print(f"\n🔴 Top Missing Words:")
        for entry in report['missing_entries'][:5]:
            print(f"   • {entry['word']}: {entry['correction_count']} corrections")
            print(f"     Queries: {', '.join(entry['queries'][:3])}")

    print(f"\n💡 Generating synonym mappings...")
    synonyms = fixer.auto_create_synonym_mapping()
    print(f"   Created {len(synonyms)} synonym mappings")

    print(f"\n📝 Generating extraction targets...")
    targets = fixer.generate_extraction_targets()
    print(f"   Generated {len(targets)} extraction targets")

    print(f"\n✅ Reports saved to:")
    print(f"   • {fixer.missing_entries_file}")
    print(f"   • {brain_dir / 'auto_synonyms.json'}")
    print(f"   • {brain_dir / 'extraction_targets.json'}")
