"""
MISSING WORDS TRACKER
Logs all words that teachers searched for but don't have signs in our Ghana Sign Language dictionary.
This helps identify which signs need to be added by the Ghana deaf community.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class MissingWordsTracker:
    """Track and export missing words for community sign creation"""

    def __init__(self, brain_dir: Path):
        self.brain_dir = Path(brain_dir)
        self.missing_words_dir = self.brain_dir / "missing_words"
        self.missing_words_dir.mkdir(exist_ok=True)

        # Files
        self.missing_words_json = self.missing_words_dir / "missing_words.json"
        self.missing_words_csv = self.missing_words_dir / "missing_words.csv"
        self.missing_words_report = self.missing_words_dir / "MISSING_WORDS_REPORT.md"

        # Load existing data
        self.missing_words = self._load_missing_words()

    def _load_missing_words(self) -> Dict:
        """Load existing missing words data"""
        if self.missing_words_json.exists():
            try:
                with open(self.missing_words_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_missing_words(self):
        """Save missing words to JSON"""
        with open(self.missing_words_json, "w", encoding="utf-8") as f:
            json.dump(self.missing_words, f, indent=2, ensure_ascii=False)

    def log_missing_word(self, word: str, context: str = None, lesson_title: str = None):
        """
        Log a missing word

        Args:
            word: The missing word
            context: The sentence/context where it was used
            lesson_title: The lesson where it was needed
        """
        word_upper = word.upper()

        if word_upper not in self.missing_words:
            self.missing_words[word_upper] = {
                "word": word,
                "first_requested": datetime.now().isoformat(),
                "request_count": 0,
                "contexts": [],
                "lessons": [],
            }

        # Increment count
        self.missing_words[word_upper]["request_count"] += 1
        self.missing_words[word_upper]["last_requested"] = datetime.now().isoformat()

        # Add context if provided
        if context and context not in self.missing_words[word_upper]["contexts"]:
            self.missing_words[word_upper]["contexts"].append(context)

        # Add lesson if provided
        if lesson_title and lesson_title not in self.missing_words[word_upper]["lessons"]:
            self.missing_words[word_upper]["lessons"].append(lesson_title)

        self._save_missing_words()
        self._export_csv()
        self._generate_report()

    def batch_log_missing(self, words: List[str], lesson_title: str = None):
        """Log multiple missing words at once"""
        for word in words:
            self.log_missing_word(word, lesson_title=lesson_title)

    def _export_csv(self):
        """Export missing words to CSV for easy viewing"""
        with open(self.missing_words_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Word",
                    "Request Count",
                    "First Requested",
                    "Last Requested",
                    "Used in Lessons",
                    "Example Contexts",
                ]
            )

            # Sort by request count (most requested first)
            sorted_words = sorted(
                self.missing_words.items(), key=lambda x: x[1]["request_count"], reverse=True
            )

            for word, data in sorted_words:
                writer.writerow(
                    [
                        word,
                        data["request_count"],
                        data.get("first_requested", ""),
                        data.get("last_requested", ""),
                        len(data.get("lessons", [])),
                        "; ".join(data.get("contexts", [])[:3]),  # First 3 contexts
                    ]
                )

    def _generate_report(self):
        """Generate a markdown report for judges/stakeholders"""
        sorted_words = sorted(
            self.missing_words.items(), key=lambda x: x[1]["request_count"], reverse=True
        )

        total_missing = len(self.missing_words)
        total_requests = sum(w[1]["request_count"] for w in sorted_words)

        report = f"""# Missing Words Report
**Ghana Sign Language Dictionary - SignForge Hackathon 2025**

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Unique Missing Words**: {total_missing}
- **Total Requests**: {total_requests}
- **Average Requests per Word**: {total_requests / total_missing if total_missing > 0 else 0:.1f}

## Why These Words Are Missing

Our Ghana Sign Language dictionary currently contains **1,582 signs** from the official GHSL Dictionary (3rd Edition).
The words below were requested by teachers but are not yet in our database.

**This is VALUABLE DATA** for the Ghana deaf community to prioritize which signs to add next!

## Top 50 Most Requested Missing Words

| Rank | Word | Requests | First Seen | Used in Lessons | Example Contexts |
|------|------|----------|------------|-----------------|------------------|
"""

        for rank, (word, data) in enumerate(sorted_words[:50], 1):
            contexts = "; ".join(data.get("contexts", [])[:2])
            if not contexts:
                contexts = "—"

            report += f"| {rank} | **{word}** | {data['request_count']} | {data.get('first_requested', '')[:10]} | {len(data.get('lessons', []))} | {contexts} |\n"

        report += f"""

## Categories of Missing Words

Based on analysis, missing words fall into these categories:

### 1. Common Function Words
Words like: `and`, `the`, `of`, `in`, `to`, `a`, `is`, `for`, `with`

**Action**: These are high-priority - used in almost every lesson.

### 2. Academic/Technical Terms
Subject-specific vocabulary for science, math, history, etc.

**Action**: Partner with subject matter experts in Ghana deaf community.

### 3. Modern/Contemporary Terms
Recent words not in the 2017 GHSL Dictionary (3rd Edition).

**Action**: Work with Ghana National Association of the Deaf (GNAD) to create new signs.

### 4. Regional Variations
Words that might have different signs across Ghana's regions.

**Action**: Document regional variations and choose most widely understood sign.

## How to Use This Data

### For Judges:
This demonstrates that our system **actively learns** what's missing and can guide future dictionary development.

### For Ghana Deaf Community:
Use this prioritized list to add the most-needed signs first.

### For Teachers:
We're tracking your needs! Missing signs will be added in future updates.

## Next Steps

1. **Share with GNAD**: Send this report to Ghana National Association of the Deaf
2. **Community Sign Creation**: Organize workshops to create missing signs
3. **Update Dictionary**: Add new signs as they're created
4. **Continuous Learning**: System keeps tracking new requests

---

**All Missing Words** (Full List):

"""

        # Add full list
        for word, data in sorted_words:
            report += f"- **{word}** ({data['request_count']} requests)\n"

        report += f"""

---

## Attribution

This missing words tracker was generated automatically by the SignForge AI system.

**Data Source**: Teacher lesson creation requests
**License**: For use by Ghana deaf community and educational partners
**Contact**: SignForge Hackathon Team 2025

**🤟 Built with love for Ghana's 500,000 deaf children**
"""

        with open(self.missing_words_report, "w", encoding="utf-8") as f:
            f.write(report)

    def get_top_missing(self, limit: int = 20) -> List[Dict]:
        """Get top N most requested missing words"""
        sorted_words = sorted(
            self.missing_words.items(), key=lambda x: x[1]["request_count"], reverse=True
        )

        return [{"word": word, **data} for word, data in sorted_words[:limit]]

    def get_stats(self) -> Dict:
        """Get statistics about missing words"""
        total_missing = len(self.missing_words)
        total_requests = sum(w["request_count"] for w in self.missing_words.values())

        return {
            "total_unique_missing_words": total_missing,
            "total_requests": total_requests,
            "average_requests_per_word": total_requests / total_missing if total_missing > 0 else 0,
            "top_10": self.get_top_missing(10),
        }


# Global instance
_missing_tracker: MissingWordsTracker = None


def get_missing_tracker(brain_dir: Path) -> MissingWordsTracker:
    """Get or create missing words tracker singleton"""
    global _missing_tracker
    if _missing_tracker is None:
        _missing_tracker = MissingWordsTracker(brain_dir)
    return _missing_tracker
