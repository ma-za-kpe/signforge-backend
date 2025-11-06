"""
Human-in-the-Loop Correction Service
Allows users to flag incorrect results and provide feedback
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


class CorrectionFeedback(BaseModel):
    """User feedback on a search result"""

    query: str
    returned_word: str
    correct_word: Optional[str] = None
    is_correct: bool
    user_comment: Optional[str] = None
    timestamp: str = None
    confidence_score: float = 0.0


class CorrectionService:
    """Manages user corrections and feedback"""

    def __init__(self, brain_dir: Path):
        self.brain_dir = brain_dir
        self.corrections_file = brain_dir / "user_corrections.json"
        self.corrections: List[Dict] = []
        self._load_corrections()

    def _load_corrections(self):
        """Load existing corrections from file"""
        if self.corrections_file.exists():
            with open(self.corrections_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.corrections = data.get("corrections", [])

    def _save_corrections(self):
        """Save corrections to file"""
        data = {
            "corrections": self.corrections,
            "total_corrections": len(self.corrections),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.corrections_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def flag_result(
        self,
        query: str,
        returned_word: str,
        correct_word: Optional[str] = None,
        is_correct: bool = False,
        user_comment: Optional[str] = None,
        confidence_score: float = 0.0,
        ip_address: Optional[str] = None,  # NEW: Security tracking
        user_agent: Optional[str] = None,  # NEW: Security tracking
    ) -> Dict:
        """
        Flag a search result as incorrect

        Args:
            query: Original search query
            returned_word: Word that was returned by the system
            correct_word: The correct word (if user knows it)
            is_correct: Whether the result was correct
            user_comment: Optional user comment
            confidence_score: Confidence score of the result
            ip_address: Client IP address (for security/rate limiting)
            user_agent: Client user agent (for security/bot detection)

        Returns:
            Feedback entry with ID
        """
        feedback = {
            "id": len(self.corrections) + 1,
            "query": query.lower(),
            "returned_word": returned_word,
            "correct_word": correct_word,
            "is_correct": is_correct,
            "user_comment": user_comment,
            "confidence_score": confidence_score,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",  # pending, reviewed, applied
            "ip_address": ip_address,  # NEW: Track source
            "user_agent": user_agent,  # NEW: Detect bots
        }

        self.corrections.append(feedback)
        self._save_corrections()

        # NEW: Auto-trigger extraction if threshold met
        if not is_correct and correct_word:
            self._check_auto_fix_threshold(correct_word, query)

        return feedback

    def get_query_corrections(self, query: str) -> List[Dict]:
        """Get all corrections for a specific query"""
        query_lower = query.lower()
        return [c for c in self.corrections if c["query"] == query_lower]

    def get_correction_stats(self) -> Dict:
        """Get statistics about corrections"""
        total = len(self.corrections)
        incorrect = sum(1 for c in self.corrections if not c["is_correct"])
        correct = sum(1 for c in self.corrections if c["is_correct"])
        pending = sum(1 for c in self.corrections if c["status"] == "pending")

        # Most commonly corrected queries (excluding auto-fixed ones)
        query_counts = {}
        for c in self.corrections:
            if not c["is_correct"] and c.get("status") not in ["auto_fixed"]:
                query = c["query"]
                query_counts[query] = query_counts.get(query, 0) + 1

        most_corrected = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_feedback": total,
            "incorrect_results": incorrect,
            "correct_results": correct,
            "pending_review": pending,
            "most_corrected_queries": [{"query": q, "count": c} for q, c in most_corrected],
        }

    def get_learning_data(self) -> Dict[str, str]:
        """
        Extract learning data from corrections
        Returns a mapping of query -> correct_word
        """
        learning_map = {}

        for correction in self.corrections:
            if not correction["is_correct"] and correction["correct_word"]:
                query = correction["query"]
                correct_word = correction["correct_word"]

                # If multiple people corrected the same query,
                # use the most common correction
                if query in learning_map:
                    # Could implement voting mechanism here
                    pass
                else:
                    learning_map[query] = correct_word

        return learning_map

    def apply_correction_boost(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """
        Boost search results based on user corrections

        If users have flagged a query before, boost the correct result

        SECURITY: Disabled by default until consensus-based system implemented
        """
        import os

        # SECURITY FIX: Require explicit opt-in to enable boosting
        # Set environment variable ENABLE_CORRECTION_BOOST=true to enable
        if os.getenv("ENABLE_CORRECTION_BOOST", "false").lower() != "true":
            import logging
            logger = logging.getLogger(__name__)
            logger.info("⚠️ Correction boosting disabled for security (set ENABLE_CORRECTION_BOOST=true to enable)")
            return search_results

        learning_map = self.get_learning_data()
        query_lower = query.lower()

        if query_lower not in learning_map:
            return search_results

        correct_word = learning_map[query_lower].lower()

        # Boost the correct result to the top
        boosted_results = []
        correct_result = None

        for result in search_results:
            if result["word"].lower() == correct_word:
                correct_result = result.copy()
                correct_result["confidence"] = min(1.0, result["confidence"] + 0.2)
                correct_result["boosted"] = True
                correct_result["boost_reason"] = "user_correction"
            else:
                boosted_results.append(result)

        # Place corrected result first
        if correct_result:
            return [correct_result] + boosted_results

        return search_results

    def _check_auto_fix_threshold(self, correct_word: str, query: str):
        """
        Check if auto-fix should be triggered based on correction count

        Threshold: 2+ users report the same missing word
        """
        import logging

        logger = logging.getLogger(__name__)

        if not correct_word:
            return  # No correct word provided

        # Count how many times this word has been reported as missing
        correction_count = len(
            [
                c
                for c in self.corrections
                if c.get("correct_word")
                and c.get("correct_word").upper() == correct_word.upper()
                and not c.get("is_correct", True)  # Only count "incorrect" flags
            ]
        )

        logger.info(f"📊 '{correct_word}' has {correction_count} correction(s)")

        # Threshold: 10+ corrections (increased for security - was 2)
        AUTO_FIX_THRESHOLD = 10  # SECURITY FIX: Prevent easy data poisoning attacks

        if correction_count >= AUTO_FIX_THRESHOLD:
            # Check if already queued/fixed
            recent_fixes = [
                c
                for c in self.corrections
                if c.get("correct_word")
                and c.get("correct_word").upper() == correct_word.upper()
                and c.get("status") in ["auto_fix_queued", "auto_fixed"]
            ]

            if recent_fixes:
                logger.info(f"ℹ️ '{correct_word}' already queued/fixed, skipping")
                return

            # Trigger automatic extraction!
            try:
                from task_queue import queue_extraction

                # Determine category from query context
                category = self._guess_category(query, correct_word)

                # Determine priority based on correction count
                priority = "high" if correction_count >= 5 else "medium"

                success = queue_extraction(correct_word, category, priority)

                if success:
                    logger.info(
                        f"🚀 AUTO-FIX TRIGGERED for '{correct_word}' ({correction_count} reports, {priority} priority)"
                    )

                    # Mark all pending corrections for this word as queued
                    for correction in self.corrections:
                        if (
                            correction.get("correct_word")
                            and correction.get("correct_word").upper() == correct_word.upper()
                            and correction.get("status") == "pending"
                        ):
                            correction["status"] = "auto_fix_queued"
                    self._save_corrections()
                else:
                    logger.error(f"❌ Failed to queue '{correct_word}' - queue full")

            except Exception as e:
                logger.error(f"❌ Error triggering auto-fix for '{correct_word}': {e}")

    def _guess_category(self, query: str, word: str) -> str:
        """Guess category from context"""
        # Simple category detection based on keywords
        categories_map = {
            "COLORS": [
                "color",
                "colour",
                "red",
                "blue",
                "green",
                "yellow",
                "orange",
                "purple",
                "pink",
                "brown",
                "black",
                "white",
            ],
            "FAMILY": [
                "mother",
                "father",
                "brother",
                "sister",
                "family",
                "parent",
                "child",
                "son",
                "daughter",
            ],
            "ANIMALS": ["cow", "dog", "cat", "bird", "fish", "animal", "lion", "elephant"],
            "FOOD": ["food", "eat", "drink", "rice", "water", "bread"],
            "EDUCATION": ["school", "teacher", "student", "learn", "study", "book"],
        }

        query_lower = f"{query} {word}".lower()

        for category, keywords in categories_map.items():
            if any(keyword in query_lower for keyword in keywords):
                return category

        return "GENERAL"

    def mark_auto_fixed(self, word: str):
        """Mark all corrections for a word as auto-fixed"""
        import logging

        logger = logging.getLogger(__name__)

        word_upper = word.upper()
        fixed_count = 0

        for correction in self.corrections:
            correct_word = correction.get("correct_word")
            if (
                correct_word
                and correct_word.upper() == word_upper
                and correction.get("status") in ["pending", "auto_fix_queued"]
            ):
                correction["status"] = "auto_fixed"
                fixed_count += 1

        if fixed_count > 0:
            self._save_corrections()
            logger.info(f"✅ Marked {fixed_count} correction(s) for '{word}' as auto-fixed")


# Global instance
_correction_service: Optional[CorrectionService] = None


def get_correction_service(brain_dir: Path) -> CorrectionService:
    """Get or create correction service singleton"""
    global _correction_service
    if _correction_service is None:
        _correction_service = CorrectionService(brain_dir)
    return _correction_service
