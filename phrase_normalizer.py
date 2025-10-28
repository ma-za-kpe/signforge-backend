"""
PHRASE NORMALIZATION SERVICE
Handles multi-word queries and maps them to single-word signs

Example: "thank you" → "THANK"
         "good morning" → "MORNING"
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple


class PhraseNormalizer:
    """
    Smart phrase normalization for sign language queries
    """

    def __init__(self, brain_dir: Path):
        self.brain_dir = Path(brain_dir)
        self.phrase_map = {}
        self.available_words = set()
        self._build_mappings()

    def _build_mappings(self):
        """Build phrase normalization mappings"""
        # Load available words from brain
        metadata_file = self.brain_dir / "brain_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                self.available_words = {entry["word"].upper() for entry in metadata}

        # Common multi-word phrases that map to single signs
        self.phrase_map = {
            # Greetings & Politeness
            "thank you": "THANK",
            "thanks": "THANK",
            "good morning": "MORNING",
            "good afternoon": "AFTERNOON",
            "good evening": "EVENING",
            "good night": "NIGHT",
            "please help": "HELP",
            "excuse me": "EXCUSE",
            "i'm sorry": "SORRY",
            "im sorry": "SORRY",
            "you're welcome": "WELCOME",
            "youre welcome": "WELCOME",
            # Family & Relations
            "my mother": "MOTHER",
            "my father": "FATHER",
            "my sister": "SISTER",
            "my brother": "BROTHER",
            "my son": "SON",
            "my daughter": "DAUGHTER",
            "my wife": "WIFE",
            "my husband": "HUSBAND",
            # Questions
            "what is": "WHAT",
            "who is": "WHO",
            "where is": "WHERE",
            "when is": "WHEN",
            "how are": "HOW",
            "how much": "MUCH",
            "how many": "MANY",
            # Common phrases
            "i love you": "LOVE",
            "i need": "NEED",
            "i want": "WANT",
            "i like": "LIKE",
            "i have": "HAVE",
            "i am": "I",
            "you are": "YOU",
            "he is": "HE",
            "she is": "SHE",
            # Time
            "what time": "TIME",
            "right now": "NOW",
            "last night": "NIGHT",
            "last week": "WEEK",
            "next week": "WEEK",
            "this morning": "MORNING",
            "this afternoon": "AFTERNOON",
            # Food
            "i'm hungry": "HUNGRY",
            "i'm thirsty": "THIRSTY",
            # Health
            "i'm sick": "SICK",
            "i'm tired": "TIRED",
            # School
            "high school": "SCHOOL",
            "go to school": "SCHOOL",
            # Common missing words (mapped to alternatives)
            "and": "ALSO",  # "and" not in GHSL, use ALSO
            "four": "4",  # Number word to numeral
            "words": "WORD",  # Plural to singular
            "especially": "SPECIAL",  # No "especially", use SPECIAL
            # Number words to numerals
            "one": "1",
            "two": "2",
            "three": "3",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            # Natural language patterns
            "how to sign": "",  # Will extract next word
            "sign for": "",  # Will extract next word
            "what is the sign for": "",  # Will extract next word
            "show me": "",  # Will extract next word
            "how do you sign": "",  # Will extract next word
        }

    def normalize(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Normalize a query to match available signs

        Returns:
            (normalized_query, original_phrase) tuple
            - normalized_query: The word to search for
            - original_phrase: The matched phrase (if any), None otherwise
        """
        import re

        # CRITICAL: Strip whitespace FIRST before any processing
        query = query.strip()

        # Clean query: normalize whitespace, remove punctuation
        query_cleaned = re.sub(r"[^\w\s]", "", query)  # Remove punctuation
        query_cleaned = " ".join(query_cleaned.split())  # Normalize whitespace
        query_lower = query_cleaned.lower().strip()

        # Strategy 1: Natural language patterns FIRST (longest patterns win)
        # Sort natural language patterns by length (longest first)
        natural_patterns = {k: v for k, v in self.phrase_map.items() if v == ""}
        sorted_patterns = sorted(natural_patterns.keys(), key=len, reverse=True)

        for pattern in sorted_patterns:
            if pattern in query_lower:
                # Extract the target word after the pattern
                remaining = query_lower.replace(pattern, "").strip()
                remaining_words = remaining.split()
                if remaining_words:
                    target_word = remaining_words[0].upper()
                    if target_word in self.available_words:
                        return (target_word, None)

        # Strategy 2: Direct phrase mapping
        if query_lower in self.phrase_map:
            normalized = self.phrase_map[query_lower]
            # Only use if word exists in brain
            if normalized in self.available_words:
                return (normalized, query_lower)

        # Strategy 3: Partial phrase matching
        # Check if query contains any known phrases
        for phrase, word in self.phrase_map.items():
            if word != "" and phrase in query_lower:  # Skip natural language patterns
                if word in self.available_words:
                    return (word, phrase)

        # Strategy 4: Extract main word from multi-word query
        # e.g., "thank you very much" → check for "thank"
        words = query_lower.split()
        if len(words) > 1:
            # Try each word to see if it exists
            for word in words:
                word_upper = word.upper()
                if word_upper in self.available_words:
                    return (word_upper, None)

            # Try combinations
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    combo = " ".join(words[i:j])
                    if combo in self.phrase_map:
                        normalized = self.phrase_map[combo]
                        if normalized in self.available_words:
                            return (normalized, combo)

        # Strategy 5: No normalization needed - return cleaned query
        return (query_cleaned.upper(), None)

    def get_suggestions_for_phrase(self, query: str) -> List[str]:
        """
        Get suggestions for multi-word queries
        Returns list of possible matches
        """
        query_lower = query.lower().strip()
        suggestions = []

        # Find all matching phrases
        for phrase, word in self.phrase_map.items():
            if query_lower in phrase or phrase.startswith(query_lower):
                if word in self.available_words:
                    suggestions.append(word)

        return list(set(suggestions))  # Remove duplicates

    def explain_normalization(self, query: str) -> str:
        """
        Explain how a query was normalized (for debugging)
        """
        normalized, matched_phrase = self.normalize(query)

        if matched_phrase:
            return f'"{query}" → matched phrase "{matched_phrase}" → sign "{normalized}"'
        elif normalized != query.upper():
            return f'"{query}" → extracted word → sign "{normalized}"'
        else:
            return f'"{query}" → direct lookup → sign "{normalized}"'


# Global instance
_normalizer: Optional[PhraseNormalizer] = None


def get_phrase_normalizer(brain_dir: Path) -> PhraseNormalizer:
    """Get or create phrase normalizer singleton"""
    global _normalizer
    if _normalizer is None:
        _normalizer = PhraseNormalizer(brain_dir)
    return _normalizer
