"""
Unit Tests for Phrase Normalizer
Tests phrase mapping and normalization logic
"""

import pytest
from pathlib import Path
from phrase_normalizer import PhraseNormalizer


@pytest.fixture
def brain_dir():
    """Get brain directory path"""
    return Path(__file__).parent.parent.parent / "ghsl_brain"


@pytest.fixture
def normalizer(brain_dir):
    """Create PhraseNormalizer instance"""
    return PhraseNormalizer(brain_dir)


class TestPhraseNormalization:
    """Test phrase normalization functionality"""

    def test_thank_you_normalization(self, normalizer):
        """Test 'thank you' maps to 'THANK'"""
        result, normalized, confidence, boosted = normalizer.normalize("thank you")
        assert normalized == "THANK"
        assert confidence == 1.0
        assert boosted is True

    def test_good_morning_normalization(self, normalizer):
        """Test 'good morning' maps to 'MORNING'"""
        result, normalized, confidence, boosted = normalizer.normalize("good morning")
        assert normalized == "MORNING"
        assert confidence == 1.0
        assert boosted is True

    def test_case_insensitivity(self, normalizer):
        """Test normalization is case-insensitive"""
        queries = ["THANK YOU", "Thank You", "thank you", "ThAnK YoU"]
        for query in queries:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == "THANK", f"Failed for: {query}"
            assert confidence == 1.0

    def test_whitespace_handling(self, normalizer):
        """Test leading/trailing whitespace is handled"""
        queries = ["  thank you  ", "\tthank you\n", " thank you"]
        for query in queries:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == "THANK", f"Failed for: '{query}'"

    def test_extra_whitespace_between_words(self, normalizer):
        """Test extra whitespace between words"""
        result, normalized, confidence, boosted = normalizer.normalize("thank  you")
        assert normalized == "THANK"

    def test_family_phrases(self, normalizer):
        """Test family-related phrase mapping"""
        test_cases = [
            ("my father", "FATHER"),
            ("my mother", "MOTHER"),
            ("my sister", "SISTER"),
            ("my brother", "BROTHER"),
        ]
        for query, expected in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"
            assert confidence == 1.0

    def test_natural_language_extraction(self, normalizer):
        """Test natural language pattern extraction"""
        test_cases = [
            ("how to sign hello", "hello"),
            ("sign for father", "father"),
            ("what is the sign for love", "love"),
            ("show me the sign mother", "mother"),
        ]
        for query, expected_extract in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            # Should extract the word after the pattern
            assert expected_extract.upper() in result.upper() or normalized == expected_extract.upper()

    def test_single_word_passthrough(self, normalizer):
        """Test single words pass through unchanged"""
        words = ["HELLO", "FATHER", "MOTHER", "SCHOOL"]
        for word in words:
            result, normalized, confidence, boosted = normalizer.normalize(word.lower())
            assert normalized.upper() == word
            # Single words that aren't phrase-mapped should not be boosted
            if word.lower() not in normalizer.phrase_map:
                assert boosted is False

    def test_empty_query(self, normalizer):
        """Test empty query handling"""
        result, normalized, confidence, boosted = normalizer.normalize("")
        assert normalized == ""
        assert confidence == 0.0

    def test_unknown_phrase(self, normalizer):
        """Test unknown multi-word phrase"""
        # A phrase not in the mapping should return the original or first word
        result, normalized, confidence, boosted = normalizer.normalize("random unknown phrase")
        # Should not boost confidence
        assert boosted is False or confidence < 1.0

    def test_youre_welcome_with_apostrophe(self, normalizer):
        """Test apostrophe handling in phrases"""
        test_cases = [
            "you're welcome",
            "youre welcome",  # Without apostrophe
            "YOU'RE WELCOME",
        ]
        for query in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == "WELCOME", f"Failed for: {query}"

    def test_im_sorry_variations(self, normalizer):
        """Test I'm sorry variations"""
        test_cases = ["i'm sorry", "im sorry", "I'M SORRY", "Im sorry"]
        for query in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == "SORRY", f"Failed for: {query}"

    def test_greeting_phrases(self, normalizer):
        """Test greeting phrase mappings"""
        test_cases = [
            ("good afternoon", "AFTERNOON"),
            ("good evening", "EVENING"),
            ("good night", "NIGHT"),
        ]
        for query, expected in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"

    def test_question_word_extraction(self, normalizer):
        """Test question word extraction"""
        test_cases = [
            ("what is", "WHAT"),
            ("who is", "WHO"),
            ("where is", "WHERE"),
            ("when is", "WHEN"),
            ("how are", "HOW"),
        ]
        for query, expected in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert expected in normalized.upper(), f"Failed for: {query}"

    def test_i_statements(self, normalizer):
        """Test 'I' statement phrases"""
        test_cases = [
            ("i love you", "LOVE"),
            ("i need", "NEED"),
            ("i want", "WANT"),
            ("i like", "LIKE"),
            ("i have", "HAVE"),
        ]
        for query, expected in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"

    def test_time_phrases(self, normalizer):
        """Test time-related phrases"""
        test_cases = [
            ("what time", "TIME"),
            ("right now", "NOW"),
            ("last night", "NIGHT"),
            ("this morning", "MORNING"),
            ("this afternoon", "AFTERNOON"),
        ]
        for query, expected in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"

    def test_health_phrases(self, normalizer):
        """Test health-related phrases"""
        test_cases = [("i'm hungry", "HUNGRY"), ("i'm thirsty", "THIRSTY"), ("i'm sick", "SICK")]
        for query, expected in test_cases:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"

    def test_school_phrases(self, normalizer):
        """Test school-related phrases"""
        queries = ["high school", "go to school"]
        for query in queries:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            assert normalized == "SCHOOL", f"Failed for: {query}"


class TestConfidenceScoring:
    """Test confidence scoring logic"""

    def test_exact_phrase_match_has_perfect_confidence(self, normalizer):
        """Exact phrase matches should have 1.0 confidence"""
        result, normalized, confidence, boosted = normalizer.normalize("thank you")
        assert confidence == 1.0
        assert boosted is True

    def test_single_word_has_lower_confidence_if_not_mapped(self, normalizer):
        """Single words not in phrase_map should have lower confidence"""
        result, normalized, confidence, boosted = normalizer.normalize("hello")
        # Should not boost if it's just a single word
        if "hello" not in normalizer.phrase_map:
            assert boosted is False

    def test_empty_query_has_zero_confidence(self, normalizer):
        """Empty queries should have 0.0 confidence"""
        result, normalized, confidence, boosted = normalizer.normalize("")
        assert confidence == 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_long_query(self, normalizer):
        """Test handling of very long queries"""
        long_query = "I want to know how to sign thank you in Ghana Sign Language please"
        result, normalized, confidence, boosted = normalizer.normalize(long_query)
        # Should extract "thank you" pattern
        assert "THANK" in normalized.upper() or boosted

    def test_special_characters_in_phrase(self, normalizer):
        """Test phrases with special characters"""
        queries = ["thank you!", "hello?", "father...", "mother,"]
        for query in queries:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            # Should handle gracefully, might strip punctuation
            assert len(normalized) > 0, f"Failed for: {query}"

    def test_numbers_in_query(self, normalizer):
        """Test queries with numbers"""
        result, normalized, confidence, boosted = normalizer.normalize("i want 5 apples")
        # Should handle without crashing
        assert isinstance(normalized, str)

    def test_unicode_characters(self, normalizer):
        """Test Unicode character handling"""
        queries = ["thañk you", "héllo", "fathër"]
        for query in queries:
            result, normalized, confidence, boosted = normalizer.normalize(query)
            # Should handle without crashing
            assert isinstance(normalized, str)

    def test_repeated_words(self, normalizer):
        """Test repeated word handling"""
        result, normalized, confidence, boosted = normalizer.normalize("thank thank you you")
        # Should handle gracefully
        assert isinstance(normalized, str)


class TestInitialization:
    """Test normalizer initialization"""

    def test_normalizer_loads_available_words(self, normalizer):
        """Test that available words are loaded from metadata"""
        assert hasattr(normalizer, "available_words")
        assert isinstance(normalizer.available_words, set)
        # Should have loaded signs from brain_metadata.json
        assert len(normalizer.available_words) > 0

    def test_phrase_map_is_populated(self, normalizer):
        """Test that phrase map is populated"""
        assert hasattr(normalizer, "phrase_map")
        assert isinstance(normalizer.phrase_map, dict)
        assert len(normalizer.phrase_map) > 0
        # Check some known mappings exist
        assert "thank you" in normalizer.phrase_map
        assert normalizer.phrase_map["thank you"] == "THANK"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
