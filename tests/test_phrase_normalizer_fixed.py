"""
Unit Tests for Phrase Normalizer
Tests phrase mapping and normalization logic - REAL API CONTRACT
"""

import pytest
from pathlib import Path
from phrase_normalizer import PhraseNormalizer


@pytest.fixture
def brain_dir():
    """Get brain directory path"""
    # In Docker, brain is at /app/ghsl_brain
    docker_path = Path("/app/ghsl_brain")
    if docker_path.exists():
        return docker_path
    # Local development
    return Path(__file__).parent.parent.parent / "ghsl_brain"


@pytest.fixture
def normalizer(brain_dir):
    """Create PhraseNormalizer instance"""
    return PhraseNormalizer(brain_dir)


class TestPhraseNormalization:
    """Test phrase normalization functionality"""

    def test_thank_you_normalization(self, normalizer):
        """Test 'thank you' maps to 'THANK'"""
        normalized, matched_phrase = normalizer.normalize("thank you")
        assert normalized == "THANK"
        assert matched_phrase == "thank you"

    def test_good_morning_normalization(self, normalizer):
        """Test 'good morning' maps to 'MORNING'"""
        normalized, matched_phrase = normalizer.normalize("good morning")
        assert normalized == "MORNING"
        assert matched_phrase == "good morning"

    def test_case_insensitivity(self, normalizer):
        """Test normalization is case-insensitive"""
        queries = ["THANK YOU", "Thank You", "thank you", "ThAnK YoU"]
        for query in queries:
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == "THANK", f"Failed for: {query}"

    def test_whitespace_handling(self, normalizer):
        """Test leading/trailing whitespace is handled"""
        queries = ["  thank you  ", "\tthank you\n", " thank you"]
        for query in queries:
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == "THANK", f"Failed for: '{query}'"

    def test_extra_whitespace_between_words(self, normalizer):
        """Test extra whitespace between words"""
        normalized, matched_phrase = normalizer.normalize("thank  you")
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
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"

    def test_natural_language_extraction(self, normalizer):
        """Test natural language pattern extraction"""
        test_cases = [
            ("how to sign hello", "HELLO"),
            ("sign for father", "FATHER"),
            ("what is the sign for love", "LOVE"),
        ]
        for query, expected_word in test_cases:
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == expected_word, f"Failed for: {query}"

    def test_single_word_passthrough(self, normalizer):
        """Test single words pass through to uppercase"""
        words = ["hello", "father", "mother", "school"]
        for word in words:
            normalized, matched_phrase = normalizer.normalize(word)
            assert normalized.upper() == word.upper()

    def test_empty_query(self, normalizer):
        """Test empty query handling"""
        normalized, matched_phrase = normalizer.normalize("")
        assert normalized == ""

    def test_youre_welcome_with_apostrophe(self, normalizer):
        """Test apostrophe handling in phrases"""
        test_cases = [
            "you're welcome",
            "youre welcome",  # Without apostrophe
            "YOU'RE WELCOME",
        ]
        for query in test_cases:
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == "WELCOME", f"Failed for: {query}"

    def test_im_sorry_variations(self, normalizer):
        """Test I'm sorry variations"""
        test_cases = ["i'm sorry", "im sorry", "I'M SORRY", "Im sorry"]
        for query in test_cases:
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == "SORRY", f"Failed for: {query}"

    def test_greeting_phrases(self, normalizer):
        """Test greeting phrase mappings"""
        test_cases = [
            ("good afternoon", "AFTERNOON"),
            ("good evening", "EVENING"),
            ("good night", "NIGHT"),
        ]
        for query, expected in test_cases:
            normalized, matched_phrase = normalizer.normalize(query)
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
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"

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
            normalized, matched_phrase = normalizer.normalize(query)
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
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"

    def test_health_phrases(self, normalizer):
        """Test health-related phrases"""
        test_cases = [("i'm hungry", "HUNGRY"), ("i'm thirsty", "THIRSTY"), ("i'm sick", "SICK")]
        for query, expected in test_cases:
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == expected, f"Failed for: {query}"

    def test_school_phrases(self, normalizer):
        """Test school-related phrases"""
        queries = ["high school", "go to school"]
        for query in queries:
            normalized, matched_phrase = normalizer.normalize(query)
            assert normalized == "SCHOOL", f"Failed for: {query}"


class TestReturnFormat:
    """Test return format is correct"""

    def test_returns_tuple(self, normalizer):
        """Normalize should return a tuple"""
        result = normalizer.normalize("thank you")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_normalized_word(self, normalizer):
        """First element should be the normalized word"""
        normalized, matched_phrase = normalizer.normalize("thank you")
        assert isinstance(normalized, str)
        assert normalized == "THANK"

    def test_second_element_is_matched_phrase(self, normalizer):
        """Second element should be the original matched phrase or None"""
        normalized, matched_phrase = normalizer.normalize("thank you")
        assert matched_phrase in ["thank you", None] or isinstance(matched_phrase, str)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_long_query(self, normalizer):
        """Test handling of very long queries"""
        long_query = "I want to know how to sign thank you in Ghana Sign Language please"
        normalized, matched_phrase = normalizer.normalize(long_query)
        # Should handle without crashing
        assert isinstance(normalized, str)

    def test_special_characters_in_phrase(self, normalizer):
        """Test phrases with special characters"""
        queries = ["thank you!", "hello?", "father...", "mother,"]
        for query in queries:
            normalized, matched_phrase = normalizer.normalize(query)
            # Should handle gracefully
            assert len(normalized) > 0, f"Failed for: {query}"

    def test_numbers_in_query(self, normalizer):
        """Test queries with numbers"""
        normalized, matched_phrase = normalizer.normalize("i want 5 apples")
        # Should handle without crashing
        assert isinstance(normalized, str)

    def test_repeated_words(self, normalizer):
        """Test repeated word handling"""
        normalized, matched_phrase = normalizer.normalize("thank thank you you")
        # Should handle gracefully
        assert isinstance(normalized, str)


class TestInitialization:
    """Test normalizer initialization"""

    def test_normalizer_initializes(self, normalizer):
        """Test that normalizer initializes correctly"""
        assert normalizer is not None
        assert hasattr(normalizer, "normalize")

    def test_phrase_map_is_populated(self, normalizer):
        """Test that phrase map is populated"""
        assert hasattr(normalizer, "phrase_map")
        assert isinstance(normalizer.phrase_map, dict)
        assert len(normalizer.phrase_map) > 0
        # Check some known mappings exist
        assert "thank you" in normalizer.phrase_map
        assert normalizer.phrase_map["thank you"] == "THANK"

    def test_brain_dir_is_set(self, normalizer):
        """Test brain_dir is correctly set"""
        assert hasattr(normalizer, "brain_dir")
        assert isinstance(normalizer.brain_dir, Path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
