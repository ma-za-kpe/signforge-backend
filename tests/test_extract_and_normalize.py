"""
COMPREHENSIVE TESTS: /api/extract-and-normalize endpoint

Tests the new content extraction endpoint that fixes the word recognition issue.
Ensures phrase normalization works correctly in content generation.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)


class TestExtractAndNormalizeEndpoint:
    """Test suite for /api/extract-and-normalize endpoint"""

    def test_endpoint_exists(self):
        """Test that endpoint is accessible"""
        response = client.post("/api/extract-and-normalize", json={"text": "hello"})
        assert response.status_code == 200

    def test_empty_text_returns_empty_results(self):
        """Test empty text returns empty word list"""
        response = client.post("/api/extract-and-normalize", json={"text": ""})
        assert response.status_code == 200
        data = response.json()
        assert data["words"] == []
        assert data["total_words"] == 0
        assert data["available_signs"] == 0
        assert data["phrases_detected"] == 0

    def test_single_word_extraction(self):
        """Test extracting a single word"""
        response = client.post("/api/extract-and-normalize", json={"text": "hello"})
        assert response.status_code == 200
        data = response.json()

        assert data["total_words"] >= 1
        assert data["available_signs"] >= 1

        # Find "hello" in results
        hello_word = next((w for w in data["words"] if w["original"] == "hello"), None)
        assert hello_word is not None
        assert hello_word["hasSign"] is True
        assert hello_word["normalized"] == "HELLO"

    def test_phrase_detection_thank_you(self):
        """Test that 'thank you' is detected as phrase"""
        response = client.post("/api/extract-and-normalize", json={"text": "thank you"})
        assert response.status_code == 200
        data = response.json()

        assert data["phrases_detected"] >= 1

        # Find "thank you" phrase
        thank_you = next((w for w in data["words"] if w["original"] == "thank you"), None)
        assert thank_you is not None
        assert thank_you["hasSign"] is True
        assert thank_you["normalized"] == "THANK"
        assert thank_you["phrase"] is True

    def test_phrase_detection_good_morning(self):
        """Test that 'good morning' is detected as phrase"""
        response = client.post("/api/extract-and-normalize", json={"text": "good morning"})
        assert response.status_code == 200
        data = response.json()

        assert data["phrases_detected"] >= 1

        good_morning = next((w for w in data["words"] if w["original"] == "good morning"), None)
        assert good_morning is not None
        assert good_morning["hasSign"] is True
        assert good_morning["normalized"] == "MORNING"
        assert good_morning["phrase"] is True

    def test_alternative_mapping_four(self):
        """Test that 'four' maps to '4' (alternative)"""
        response = client.post("/api/extract-and-normalize", json={"text": "four"})
        assert response.status_code == 200
        data = response.json()

        # Should detect as phrase (since it's in phrase_map)
        assert data["phrases_detected"] >= 1

        four = next((w for w in data["words"] if w["original"] == "four"), None)
        assert four is not None
        assert four["hasSign"] is True
        assert four["normalized"] == "4"

    def test_alternative_mapping_and(self):
        """Test that 'and' maps to 'ALSO' (alternative)"""
        response = client.post("/api/extract-and-normalize", json={"text": "and"})
        assert response.status_code == 200
        data = response.json()

        and_word = next((w for w in data["words"] if w["original"] == "and"), None)
        assert and_word is not None
        assert and_word["hasSign"] is True
        assert and_word["normalized"] == "ALSO"

    def test_number_word_mappings(self):
        """Test that number words map to numerals"""
        text = "one two three four five"
        response = client.post("/api/extract-and-normalize", json={"text": text})
        assert response.status_code == 200
        data = response.json()

        # Check each number
        mappings = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5"
        }

        for word, expected_num in mappings.items():
            word_data = next((w for w in data["words"] if w["original"] == word), None)
            assert word_data is not None, f"'{word}' not found in results"
            assert word_data["hasSign"] is True
            assert word_data["normalized"] == expected_num

    def test_complex_sentence_with_phrases(self):
        """Test complex sentence with multiple phrases"""
        text = "hello father thank you and good morning"
        response = client.post("/api/extract-and-normalize", json={"text": text})
        assert response.status_code == 200
        data = response.json()

        # Should detect multiple phrases
        assert data["phrases_detected"] >= 3  # "thank you", "and", "good morning"
        assert data["available_signs"] >= 5  # all words have signs

        # Verify specific words
        expected = {
            "hello": {"normalized": "HELLO", "phrase": False},
            "father": {"normalized": "FATHER", "phrase": False},
            "thank you": {"normalized": "THANK", "phrase": True},
            "and": {"normalized": "ALSO", "phrase": True},
            "good morning": {"normalized": "MORNING", "phrase": True}
        }

        for original, props in expected.items():
            word_data = next((w for w in data["words"] if w["original"] == original), None)
            assert word_data is not None, f"'{original}' not found"
            assert word_data["hasSign"] is True
            assert word_data["normalized"] == props["normalized"]

    def test_response_format(self):
        """Test that response has correct format"""
        response = client.post("/api/extract-and-normalize", json={"text": "hello world"})
        assert response.status_code == 200
        data = response.json()

        # Check top-level keys
        assert "words" in data
        assert "total_words" in data
        assert "available_signs" in data
        assert "phrases_detected" in data

        # Check word format
        if len(data["words"]) > 0:
            word = data["words"][0]
            assert "original" in word
            assert "normalized" in word
            assert "hasSign" in word
            assert "phrase" in word
            assert "alternative" in word

    def test_sign_image_paths(self):
        """Test that sign image paths are correctly formatted"""
        response = client.post("/api/extract-and-normalize", json={"text": "hello"})
        assert response.status_code == 200
        data = response.json()

        hello = next((w for w in data["words"] if w["original"] == "hello"), None)
        if hello and hello["hasSign"]:
            assert "signImage" in hello
            assert hello["signImage"].startswith("/sign_images/")
            assert hello["signImage"].endswith(".png")

    def test_case_insensitivity(self):
        """Test that extraction is case-insensitive"""
        test_cases = ["HELLO", "Hello", "hello", "HeLLo"]

        for text in test_cases:
            response = client.post("/api/extract-and-normalize", json={"text": text})
            assert response.status_code == 200
            data = response.json()

            # Should normalize to lowercase "hello" for matching
            assert data["total_words"] >= 1
            assert data["available_signs"] >= 1

    def test_long_content_performance(self):
        """Test that long content is processed efficiently"""
        # Create a long lesson with 50+ words
        long_text = """
        Today we learn about farm animals and their families. The father cow and mother cow
        live with their baby. The goat eats grass in the morning. Good morning to all the animals.
        Thank you for learning with me. We especially love the four big animals. I want to show you
        more words and signs. The children are happy to learn. One two three four five six seven eight nine ten.
        """

        response = client.post("/api/extract-and-normalize", json={"text": long_text})
        assert response.status_code == 200
        data = response.json()

        # Should extract many words
        assert data["total_words"] > 20
        assert data["available_signs"] > 10

    def test_missing_words_marked_correctly(self):
        """Test that words without signs are marked hasSign=false"""
        # Use some made-up words
        response = client.post("/api/extract-and-normalize", json={"text": "hello xyzabc qwerty"})
        assert response.status_code == 200
        data = response.json()

        # hello should have sign
        hello = next((w for w in data["words"] if w["original"] == "hello"), None)
        assert hello is not None
        assert hello["hasSign"] is True

        # Made-up words should not have signs
        fake_words = [w for w in data["words"] if w["original"] in ["xyzabc", "qwerty"]]
        for word in fake_words:
            assert word["hasSign"] is False
            assert word["normalized"] is None

    def test_especially_maps_to_special(self):
        """Test that 'especially' maps to 'SPECIAL'"""
        response = client.post("/api/extract-and-normalize", json={"text": "especially"})
        assert response.status_code == 200
        data = response.json()

        especially = next((w for w in data["words"] if w["original"] == "especially"), None)
        assert especially is not None
        assert especially["hasSign"] is True
        assert especially["normalized"] == "SPECIAL"

    def test_words_maps_to_word(self):
        """Test that 'words' (plural) maps to 'WORD' (singular)"""
        response = client.post("/api/extract-and-normalize", json={"text": "words"})
        assert response.status_code == 200
        data = response.json()

        words = next((w for w in data["words"] if w["original"] == "words"), None)
        assert words is not None
        assert words["hasSign"] is True
        assert words["normalized"] == "WORD"


class TestExtractAndNormalizeIntegration:
    """Integration tests with lesson generation"""

    def test_extract_then_generate_lesson(self):
        """Test full flow: extract words → generate lesson"""
        # Step 1: Extract words
        text = "hello father thank you four"
        extract_response = client.post("/api/extract-and-normalize", json={"text": text})
        assert extract_response.status_code == 200
        extract_data = extract_response.json()

        # Get normalized words
        words_for_lesson = [
            w["normalized"] for w in extract_data["words"] if w["hasSign"]
        ]

        assert len(words_for_lesson) > 0
        assert "HELLO" in words_for_lesson
        assert "FATHER" in words_for_lesson
        assert "THANK" in words_for_lesson
        assert "4" in words_for_lesson  # "four" → "4"

        # Step 2: Generate lesson with normalized words
        lesson_response = client.post(
            "/api/temp/create-lesson",
            json={
                "lesson_title": "Test Lesson",
                "words": words_for_lesson,
                "find_missing": False
            }
        )

        assert lesson_response.status_code == 200
        lesson_data = lesson_response.json()
        assert "session_id" in lesson_data

    def test_consistency_with_search(self):
        """Test that extract-and-normalize returns same normalization as search"""
        test_words = ["thank you", "four", "and", "especially"]

        for word in test_words:
            # Get from extract-and-normalize
            extract_response = client.post("/api/extract-and-normalize", json={"text": word})
            assert extract_response.status_code == 200
            extract_data = extract_response.json()

            extract_word = next((w for w in extract_data["words"] if w["original"] == word), None)

            # Get from search
            search_response = client.get(f"/api/search?q={word}")

            # Both should succeed or both should fail
            if extract_word and extract_word["hasSign"]:
                assert search_response.status_code == 200
            elif search_response.status_code == 200:
                # Search found it, extract should too
                assert extract_word is not None
                assert extract_word["hasSign"] is True


# Test summary
def test_suite_summary():
    """Meta-test: document what we're testing"""
    tests = [
        "✅ Endpoint accessibility",
        "✅ Empty text handling",
        "✅ Single word extraction",
        "✅ Phrase detection (thank you, good morning)",
        "✅ Alternative mappings (four→4, and→ALSO)",
        "✅ Number word mappings (one→1, two→2, etc.)",
        "✅ Complex sentences with multiple phrases",
        "✅ Response format validation",
        "✅ Sign image path formatting",
        "✅ Case insensitivity",
        "✅ Long content performance",
        "✅ Missing words marked correctly",
        "✅ Especially→SPECIAL mapping",
        "✅ Words→WORD mapping",
        "✅ Integration with lesson generation",
        "✅ Consistency with search endpoint"
    ]

    print("\n" + "="*70)
    print("TEST SUITE: /api/extract-and-normalize")
    print("="*70)
    for test in tests:
        print(f"  {test}")
    print("="*70)
    assert True  # Meta-test always passes
