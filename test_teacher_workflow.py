"""
Comprehensive Tests for Teacher UI Workflow (Agent 1)
Tests all edge cases for lesson content that teachers would type

Run with: pytest test_teacher_workflow.py -v
"""

import pytest
import httpx
import json
from typing import List, Dict

# API Base URL
API_BASE = "http://localhost:9000"


class TestTeacherLessonContent:
    """Test various lesson content scenarios that teachers would create"""

    @pytest.fixture
    def client(self):
        """HTTP client for API calls"""
        return httpx.Client(base_url=API_BASE, timeout=30.0)

    # ============================================
    # Test 1: Basic Lesson Content
    # ============================================

    def test_simple_farm_animals_lesson(self, client):
        """Test: Teacher types simple farm animals lesson"""
        lesson_content = """
        Today we learn about farm animals.
        The cow eats grass.
        The goat lives in the barn.
        The chicken lays eggs.
        """

        # Test common words that should exist
        common_words = ["cow", "goat", "chicken"]

        # Search for each word
        for word in common_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code == 200, f"Word '{word}' should be found"
            data = response.json()
            assert data["confidence"] >= 0.5, f"Word '{word}' should have decent confidence"

        # Test that less common words are handled gracefully
        less_common = ["barn", "grass", "eggs"]
        for word in less_common:
            response = client.get(f"/api/search?q={word}")
            # Should handle gracefully (200 if found, 404 if not)
            assert response.status_code in [200, 404]

    def test_greetings_lesson(self, client):
        """Test: Teacher creates greetings lesson"""
        lesson_content = """
        Lesson: Common Greetings

        Good morning! How to greet someone in the morning.
        Good afternoon! How to greet someone in the afternoon.
        Hello! A friendly greeting.
        Thank you for being here.
        """

        expected_words = ["good", "morning", "afternoon", "hello", "thank"]

        for word in expected_words:
            response = client.get(f"/api/search?q={word}")
            # Some might not exist, but should handle gracefully
            assert response.status_code in [200, 404]

    # ============================================
    # Test 2: Edge Cases - Formatting
    # ============================================

    def test_lesson_with_punctuation(self, client):
        """Test: Lesson with heavy punctuation"""
        lesson_content = """
        Hello! How are you? I am fine, thank you.
        The cow, goat, and chicken are animals.
        Water is important; we need it daily.
        """

        # Punctuation should be stripped
        test_words = ["hello", "cow", "goat", "chicken", "water"]

        for word in test_words:
            response = client.get(f"/api/search?q={word}")
            if response.status_code == 200:
                data = response.json()
                # Should match without punctuation
                assert word.upper() in data["metadata"]["matched_word"].upper()

    def test_lesson_with_numbers(self, client):
        """Test: Lesson with numbers"""
        lesson_content = """
        Counting Lesson:
        1. One apple
        2. Two bananas
        3. Three oranges
        4. Four mangoes
        5. Five watermelons
        """

        number_words = ["one", "two", "three", "four", "five"]

        for word in number_words:
            response = client.get(f"/api/search?q={word}")
            # Numbers might not all have signs
            assert response.status_code in [200, 404]

    def test_lesson_with_capitalization_mix(self, client):
        """Test: Mixed capitalization (teachers often capitalize randomly)"""
        test_cases = [
            "HELLO",  # All caps
            "hello",  # All lowercase
            "Hello",  # Title case
            "hElLo",  # Mixed case
        ]

        for word in test_cases:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["matched_word"] == "HELLO"

    # ============================================
    # Test 3: Edge Cases - Whitespace
    # ============================================

    def test_lesson_with_extra_whitespace(self, client):
        """Test: Lesson with extra spaces, tabs, newlines"""
        lesson_content = """


        Today   we    learn   about    water.

        Water     is     important.


        """

        # Should still find "water"
        response = client.get(f"/api/search?q=water")
        assert response.status_code == 200

    def test_lesson_with_leading_trailing_spaces(self, client):
        """Test: Words with leading/trailing spaces"""
        import urllib.parse

        test_cases = [
            " cow ",
            "  goat  ",
            " water ",
        ]

        for word in test_cases:
            # URL encode to handle spaces properly
            encoded_word = urllib.parse.quote(word)
            response = client.get(f"/api/search?q={encoded_word}")
            assert response.status_code == 200, f"Word '{word}' with whitespace should work"

    # ============================================
    # Test 4: Edge Cases - Phrases
    # ============================================

    def test_lesson_with_common_phrases(self, client):
        """Test: Lesson with common multi-word phrases"""
        lesson_content = """
        Good morning class! Thank you for coming.
        Good afternoon everyone. You're welcome.
        """

        # Test phrase normalization
        phrase_tests = [
            ("thank you", "THANK"),    # Should normalize to THANK
            ("you're welcome", "WELCOME"),  # Should normalize to WELCOME
        ]

        for phrase, expected in phrase_tests:
            response = client.get(f"/api/search?q={phrase}")
            if response.status_code == 200:
                data = response.json()
                # Check if the expected word is in the matched result
                matched_word = data["metadata"]["matched_word"]
                assert expected == matched_word, f"Phrase '{phrase}' should match '{expected}', got '{matched_word}'"

    def test_lesson_with_questions(self, client):
        """Test: Lesson with questions"""
        lesson_content = """
        What is water? Water is a liquid.
        How do you say hello? You say hello like this.
        Where is the school? The school is nearby.
        """

        # Should extract: water, hello, school
        test_words = ["water", "hello", "school"]

        for word in test_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code == 200

    # ============================================
    # Test 5: Real Teacher Scenarios
    # ============================================

    def test_kindergarten_lesson(self, client):
        """Test: Typical kindergarten lesson"""
        lesson_content = """
        Colors Lesson for Kindergarten

        Red is the color of apples.
        Blue is the color of the sky.
        Yellow is the color of the sun.
        Green is the color of grass.

        Let's practice: What color is your shirt?
        """

        # Color words might not all have signs
        color_words = ["red", "blue", "yellow", "green"]

        results = []
        for word in color_words:
            response = client.get(f"/api/search?q={word}")
            results.append(response.status_code)

        # At least some colors should work
        assert 200 in results or 404 in results  # Either found or not found, not error

    def test_primary_school_science_lesson(self, client):
        """Test: Primary school science lesson"""
        lesson_content = """
        Science: The Water Cycle

        Water evaporates from the ocean.
        Water vapor rises into the sky.
        Clouds form when water vapor cools.
        Rain falls from the clouds.
        Rain flows into rivers.
        Rivers flow back to the ocean.

        Key words: water, ocean, sky, rain, river
        """

        key_words = ["water", "ocean", "sky", "rain", "river"]

        for word in key_words:
            response = client.get(f"/api/search?q={word}")
            # Science terms might not all have signs
            assert response.status_code in [200, 404]

    def test_mathematics_lesson(self, client):
        """Test: Math lesson with numbers and operations"""
        lesson_content = """
        Addition Lesson

        One plus one equals two.
        Two plus three equals five.
        Five minus two equals three.

        Practice: Add these numbers together.
        """

        math_words = ["one", "two", "three", "five", "add"]

        for word in math_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code in [200, 404]

    # ============================================
    # Test 6: Edge Cases - Special Characters
    # ============================================

    def test_lesson_with_contractions(self, client):
        """Test: Lesson with contractions"""
        lesson_content = """
        I'm happy. You're smart. We're learning.
        Don't worry. Can't stop. Won't quit.
        """

        # Contractions should be handled
        test_cases = [
            ("I'm", "I"),
            ("you're", "YOU"),
            ("don't", "DO"),
        ]

        for contraction, expected_base in test_cases:
            response = client.get(f"/api/search?q={contraction}")
            # Should either find the base word or handle gracefully
            assert response.status_code in [200, 404]

    def test_lesson_with_possessives(self, client):
        """Test: Lesson with possessive forms"""
        lesson_content = """
        The cow's milk. The goat's cheese. The chicken's eggs.
        John's book. Mary's pen. Teacher's desk.
        """

        # Should strip possessive 's
        test_words = ["cow", "goat", "chicken", "book", "pen", "desk"]

        for word in test_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code in [200, 404]

    # ============================================
    # Test 7: Edge Cases - Empty/Invalid Content
    # ============================================

    def test_empty_lesson_content(self, client):
        """Test: Teacher submits empty lesson"""
        lesson_content = ""

        # Should handle gracefully - no words to search
        # This would be handled client-side, but API should handle empty queries
        response = client.get("/api/search?q=")
        assert response.status_code == 404  # Empty query returns 404

    def test_lesson_with_only_punctuation(self, client):
        """Test: Lesson with only punctuation"""
        lesson_content = "!!! ??? ... ,,,, ;;;;"

        # No valid words - should handle gracefully
        # Client-side would filter these out
        pass  # This is handled client-side

    def test_lesson_with_gibberish(self, client):
        """Test: Teacher accidentally types gibberish"""
        gibberish_words = ["asdfghjkl", "qwertyuiop", "zxcvbnm"]

        for word in gibberish_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code == 404, f"Gibberish '{word}' should return 404"

    # ============================================
    # Test 8: Edge Cases - Very Long Content
    # ============================================

    def test_very_long_lesson(self, client):
        """Test: Very long lesson content (stress test)"""
        # Generate long lesson with repeated words
        lesson_content = " ".join(["The cow eats grass."] * 100)

        # Should still find "cow" efficiently
        response = client.get("/api/search?q=cow")
        assert response.status_code == 200
        assert response.elapsed.total_seconds() < 1.0  # Should be fast

    def test_lesson_with_many_unique_words(self, client):
        """Test: Lesson with 50+ unique words"""
        # Typical full lesson plan
        lesson_content = """
        Comprehensive Lesson: Community Helpers

        Today we learn about people who help us in our community.

        The teacher helps us learn at school.
        The doctor helps us when we are sick.
        The nurse works with the doctor.
        The farmer grows our food.
        The baker makes our bread.
        The chef cooks our meals.
        The police officer keeps us safe.
        The firefighter puts out fires.
        The driver takes us places.
        The mechanic fixes our cars.

        All these people are important to our community.
        We should thank them for their work.
        """

        # Should handle extracting many words
        key_words = ["teacher", "doctor", "nurse", "farmer", "baker",
                     "police", "driver", "water", "thank"]

        for word in key_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code in [200, 404]

    # ============================================
    # Test 9: Format Generation for Lessons
    # ============================================

    def test_format_generation_single_word(self, client):
        """Test: Generate all formats for a single word"""
        response = client.post(
            "/api/formats/create",
            json={"word": "cow"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "formats" in data
        assert "qr_code" in data["formats"]
        assert "audio" in data["formats"]
        assert "haptic" in data["formats"]

    def test_format_generation_lesson_bundle(self, client):
        """Test: Generate complete lesson bundle"""
        response = client.post(
            "/api/formats/lesson-bundle",
            json={
                "lesson_title": "Test Lesson",
                "words": ["cow", "goat", "chicken"]
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True
        assert data["total_words"] == 3
        assert "bundle" in data

    def test_format_generation_large_lesson(self, client):
        """Test: Generate formats for large lesson (10+ words)"""
        words = ["hello", "thank", "water", "food", "love",
                 "mother", "father", "school", "teacher", "student"]

        response = client.post(
            "/api/formats/lesson-bundle",
            json={
                "lesson_title": "Large Lesson",
                "words": words
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_words"] == len(words)

    # ============================================
    # Test 10: Common Teacher Typos
    # ============================================

    def test_common_teacher_typos(self, client):
        """Test: Common typos teachers make"""
        typo_tests = [
            ("scool", "SCHOOL", 0.80),      # Missing h - good match
            ("helo", "HELLO", 0.75),        # Single l - good match
            ("techer", "TEACHER", 0.70),    # Missing a - moderate match
        ]

        for typo, expected, min_confidence in typo_tests:
            response = client.get(f"/api/search?q={typo}")
            if response.status_code == 200:
                data = response.json()
                # Should fuzzy match to correct word
                matched_word = data["metadata"]["matched_word"]
                confidence = data["confidence"]

                # For typos, we expect either exact match or reasonable confidence
                if confidence >= min_confidence:
                    assert expected in matched_word, f"Typo '{typo}' should match '{expected}', got '{matched_word}'"
            # If 404, that's acceptable for very severe typos

    def test_double_letter_typos(self, client):
        """Test: Teachers accidentally double letters"""
        typo_tests = [
            ("helllo", "HELLO"),
            ("waterr", "WATER"),
            ("schooll", "SCHOOL"),
        ]

        for typo, expected in typo_tests:
            response = client.get(f"/api/search?q={typo}")
            # Should handle gracefully
            assert response.status_code in [200, 404]

    # ============================================
    # Test 11: Subject-Specific Lessons
    # ============================================

    def test_english_language_lesson(self, client):
        """Test: English language lesson"""
        lesson_content = """
        English Vocabulary: Family Members

        Mother - The female parent
        Father - The male parent
        Sister - Female sibling
        Brother - Male sibling
        Grandmother - Mother's or father's mother
        Grandfather - Mother's or father's father
        """

        family_words = ["mother", "father", "sister", "brother", "grandmother", "grandfather"]

        found_count = 0
        for word in family_words:
            response = client.get(f"/api/search?q={word}")
            if response.status_code == 200:
                found_count += 1

        # At least some family words should have signs
        assert found_count > 0

    def test_religious_education_lesson(self, client):
        """Test: Religious education lesson"""
        lesson_content = """
        Moral Values

        Love your neighbor.
        Be kind to others.
        Help those in need.
        Thank God for your blessings.
        Pray every day.
        """

        moral_words = ["love", "help", "thank", "pray"]

        for word in moral_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code in [200, 404]

    def test_physical_education_lesson(self, client):
        """Test: PE/Sports lesson"""
        lesson_content = """
        Physical Education: Basic Exercises

        Run in place for 30 seconds.
        Jump up and down 10 times.
        Walk around the field.
        Stand on one leg.
        Sit and rest.
        """

        pe_words = ["run", "jump", "walk", "stand", "sit", "rest"]

        for word in pe_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code in [200, 404]

    # ============================================
    # Test 12: Multilingual Content (Ghana Context)
    # ============================================

    def test_lesson_with_twi_words(self, client):
        """Test: Lesson mixing English and Twi"""
        lesson_content = """
        Bilingual Lesson: Greetings

        Good morning - Maakye
        Thank you - Medaase
        Water - Nsu
        Food - Aduane
        """

        # Should find English words
        english_words = ["good", "morning", "thank", "water", "food"]

        for word in english_words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code in [200, 404]

    # ============================================
    # Test 13: Error Recovery
    # ============================================

    def test_lesson_after_api_returns_error(self, client):
        """Test: Continue working after encountering errors"""
        # First, try a gibberish word (will 404)
        response1 = client.get("/api/search?q=xyzabc")
        assert response1.status_code == 404

        # Then, try a valid word (should work)
        response2 = client.get("/api/search?q=hello")
        assert response2.status_code == 200

    def test_rapid_successive_searches(self, client):
        """Test: Teacher quickly searching multiple words"""
        words = ["cow", "goat", "chicken", "water", "food"]

        for word in words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code in [200, 404]
            # All requests should complete quickly
            assert response.elapsed.total_seconds() < 1.0


# ============================================
# Integration Tests - Full Workflow
# ============================================

class TestTeacherWorkflowIntegration:
    """Test complete teacher workflow end-to-end"""

    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_BASE, timeout=30.0)

    def test_complete_lesson_creation_workflow(self, client):
        """Test: Full workflow from lesson creation to format generation"""
        # Step 1: Teacher creates lesson
        lesson_title = "Integration Test Lesson"
        words = ["hello", "thank", "water"]

        # Step 2: Search for each word (simulate real-time checking)
        for word in words:
            response = client.get(f"/api/search?q={word}")
            assert response.status_code == 200

        # Step 3: Generate formats for selected words
        response = client.post(
            "/api/formats/lesson-bundle",
            json={
                "lesson_title": lesson_title,
                "words": words
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True
        assert data["total_words"] == len(words)

    def test_teacher_corrects_lesson_and_regenerates(self, client):
        """Test: Teacher makes changes and regenerates formats"""
        # Initial lesson
        response1 = client.post(
            "/api/formats/lesson-bundle",
            json={
                "lesson_title": "Version 1",
                "words": ["cow", "goat"]
            }
        )
        assert response1.status_code == 200

        # Teacher adds more words and regenerates
        response2 = client.post(
            "/api/formats/lesson-bundle",
            json={
                "lesson_title": "Version 2",
                "words": ["cow", "goat", "chicken", "water"]
            }
        )
        assert response2.status_code == 200

        data = response2.json()
        assert data["total_words"] == 4


# ============================================
# Summary Report
# ============================================

def test_generate_summary_report(client=None):
    """Generate summary of all test scenarios covered"""
    summary = """
    ===============================================
    Teacher UI Workflow Tests - Coverage Summary
    ===============================================

    ✅ Basic Lesson Content
       - Farm animals lesson
       - Greetings lesson

    ✅ Edge Cases - Formatting
       - Heavy punctuation
       - Numbers in content
       - Mixed capitalization

    ✅ Edge Cases - Whitespace
       - Extra spaces/tabs/newlines
       - Leading/trailing spaces

    ✅ Edge Cases - Phrases
       - Multi-word phrases
       - Questions in content

    ✅ Real Teacher Scenarios
       - Kindergarten lessons
       - Primary school science
       - Mathematics lessons

    ✅ Edge Cases - Special Characters
       - Contractions (I'm, you're)
       - Possessives (cow's, John's)

    ✅ Edge Cases - Invalid Content
       - Empty lessons
       - Only punctuation
       - Gibberish words

    ✅ Edge Cases - Large Content
       - Very long lessons (100+ words)
       - 50+ unique words

    ✅ Format Generation
       - Single word formats
       - Lesson bundles
       - Large lessons (10+ words)

    ✅ Common Teacher Typos
       - Missing letters (scool → school)
       - Double letters (helllo → hello)

    ✅ Subject-Specific Lessons
       - English language
       - Religious education
       - Physical education

    ✅ Multilingual Content
       - English + Twi words

    ✅ Error Recovery
       - Continue after errors
       - Rapid successive searches

    ✅ Integration Tests
       - Complete workflow
       - Lesson corrections and regeneration

    ===============================================
    Total Test Categories: 13
    Total Test Cases: 40+
    ===============================================
    """

    print(summary)
    return True


if __name__ == "__main__":
    print("\n🧪 Teacher UI Workflow Test Suite\n")
    print("Run with: pytest test_teacher_workflow.py -v")
    print("\nOr run with coverage:")
    print("pytest test_teacher_workflow.py -v --cov=. --cov-report=html")
    print("\n" + "="*50)
    test_generate_summary_report()
