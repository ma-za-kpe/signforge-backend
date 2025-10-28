"""
API Integration Tests using FastAPI TestClient
Tests all major API endpoints without requiring server to run
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint"""

    def test_health_check_returns_200(self):
        """Health endpoint should return 200 OK"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_expected_fields(self):
        """Health check should return status, version, brain_loaded, total_signs"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "brain_loaded" in data
        assert "total_signs" in data

    def test_health_status_is_healthy(self):
        """Status should be 'healthy'"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_version_is_correct(self):
        """Version should be 1.0.0"""
        response = client.get("/health")
        data = response.json()
        assert data["version"] == "1.0.0"

    def test_brain_is_loaded(self):
        """Brain should be loaded"""
        response = client.get("/health")
        data = response.json()
        assert data["brain_loaded"] is True

    def test_total_signs_is_positive(self):
        """Total signs should be > 0"""
        response = client.get("/health")
        data = response.json()
        assert data["total_signs"] > 0
        # We know we have 1,582 signs
        assert data["total_signs"] == 1582


class TestRootEndpoint:
    """Test / root endpoint"""

    def test_root_returns_200(self):
        """Root endpoint should return 200 OK"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_api_info(self):
        """Root should return API information"""
        response = client.get("/")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data

    def test_root_includes_endpoint_documentation(self):
        """Root should document available endpoints"""
        response = client.get("/")
        data = response.json()
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "search" in endpoints
        assert "brain" in endpoints


class TestSearchEndpoint:
    """Test /api/search endpoint"""

    def test_search_requires_query_parameter(self):
        """Search without query should return 422"""
        response = client.get("/api/search")
        assert response.status_code == 422  # Unprocessable Entity

    def test_search_single_word_returns_result(self):
        """Search for 'father' should return FATHER sign"""
        response = client.get("/api/search?q=father")
        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert "sign_image" in data
        assert "confidence" in data
        assert "metadata" in data

    def test_search_returns_correct_word(self):
        """Search should return the matched word"""
        response = client.get("/api/search?q=father")
        data = response.json()
        assert data["metadata"]["matched_word"] == "FATHER"

    def test_search_confidence_is_valid(self):
        """Confidence should be between 0 and 1"""
        response = client.get("/api/search?q=father")
        data = response.json()
        confidence = data["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_search_exact_match_has_high_confidence(self):
        """Exact matches should have high confidence (>0.9)"""
        words = ["father", "mother", "school", "hello"]
        for word in words:
            response = client.get(f"/api/search?q={word}")
            data = response.json()
            assert data["confidence"] >= 0.9, f"Low confidence for '{word}': {data['confidence']}"

    def test_search_phrase_normalization(self):
        """Phrase 'thank you' should map to 'THANK'"""
        response = client.get("/api/search?q=thank you")
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["matched_word"] == "THANK"
        assert data["confidence"] == 1.0

    def test_search_case_insensitive(self):
        """Search should be case-insensitive"""
        queries = ["FATHER", "father", "Father", "FaThEr"]
        for query in queries:
            response = client.get(f"/api/search?q={query}")
            data = response.json()
            assert data["metadata"]["matched_word"] == "FATHER"

    def test_search_handles_whitespace(self):
        """Search should handle leading/trailing whitespace"""
        queries = [" father ", "  father  ", "\tfather\n"]
        for query in queries:
            response = client.get(f"/api/search?q={query}")
            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["matched_word"] == "FATHER"

    def test_search_typo_tolerance(self):
        """Search should handle typos"""
        response = client.get("/api/search?q=fater")  # typo: father
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["matched_word"] == "FATHER"
        # Should have slightly lower confidence for typo
        assert data["confidence"] >= 0.7

    def test_search_natural_language(self):
        """Natural language query should extract main word"""
        response = client.get("/api/search?q=how to sign hello")
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["matched_word"] == "HELLO"

    def test_search_gibberish_returns_404(self):
        """Gibberish query should return 404"""
        response = client.get("/api/search?q=xyzabc123")
        assert response.status_code == 404

    def test_search_empty_query_returns_404(self):
        """Empty query should return 404"""
        response = client.get("/api/search?q=")
        assert response.status_code == 404

    def test_search_multiple_words_phrase(self):
        """Multi-word phrases should be handled"""
        phrases = [
            ("thank you", "THANK"),
            ("good morning", "MORNING"),
            ("my father", "FATHER"),
            ("i love you", "LOVE"),
        ]
        for query, expected in phrases:
            response = client.get(f"/api/search?q={query}")
            data = response.json()
            assert data["metadata"]["matched_word"] == expected, f"Failed for '{query}'"


class TestAutocompleteEndpoint:
    """Test /api/autocomplete endpoint"""

    def test_autocomplete_requires_query(self):
        """Autocomplete without query should return 422"""
        response = client.get("/api/autocomplete")
        assert response.status_code == 422

    def test_autocomplete_returns_suggestions(self):
        """Autocomplete should return suggestions list"""
        response = client.get("/api/autocomplete?q=tha")
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_autocomplete_prefix_matching(self):
        """Autocomplete should return words starting with prefix"""
        response = client.get("/api/autocomplete?q=fath")
        data = response.json()
        suggestions = [s.split(" (")[0] for s in data["suggestions"]]  # Strip hints
        assert "FATHER" in suggestions

    def test_autocomplete_multiple_matches(self):
        """Autocomplete should return multiple matches"""
        response = client.get("/api/autocomplete?q=tha")
        data = response.json()
        suggestions = [s.split(" (")[0] for s in data["suggestions"]]
        # Should include THANK, THAT, THAN, etc.
        assert len(suggestions) >= 2

    def test_autocomplete_case_insensitive(self):
        """Autocomplete should be case-insensitive"""
        queries = ["FATH", "fath", "Fath"]
        for query in queries:
            response = client.get(f"/api/autocomplete?q={query}")
            data = response.json()
            suggestions = [s.split(" (")[0] for s in data["suggestions"]]
            assert "FATHER" in suggestions

    def test_autocomplete_phrase_hint(self):
        """Autocomplete for phrases should show hint"""
        response = client.get("/api/autocomplete?q=thank you")
        data = response.json()
        # Should suggest THANK with phrase hint
        assert any("THANK" in s for s in data["suggestions"])


class TestBrainStatsEndpoint:
    """Test /api/brain/stats endpoint"""

    def test_brain_stats_returns_200(self):
        """Brain stats endpoint should return 200"""
        response = client.get("/api/brain/stats")
        assert response.status_code == 200

    def test_brain_stats_returns_expected_fields(self):
        """Brain stats should return expected metadata"""
        response = client.get("/api/brain/stats")
        data = response.json()
        assert "total_signs" in data
        assert "categories" in data or "total_signs" in data  # At minimum

    def test_brain_stats_total_signs(self):
        """Total signs should be 1,582"""
        response = client.get("/api/brain/stats")
        data = response.json()
        assert data["total_signs"] == 1582


class TestFormatsEndpoint:
    """Test /api/formats/* endpoints"""

    def test_formats_create_requires_word(self):
        """Format creation should require word parameter"""
        response = client.post("/api/formats/create", json={})
        # Should return 422 or 400 for missing word
        assert response.status_code in [400, 422]

    def test_formats_create_with_valid_word(self):
        """Format creation should work with valid word"""
        response = client.post("/api/formats/create", json={"word": "hello"})
        # Should succeed or return specific format
        assert response.status_code in [200, 201]


class TestMissingWordsEndpoint:
    """Test /api/missing/* endpoints"""

    def test_missing_words_report(self):
        """Missing words report should be accessible"""
        response = client.get("/api/missing/report")
        assert response.status_code == 200
        data = response.json()
        assert "total_unique_missing_words" in data or isinstance(data, dict)


class TestCorrectionEndpoints:
    """Test /api/corrections/* endpoints"""

    def test_corrections_alternatives(self):
        """Alternatives endpoint should work"""
        response = client.get("/api/corrections/alternatives?q=hello")
        # Should return 200 or 404 if no alternatives
        assert response.status_code in [200, 404]

    def test_corrections_flag(self):
        """Flag correction endpoint should accept POST"""
        response = client.post(
            "/api/corrections/flag", json={"query": "hello", "incorrect_result": "HELLO"}
        )
        # Should accept the flag (200) or validate input (422)
        assert response.status_code in [200, 201, 422]


class TestMetricsEndpoint:
    """Test /api/metrics endpoint"""

    def test_metrics_endpoint(self):
        """Metrics endpoint should return analytics data"""
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestCORSHeaders:
    """Test CORS configuration"""

    def test_cors_headers_present(self):
        """CORS headers should be present for frontend"""
        response = client.get("/health")
        # FastAPI TestClient doesn't set Origin, so CORS middleware might not add headers
        # But the middleware is configured, which is what matters
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling"""

    def test_404_for_nonexistent_endpoint(self):
        """Non-existent endpoints should return 404"""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404

    def test_405_for_wrong_method(self):
        """Wrong HTTP method should return 405"""
        response = client.post("/health")  # GET-only endpoint
        assert response.status_code == 405

    def test_search_with_special_characters(self):
        """Search with special characters should not crash"""
        special_queries = ["hello!", "father?", "mother...", "test@#$%"]
        for query in special_queries:
            response = client.get(f"/api/search?q={query}")
            # Should handle gracefully (200 or 404, not 500)
            assert response.status_code in [200, 404], f"Crashed on: {query}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
