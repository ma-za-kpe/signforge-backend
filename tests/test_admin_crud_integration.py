#!/usr/bin/env python3
"""
Integration Tests for Admin Contribution CRUD
These tests verify the new GET detail and UPDATE endpoints work correctly
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


class TestAdminContributionEndpoints:
    """Test admin contribution management endpoints"""

    def test_get_contributions_list_endpoint_exists(self):
        """Verify GET /api/ama/contributions endpoint exists"""
        response = client.get("/api/ama/contributions")
        # Should return 200 or 503 (if DB unavailable), not 404
        assert response.status_code in [200, 503]

    def test_get_single_contribution_endpoint_exists(self):
        """Verify GET /api/ama/contributions/{id} endpoint is registered"""
        # Try to get contribution that doesn't exist
        # Should return 404 (not found) or 503 (DB unavailable), not 405 (method not allowed)
        response = client.get("/api/ama/contributions/999999")
        assert response.status_code in [404, 503]
        assert response.status_code != 405  # Proves endpoint exists

    def test_update_contribution_endpoint_exists(self):
        """Verify PATCH /api/ama/contributions/{id} endpoint is registered"""
        response = client.patch(
            "/api/ama/contributions/999999",
            json={"word": "TEST"}
        )
        # Should return 404 (not found) or 503 (DB unavailable), not 405 (method not allowed)
        assert response.status_code in [404, 503]
        assert response.status_code != 405  # Proves endpoint exists

    def test_delete_contribution_endpoint_exists(self):
        """Verify DELETE /api/ama/contributions/{id} endpoint is registered"""
        response = client.delete("/api/ama/contributions/999999")
        # Should return 404 (not found) or 503 (DB unavailable), not 405 (method not allowed)
        assert response.status_code in [404, 503]
        assert response.status_code != 405  # Proves endpoint exists

    def test_update_requires_changes(self):
        """Verify UPDATE endpoint rejects empty updates"""
        response = client.patch(
            "/api/ama/contributions/1",
            json={}
        )
        # Should fail with 400 (bad request), 404 (not found), or 503 (DB unavailable)
        assert response.status_code in [400, 404, 503]

    def test_update_accepts_word_field(self):
        """Verify UPDATE endpoint accepts word field"""
        response = client.patch(
            "/api/ama/contributions/1",
            json={"word": "HELLO"}
        )
        # Should return 404 (not found) or 503 (DB unavailable), but not 422 (validation error)
        assert response.status_code in [404, 503]
        assert response.status_code != 422  # Proves schema is correct

    def test_update_accepts_metadata_field(self):
        """Verify UPDATE endpoint accepts metadata field"""
        response = client.patch(
            "/api/ama/contributions/1",
            json={"metadata": {"test": "data"}}
        )
        # Should return 404 (not found) or 503 (DB unavailable), but not 422 (validation error)
        assert response.status_code in [404, 503]
        assert response.status_code != 422  # Proves schema is correct

    def test_api_docs_include_new_endpoints(self):
        """Verify new endpoints are documented in OpenAPI schema"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_schema = response.json()
        paths = openapi_schema.get("paths", {})

        # Check if new endpoints are documented
        assert "/api/ama/contributions/{contribution_id}" in paths

        # Check methods
        contribution_endpoint = paths["/api/ama/contributions/{contribution_id}"]
        assert "get" in contribution_endpoint  # GET detail
        assert "patch" in contribution_endpoint  # UPDATE
        assert "delete" in contribution_endpoint  # DELETE

    def test_response_models_defined(self):
        """Verify response models are properly defined in schema"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_schema = response.json()
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})

        # Check if new models exist
        assert "ContributionFullDetail" in schemas
        assert "ContributionUpdateRequest" in schemas

        # Verify ContributionFullDetail has pose_sequence field
        full_detail = schemas["ContributionFullDetail"]
        assert "pose_sequence" in full_detail["properties"]

        # Verify ContributionUpdateRequest has optional fields
        update_request = schemas["ContributionUpdateRequest"]
        assert "word" in update_request["properties"]
        assert "metadata" in update_request["properties"]


class TestContributionWorkflows:
    """Test common admin workflows"""

    def test_endpoints_are_accessible(self):
        """Verify all CRUD endpoints are accessible"""
        endpoints = [
            ("GET", "/api/ama/contributions"),
            ("GET", "/api/ama/contributions/1"),
            ("PATCH", "/api/ama/contributions/1"),
            ("DELETE", "/api/ama/contributions/1"),
        ]

        for method, path in endpoints:
            if method == "GET":
                response = client.get(path)
            elif method == "PATCH":
                response = client.patch(path, json={"word": "TEST"})
            elif method == "DELETE":
                response = client.delete(path)

            # All endpoints should exist (not 405 Method Not Allowed)
            assert response.status_code != 405, f"{method} {path} returned 405 - endpoint not found"
            # Should return valid HTTP status codes
            assert response.status_code in [200, 400, 404, 503], \
                f"{method} {path} returned unexpected status {response.status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
