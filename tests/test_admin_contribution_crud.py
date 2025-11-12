#!/usr/bin/env python3
"""
Test Suite for Admin Contribution CRUD Operations
Tests GET detail, UPDATE, and DELETE endpoints for contribution management
"""

import pytest
import json
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from database import get_db, Contribution, Word


# Mock database session
class MockDB:
    def __init__(self):
        self.contributions = []
        self.words = []
        self.committed = False
        self.refreshed = []

    def query(self, model):
        if model == Contribution:
            return MockContributionQuery(self.contributions)
        elif model == Word:
            return MockWordQuery(self.words)
        return MockQuery([])

    def add(self, item):
        if isinstance(item, Contribution):
            self.contributions.append(item)
        elif isinstance(item, Word):
            self.words.append(item)

    def commit(self):
        self.committed = True

    def refresh(self, item):
        self.refreshed.append(item)

    def delete(self, item):
        if isinstance(item, Contribution):
            self.contributions.remove(item)


class MockQuery:
    def __init__(self, items):
        self.items = items

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.items[0] if self.items else None

    def all(self):
        return self.items


class MockContributionQuery(MockQuery):
    pass


class MockWordQuery(MockQuery):
    pass


# Sample contribution data
SAMPLE_POSE_SEQUENCE = [
    [[0.5, 0.5, 0.0, 1.0] for _ in range(75)]  # 75 landmarks
    for _ in range(30)  # 30 frames
]


def create_mock_contribution(
    id=1,
    word="HELLO",
    user_id="test_user_123",
    quality_score=0.85,
    num_frames=30,
    duration=1.0,
    pose_sequence=None
):
    """Helper to create mock contribution"""
    contrib = Mock(spec=Contribution)
    contrib.id = id
    contrib.word = word
    contrib.user_id = user_id
    contrib.quality_score = quality_score
    contrib.num_frames = num_frames
    contrib.duration = duration
    contrib.created_at = datetime.utcnow()
    contrib.metadata = {"test": "data"}
    contrib.pose_sequence = pose_sequence or SAMPLE_POSE_SEQUENCE
    contrib.fps = 30.0
    contrib.total_frames = num_frames
    contrib.extracted_frames = num_frames
    contrib.frames_data = {
        "hand_visibility": 0.9,
        "motion_smoothness": 0.85,
        "frame_completeness": 0.95
    }
    return contrib


class TestGetContributionDetail:
    """Test GET /api/ama/contributions/{id} endpoint"""

    def test_get_contribution_success(self):
        """Test successfully retrieving contribution detail"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        # Override dependency
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/ama/contributions/1")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["word"] == "HELLO"
            assert data["quality_score"] == 0.85
            assert "pose_sequence" in data
            assert len(data["pose_sequence"]) == 30  # 30 frames
            assert len(data["pose_sequence"][0]) == 75  # 75 landmarks
        finally:
            app.dependency_overrides.clear()

    def test_get_contribution_not_found(self, monkeypatch):
        """Test 404 when contribution doesn't exist"""
        mock_db = MockDB()

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.get("/api/ama/contributions/999")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_contribution_includes_quality_breakdown(self, monkeypatch):
        """Test that quality breakdown is included in response"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.get("/api/ama/contributions/1")

        assert response.status_code == 200
        data = response.json()
        assert "quality_breakdown" in data
        if data["quality_breakdown"]:
            assert "hand_visibility" in data["quality_breakdown"]


class TestUpdateContribution:
    """Test PATCH /api/ama/contributions/{id} endpoint"""

    def test_update_word(self, monkeypatch):
        """Test updating contribution word"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.patch(
            "/api/ama/contributions/1",
            json={"word": "GOODBYE"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["word"] == "GOODBYE"
        assert mock_db.committed

    def test_update_metadata(self, monkeypatch):
        """Test updating contribution metadata"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        new_metadata = {"corrected": True, "verified_by": "admin"}
        response = client.patch(
            "/api/ama/contributions/1",
            json={"metadata": new_metadata}
        )

        assert response.status_code == 200
        assert mock_db.committed
        assert mock_contrib.metadata == new_metadata

    def test_update_both_word_and_metadata(self, monkeypatch):
        """Test updating both word and metadata simultaneously"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.patch(
            "/api/ama/contributions/1",
            json={
                "word": "THANKS",
                "metadata": {"source": "mobile", "verified": True}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["word"] == "THANKS"
        assert mock_db.committed

    def test_update_no_changes(self, monkeypatch):
        """Test 400 error when no updates provided"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.patch(
            "/api/ama/contributions/1",
            json={}
        )

        assert response.status_code == 400
        assert "no updates" in response.json()["detail"].lower()

    def test_update_not_found(self, monkeypatch):
        """Test 404 when updating non-existent contribution"""
        mock_db = MockDB()

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.patch(
            "/api/ama/contributions/999",
            json={"word": "TEST"}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_word_case_insensitive(self, monkeypatch):
        """Test that word is automatically uppercased"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.patch(
            "/api/ama/contributions/1",
            json={"word": "goodbye"}  # lowercase
        )

        assert response.status_code == 200
        data = response.json()
        assert data["word"] == "GOODBYE"  # uppercased


class TestDeleteContribution:
    """Test DELETE /api/ama/contributions/{id} endpoint"""

    def test_delete_contribution_success(self, monkeypatch):
        """Test successfully deleting a contribution"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.delete("/api/ama/contributions/1")

        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"].lower()
        assert data["word"] == "HELLO"
        assert data["user_id"] == "test_user_123"
        assert mock_db.committed

    def test_delete_not_found(self, monkeypatch):
        """Test 404 when deleting non-existent contribution"""
        mock_db = MockDB()

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        response = client.delete("/api/ama/contributions/999")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestIntegrationScenarios:
    """Test complete workflows"""

    def test_preview_then_delete_workflow(self, monkeypatch):
        """Test common workflow: preview contribution then delete"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        # 1. Preview contribution
        preview_response = client.get("/api/ama/contributions/1")
        assert preview_response.status_code == 200
        preview_data = preview_response.json()
        assert "pose_sequence" in preview_data

        # 2. Delete contribution
        delete_response = client.delete("/api/ama/contributions/1")
        assert delete_response.status_code == 200

    def test_preview_then_update_workflow(self, monkeypatch):
        """Test workflow: preview, then correct word label"""
        mock_contrib = create_mock_contribution()
        mock_db = MockDB()
        mock_db.contributions = [mock_contrib]

        def mock_get_db():
            yield mock_db

        monkeypatch.setattr("admin_ama.get_db", mock_get_db)

        # 1. Preview contribution
        preview_response = client.get("/api/ama/contributions/1")
        assert preview_response.status_code == 200

        # 2. Update word (admin noticed it was mislabeled)
        update_response = client.patch(
            "/api/ama/contributions/1",
            json={
                "word": "HELLO_FORMAL",
                "metadata": {"corrected": True, "original_word": "HELLO"}
            }
        )
        assert update_response.status_code == 200
        assert update_response.json()["word"] == "HELLO_FORMAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
