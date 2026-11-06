"""
Integration tests for FastAPI application endpoints (app/main.py).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client):
    """
    Test GET /health returns 200 OK and status 'ok'.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint(client):
    """
    Test GET / returns 200 OK and API metadata.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Semantic Search API"
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/health"
    assert data["endpoints"]["search"] == "/search"


def test_search_endpoint_valid_query(client):
    """
    Test POST /search with a valid query returns 200 OK and structured results.
    """
    payload = {"query": "vector database and semantic embeddings"}
    response = client.post("/search", json=payload)

    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) > 0

    for item in results:
        assert "id" in item
        assert "text_snippet" in item
        assert "score" in item
        assert isinstance(item["id"], str)
        assert isinstance(item["text_snippet"], str)
        assert isinstance(item["score"], float)
        assert len(item["text_snippet"]) > 0


def test_search_endpoint_short_query_returns_400(client):
    """
    Test POST /search with query length < 3 returns 400 Bad Request.
    """
    payload = {"query": "hi"}
    response = client.post("/search", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "at least 3 characters long" in data["detail"]


def test_search_endpoint_empty_query_returns_400(client):
    """
    Test POST /search with empty query string returns 400 Bad Request.
    """
    payload = {"query": ""}
    response = client.post("/search", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Query must not be empty" in data["detail"]


def test_search_endpoint_whitespace_query_returns_400(client):
    """
    Test POST /search with whitespace-only query returns 400 Bad Request.
    """
    payload = {"query": "   "}
    response = client.post("/search", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_search_endpoint_missing_payload_field_returns_422(client):
    """
    Test POST /search with missing 'query' field returns 422 Unprocessable Entity.
    """
    payload = {}
    response = client.post("/search", json=payload)

    assert response.status_code == 422


def test_search_endpoint_invalid_type_returns_422(client):
    """
    Test POST /search with integer instead of string returns 422.
    """
    payload = {"query": 12345}
    response = client.post("/search", json=payload)

    assert response.status_code == 422


def test_search_semantic_accuracy(client):
    """
    Test semantic relevance: searching for finance fraud detection retrieves
    financial / fraud domain content in top results.
    """
    payload = {"query": "credit card fraud detection with machine learning"}
    response = client.post("/search", json=payload)

    assert response.status_code == 200
    results = response.json()
    top_snippet = results[0]["text_snippet"].lower()
    all_snippets = " ".join([r["text_snippet"].lower() for r in results])

    assert "fraud" in all_snippets or "credit" in all_snippets or "transaction" in all_snippets


def test_search_endpoint_with_custom_top_k(client):
    """
    Test POST /search with explicit top_k in payload returns the requested count.
    """
    payload = {"query": "vector similarity search", "top_k": 2}
    response = client.post("/search", json=payload)

    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2


def test_search_endpoint_respects_top_k_env(client, monkeypatch):
    """
    Test POST /search dynamically respects the TOP_K_RESULTS environment variable.
    """
    monkeypatch.setenv("TOP_K_RESULTS", "3")
    payload = {"query": "vector similarity search"}
    response = client.post("/search", json=payload)

    assert response.status_code == 200
    results = response.json()
    assert len(results) == 3


def test_search_endpoint_handles_service_error_500(client):
    """
    Test that SearchService errors are caught and return a clean HTTP 500 response.
    """
    from app.services.search_service import get_search_service, SearchServiceError

    class FaultyService:
        def search_documents(self, query: str, top_k: int = 5):
            raise SearchServiceError("Simulated vector index failure")

    app.dependency_overrides[get_search_service] = lambda: FaultyService()

    try:
        response = client.post("/search", json={"query": "test system failure"})
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    finally:
        app.dependency_overrides.clear()

