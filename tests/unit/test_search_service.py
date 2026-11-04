"""
Unit tests for app/services/search_service.py.
"""

import pytest
from app.services.search_service import (
    SearchService,
    get_search_service,
    reset_search_service,
    SearchServiceError,
    IndexLoadError,
    DataFileError,
    InvalidQueryError,
)


@pytest.fixture(scope="module")
def search_service():
    """Module-level fixture to share loaded SearchService instance across tests."""
    service = SearchService()
    yield service


def test_search_service_initialization(search_service):
    """Verify that SearchService initializes and loads index and documents."""
    assert search_service.model is not None
    assert search_service.index is not None
    assert search_service.index.ntotal >= 1000
    assert len(search_service.documents) >= 1000


def test_missing_index_file():
    """Verify that missing FAISS index raises IndexLoadError."""
    with pytest.raises(IndexLoadError):
        SearchService(faiss_index_path="data/non_existent_index.bin")


def test_missing_data_file():
    """Verify that missing documents.json raises DataFileError."""
    with pytest.raises(DataFileError):
        SearchService(data_path="data/non_existent_documents.json")


def test_search_documents_valid_query(search_service):
    """Verify search_documents returns expected list structure with correct keys."""
    results = search_service.search_documents("machine learning models", top_k=5)
    assert isinstance(results, list)
    assert len(results) == 5

    for item in results:
        assert "id" in item
        assert "text_snippet" in item
        assert "score" in item
        assert isinstance(item["id"], str)
        assert isinstance(item["text_snippet"], str)
        assert isinstance(item["score"], float)
        assert len(item["text_snippet"]) > 0
        assert item["score"] >= 0.0


def test_search_documents_custom_top_k(search_service):
    """Verify that top_k parameter is respected."""
    results_1 = search_service.search_documents("cloud computing", top_k=1)
    assert len(results_1) == 1

    results_3 = search_service.search_documents("cloud computing", top_k=3)
    assert len(results_3) == 3


def test_search_documents_empty_query(search_service):
    """Verify that empty string raises InvalidQueryError."""
    with pytest.raises(InvalidQueryError):
        search_service.search_documents("")


def test_search_documents_whitespace_query(search_service):
    """Verify that whitespace-only query raises InvalidQueryError."""
    with pytest.raises(InvalidQueryError):
        search_service.search_documents("   ")


def test_search_documents_invalid_type(search_service):
    """Verify that non-string query raises InvalidQueryError."""
    with pytest.raises(InvalidQueryError):
        search_service.search_documents(None)  # type: ignore


def test_singleton_get_search_service():
    """Verify that get_search_service returns a singleton instance."""
    reset_search_service()
    service1 = get_search_service()
    service2 = get_search_service()
    assert service1 is service2


def test_semantic_relevance(search_service):
    """
    Verify semantic quality: Domain-specific queries should rank
    domain-relevant terms among the top results.
    """
    tech_results = search_service.search_documents("Kubernetes container orchestration", top_k=3)
    assert any("Kubernetes" in item["text_snippet"] or "container" in item["text_snippet"].lower() for item in tech_results)

    health_results = search_service.search_documents("telemetry remote cardiac monitoring", top_k=3)
    assert any("cardiac" in item["text_snippet"].lower() or "telemetry" in item["text_snippet"].lower() or "health" in item["text_snippet"].lower() for item in health_results)

