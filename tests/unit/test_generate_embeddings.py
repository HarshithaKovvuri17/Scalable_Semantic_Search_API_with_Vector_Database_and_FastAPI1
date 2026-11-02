"""
Unit tests for embedding generation and FAISS indexing pipeline (scripts/generate_embeddings.py).
"""

import json
from pathlib import Path
import numpy as np
import faiss
import pytest

from scripts.generate_embeddings import (
    generate_synthetic_documents,
    resolve_path,
)


def test_generate_synthetic_documents_custom_count():
    """Verify that generate_synthetic_documents returns requested count and expected fields."""
    count = 25
    docs = generate_synthetic_documents(count=count)

    assert isinstance(docs, list)
    assert len(docs) == count

    expected_categories = {"Technology", "Healthcare", "Finance", "E-commerce"}
    found_categories = set()
    found_ids = set()

    for doc in docs:
        assert "id" in doc
        assert "text" in doc
        assert "category" in doc
        assert isinstance(doc["id"], str)
        assert doc["id"].startswith("doc_")
        assert len(doc["text"]) > 20
        assert doc["category"] in expected_categories
        found_categories.add(doc["category"])
        found_ids.add(doc["id"])

    # Verify ID uniqueness and category diversity
    assert len(found_ids) == count
    assert found_categories == expected_categories


def test_generate_synthetic_documents_default_count():
    """Verify default document generation produces at least 1,000 documents per requirement."""
    docs = generate_synthetic_documents()
    assert len(docs) >= 1000

    unique_ids = {doc["id"] for doc in docs}
    assert len(unique_ids) == len(docs)


def test_faiss_index_flat_l2_initialization():
    """Verify FAISS IndexFlatL2 initialization, addition, and vector similarity search."""
    dimension = 384
    num_vectors = 50

    # Create synthetic float32 embeddings
    rng = np.random.default_rng(seed=42)
    synthetic_embeddings = rng.random((num_vectors, dimension), dtype=np.float32)

    # Initialize IndexFlatL2
    index = faiss.IndexFlatL2(dimension)
    assert index.d == dimension
    assert index.ntotal == 0

    # Add embeddings
    index.add(synthetic_embeddings)
    assert index.ntotal == num_vectors

    # Perform search
    query_vector = rng.random((1, dimension), dtype=np.float32)
    k = 5
    distances, indices = index.search(query_vector, k)

    assert distances.shape == (1, k)
    assert indices.shape == (1, k)
    # Distance in L2 space must be non-negative
    assert all(dist >= 0.0 for dist in distances[0])
    # Returned indices must be in valid range
    assert all(0 <= idx < num_vectors for idx in indices[0])


def test_faiss_index_persistence(tmp_path):
    """Verify FAISS index serialization to disk and re-loading."""
    dimension = 64
    index = faiss.IndexFlatL2(dimension)
    vectors = np.random.randn(20, dimension).astype(np.float32)
    index.add(vectors)

    temp_index_path = tmp_path / "test_faiss.bin"
    faiss.write_index(index, str(temp_index_path))
    assert temp_index_path.exists()

    loaded_index = faiss.read_index(str(temp_index_path))
    assert loaded_index.ntotal == 20
    assert loaded_index.d == dimension


def test_resolve_path_helper():
    """Verify resolve_path handles absolute and relative paths correctly."""
    rel = "data/documents.json"
    resolved = resolve_path(rel)
    assert resolved.is_absolute()
    assert str(resolved).endswith("data\\documents.json") or str(resolved).endswith("data/documents.json")

    abs_path = Path("C:/test/path.json")
    assert resolve_path(str(abs_path)) == abs_path
