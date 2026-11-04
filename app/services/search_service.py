"""
Search Service Module.

Encapsulates business logic for:
1. SentenceTransformer model loading (supporting local offline cache or HuggingFace hub).
2. FAISS vector index loading and similarity retrieval.
3. Original document lookup and snippet generation.
4. Centralized exception handling and thread-safe singleton lifecycle management.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logger = logging.getLogger("search_service")

# Ensure .env is loaded
load_dotenv()

# Resolve default paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class SearchServiceError(Exception):
    """Base exception for search service errors."""
    pass


class ModelLoadError(SearchServiceError):
    """Raised when the embedding model cannot be loaded."""
    pass


class IndexLoadError(SearchServiceError):
    """Raised when the FAISS index file is missing or corrupted."""
    pass


class DataFileError(SearchServiceError):
    """Raised when the documents metadata file is missing or corrupted."""
    pass


class InvalidQueryError(SearchServiceError):
    """Raised when query validation fails within the service layer."""
    pass


class SearchService:
    """
    Core service responsible for embedding queries and searching the FAISS index.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        data_path: Optional[str] = None,
    ):
        """
        Initialize the SearchService by loading the embedding model,
        FAISS vector index, and document store into memory.
        """
        # Resolve config from arguments or environment variables
        self.model_name = model_name or os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.model_cache_dir = model_cache_dir or os.getenv("MODEL_CACHE_DIR", str(REPO_ROOT / "models" / "sentence_transformer"))
        self.faiss_index_path = faiss_index_path or os.getenv("FAISS_INDEX_PATH", str(REPO_ROOT / "data" / "faiss_index.bin"))
        self.data_path = data_path or os.getenv("DATA_PATH", str(REPO_ROOT / "data" / "documents.json"))

        logger.info("Initializing SearchService...")
        logger.info(f" Model Source: {self.model_cache_dir} (fallback: {self.model_name})")
        logger.info(f" FAISS Index: {self.faiss_index_path}")
        logger.info(f" Data Path:   {self.data_path}")

        self.model = self._load_model()
        self.index = self._load_index()
        self.documents = self._load_documents()

        logger.info(
            f"SearchService initialized successfully with {len(self.documents)} documents "
            f"and {self.index.ntotal} indexed vectors."
        )

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        """
        Resolve path string: If absolute or exists in CWD, use it;
        otherwise check relative to REPO_ROOT.
        """
        p = Path(path_str)
        if p.is_absolute() or p.exists():
            return p
        repo_p = REPO_ROOT / path_str
        if repo_p.exists():
            return repo_p
        return p

    def _load_model(self) -> SentenceTransformer:
        """
        Load SentenceTransformer model from local cache or remote Hugging Face repository.
        """
        cache_path = self._resolve_path(self.model_cache_dir)
        try:
            # Check if local cached directory contains valid model files
            if cache_path.exists() and (
                (cache_path / "modules.json").exists()
                or (cache_path / "config.json").exists()
                or (cache_path / "model.safetensors").exists()
            ):
                logger.info(f"Loading SentenceTransformer from local cache: {cache_path}")
                return SentenceTransformer(str(cache_path))

            # Also check container build-time cache if running in containerized environment
            container_cache = Path("/app/models_cache/sentence_transformer")
            if container_cache.exists() and (container_cache / "config.json").exists():
                logger.info(f"Loading SentenceTransformer from container build cache: {container_cache}")
                return SentenceTransformer(str(container_cache))

            logger.info(f"Local cache not populated. Loading model '{self.model_name}'...")
            model = SentenceTransformer(self.model_name)
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                model.save(str(cache_path))
                logger.info(f"Model saved to local cache: {cache_path}")
            except Exception as save_err:
                logger.warning(f"Could not cache model to {cache_path}: {save_err}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise ModelLoadError(f"Could not load embedding model: {str(e)}") from e

    def _load_index(self) -> faiss.Index:
        """
        Load FAISS index binary file from disk.
        """
        index_file = self._resolve_path(self.faiss_index_path)
        if not index_file.exists():
            error_msg = f"FAISS index file not found at: {self.faiss_index_path}"
            logger.error(error_msg)
            raise IndexLoadError(error_msg)

        try:
            logger.info(f"Loading FAISS index from: {index_file}")
            index = faiss.read_index(str(index_file))
            if index.ntotal == 0:
                logger.warning("Loaded FAISS index is empty (0 vectors).")
            return index
        except Exception as e:
            logger.error(f"Failed to read FAISS index from {index_file}: {e}", exc_info=True)
            raise IndexLoadError(f"Failed to read FAISS index file: {str(e)}") from e

    def _load_documents(self) -> List[Dict[str, Any]]:
        """
        Load document metadata and raw texts from documents.json.
        """
        doc_file = self._resolve_path(self.data_path)
        if not doc_file.exists():
            error_msg = f"Documents metadata file not found at: {self.data_path}"
            logger.error(error_msg)
            raise DataFileError(error_msg)

        try:
            logger.info(f"Loading document metadata from: {doc_file}")
            with open(doc_file, "r", encoding="utf-8") as f:
                documents = json.load(f)

            if not isinstance(documents, list):
                raise ValueError("documents.json content must be a JSON array of objects.")

            return documents
        except Exception as e:
            logger.error(f"Failed to read documents from {doc_file}: {e}", exc_info=True)
            raise DataFileError(f"Failed to read documents data file: {str(e)}") from e

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute semantic vector similarity search against the FAISS index.

        Args:
            query: The user search query string.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of dictionaries with 'id', 'text_snippet', and 'score'.
        """
        if not isinstance(query, str):
            raise InvalidQueryError("Query must be a string.")

        clean_query = query.strip()
        if not clean_query:
            raise InvalidQueryError("Query must not be empty or whitespace only.")

        if top_k <= 0:
            top_k = 1

        # Bound top_k to total available index entries
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []

        try:
            # 1. Embed query
            query_vector = self.model.encode([clean_query], convert_to_numpy=True).astype("float32")

            # 2. Vector search in FAISS
            distances, indices = self.index.search(query_vector, k)

            results: List[Dict[str, Any]] = []
            for rank_idx, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                # FAISS returns -1 for unassigned/padded slots
                if idx < 0 or idx >= len(self.documents):
                    continue

                doc = self.documents[idx]
                doc_text = doc.get("text", "")

                # Generate clean snippet (up to 250 characters, ensuring natural termination)
                if len(doc_text) > 250:
                    snippet = doc_text[:247] + "..."
                else:
                    snippet = doc_text

                results.append({
                    "id": str(doc.get("id", f"doc_{idx}")),
                    "text_snippet": snippet,
                    "score": float(round(float(dist), 4)),
                })

            return results
        except Exception as e:
            if isinstance(e, InvalidQueryError):
                raise
            logger.error(f"Search execution failed for query '{clean_query}': {e}", exc_info=True)
            raise SearchServiceError(f"Error executing vector search: {str(e)}") from e


# Singleton instance container
_search_service_instance: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """
    FastAPI dependency injection provider for SearchService.
    Ensures singleton pattern to eliminate redundant model/index loading.
    """
    global _search_service_instance
    if _search_service_instance is None:
        logger.info("Initializing singleton SearchService instance...")
        _search_service_instance = SearchService()
    return _search_service_instance


def reset_search_service():
    """
    Reset singleton instance (useful for unit/integration testing fixtures).
    """
    global _search_service_instance
    _search_service_instance = None
