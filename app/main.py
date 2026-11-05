"""
FastAPI Application Entry Point.

Exposes:
- GET  /health: Health check endpoint returning {"status": "ok"}.
- POST /search: Semantic similarity vector search endpoint.
- GET  /: Root information endpoint with OpenAPI documentation links.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.services.search_service import (
    SearchService,
    get_search_service,
    SearchServiceError,
    ModelLoadError,
    IndexLoadError,
    DataFileError,
    InvalidQueryError,
)

# Load environment configuration
load_dotenv()

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("semantic_search_api")

DEFAULT_TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    Pre-warms the SearchService singleton during startup to eliminate cold starts.
    """
    logger.info("Initializing application lifespan: Pre-warming SearchService...")
    try:
        service = get_search_service()
        logger.info(
            f"Pre-warming complete: Model ready, {service.index.ntotal} vectors indexed."
        )
    except Exception as e:
        logger.error(f"Critical error during startup pre-warming: {e}", exc_info=True)
    yield
    logger.info("Application shutdown: Cleaning up resources.")


# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description=(
        "Production-ready scalable Semantic Search API combining Hugging Face "
        "Sentence Transformers with FAISS vector indexing for high-performance retrieval."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS Middleware for cross-origin client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
from pydantic import ConfigDict


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "cloud container orchestration with kubernetes"
            }
        }
    )
    query: str = Field(
        ...,
        description="The semantic search query string. Must be at least 3 characters long.",
        json_schema_extra={"example": "cloud container orchestration with kubernetes"}
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Optional number of top results to retrieve. Overrides the TOP_K_RESULTS environment variable."
    )


class SearchResultItem(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_0021",
                "text_snippet": "Recent breakthroughs in Technology demonstrate that: Kubernetes container orchestration...",
                "score": 0.3376
            }
        }
    )
    id: str = Field(..., description="Unique document identifier", json_schema_extra={"example": "doc_0021"})
    text_snippet: str = Field(..., description="Contextual snippet of the original document text")
    score: float = Field(..., description="FAISS L2 distance score (lower indicates closer semantic proximity)")


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok"
            }
        }
    )
    status: str = Field("ok", json_schema_extra={"example": "ok"})



# Exception Handlers
@app.exception_handler(InvalidQueryError)
async def invalid_query_exception_handler(request: Request, exc: InvalidQueryError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(IndexLoadError)
@app.exception_handler(DataFileError)
@app.exception_handler(ModelLoadError)
async def service_dependency_exception_handler(request: Request, exc: SearchServiceError):
    logger.error(f"Service dependency failure: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Search service dependency error: {str(exc)}"},
    )


@app.exception_handler(SearchServiceError)
async def general_search_exception_handler(request: Request, exc: SearchServiceError):
    logger.error(f"Search execution error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal vector search service error."},
    )


# Endpoints
@app.get("/", status_code=200, summary="API Root Information")
async def root_endpoint():
    """
    Returns service metadata, version, and links to interactive OpenAPI docs and health check.
    """
    return {
        "title": "Semantic Search API",
        "version": "1.0.0",
        "description": "High-performance semantic vector retrieval powered by Sentence Transformers and FAISS.",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "swagger_docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    summary="Health Check",
    tags=["System"]
)
async def health_check():
    """
    Health check endpoint to verify API readiness and liveness.
    Returns {"status": "ok"} with a 200 OK status code.
    """
    return {"status": "ok"}


@app.post(
    "/search",
    response_model=List[SearchResultItem],
    status_code=200,
    summary="Execute Semantic Search",
    tags=["Search"]
)
async def semantic_search_endpoint(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service)
):
    """
    Accepts a search query and returns the top-K semantically relevant documents
    ranked by FAISS vector similarity.
    """
    # Strict validation per requirements: empty or length < 3
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty and at least 3 characters long."
        )

    try:
        # Determine top_k: request override takes precedence, then TOP_K_RESULTS env var, fallback to 5
        top_k = request.top_k if (request.top_k is not None and request.top_k > 0) else int(os.getenv("TOP_K_RESULTS", str(DEFAULT_TOP_K)))
        results = service.search_documents(
            query=request.query,
            top_k=top_k
        )
        return results
    except InvalidQueryError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unhandled error in semantic_search_endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the semantic search request."
        )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
