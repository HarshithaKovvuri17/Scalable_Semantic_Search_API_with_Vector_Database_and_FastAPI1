#!/bin/sh
set -e

# Ensure model cache exists (e.g. when models volume is mounted empty from host)
if [ ! -f "models/sentence_transformer/config.json" ] && [ ! -f "models/sentence_transformer/modules.json" ]; then
    if [ -d "/app/models_cache/sentence_transformer" ]; then
        echo "[Entrypoint] Populating model cache from pre-built image cache..."
        mkdir -p models/sentence_transformer
        cp -r /app/models_cache/sentence_transformer/* models/sentence_transformer/ 2>/dev/null || true
    else
        echo "[Entrypoint] Model cache not found. Downloading and caching model..."
        python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('models/sentence_transformer')"
    fi
else
    echo "[Entrypoint] SentenceTransformer model cache verified."
fi

# Verify if data and index files exist; if not, initialize them automatically
if [ ! -f "data/faiss_index.bin" ] || [ ! -f "data/documents.json" ]; then
    echo "[Entrypoint] Data or FAISS index not found. Running generation pipeline..."
    python scripts/generate_embeddings.py
else
    echo "[Entrypoint] Existing data and FAISS index detected."
fi

echo "[Entrypoint] Starting application server..."
exec "$@"
