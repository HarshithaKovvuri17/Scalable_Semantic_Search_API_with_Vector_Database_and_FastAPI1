# Testing Guide: Scalable Semantic Search API

Essential testing commands to verify all core requirements of the project.

---

## 1. Docker & Containerized Execution (Primary)

The production deployment runs inside Docker using Docker Compose.

### 1.1 Build and Start Containerized Service
Build the image and launch the API service with health checks in detached mode:
```powershell
docker compose up --build -d
```

### 1.2 Verify Container Health & Status
Ensure the container is running and healthy:
```powershell
docker compose ps
```

### 1.3 Run Tests Inside the Docker Container
Verify all 27 unit and integration tests inside the isolated container environment:
```powershell
docker compose exec semantic-search-api pytest tests/ -v
```

### 1.4 Test Live HTTP Endpoints (Against Docker Container)

#### Health Check (`GET /health` -> HTTP 200 OK)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json
```
*Expected: `{"status": "ok"}`*

#### Semantic Similarity Search (`POST /search` -> HTTP 200 OK)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/search" -Method Post -ContentType "application/json" -Body '{"query": "cloud container orchestration with kubernetes", "top_k": 3}' | Format-Table id, score, text_snippet -Wrap
```
*Expected: Top-3 relevant documents with `id`, `score`, and `text_snippet`.*

#### Input Validation (`POST /search` -> HTTP 400 Bad Request)
```powershell
try {
    Invoke-RestMethod -Uri "http://localhost:8000/search" -Method Post -ContentType "application/json" -Body '{"query": ""}'
} catch {
    Write-Host "Status:" $_.Exception.Response.StatusCode.value__ -ForegroundColor Yellow
    Write-Host "Response:" $_.ErrorDetails.Message -ForegroundColor Red
}
```
*Expected: Status `400` with message `"Query must not be empty and at least 3 characters long."`*

#### Missing Payload Validation (`POST /search` -> HTTP 422 Unprocessable Entity)
```powershell
try {
    Invoke-RestMethod -Uri "http://localhost:8000/search" -Method Post -ContentType "application/json" -Body '{}'
} catch {
    Write-Host "Status:" $_.Exception.Response.StatusCode.value__ -ForegroundColor Yellow
    Write-Host "Response:" $_.ErrorDetails.Message -ForegroundColor Red
}
```
*Expected: Status `422`.*

### 1.5 Stop Containerized Service
```powershell
docker compose down
```

---

## 2. Local Environment Execution (Alternative)

For testing locally without Docker:

### 2.1 Verify Data Artifacts & FAISS Index Integrity
Verify that `documents.json` contains >= 1,000 documents and matches the FAISS vector index:
```powershell
python -c "import json, faiss, numpy as np; docs = json.load(open('data/documents.json', encoding='utf-8')); index = faiss.read_index('data/faiss_index.bin'); print('Docs:', len(docs), '| Vectors:', index.ntotal, '| Dim:', index.d); assert len(docs) >= 1000 and index.ntotal >= 1000 and index.d == 384; print('SUCCESS: All data artifacts valid!')"
```

### 2.2 Run Automated Tests Locally
```powershell
pytest -v
```

### 2.3 Launch Local API Server
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3. Cache Cleanup

Remove temporary Python bytecode and pytest session caches:

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
if (Test-Path ".pytest_cache") { Remove-Item -Path ".pytest_cache" -Recurse -Force }
```
