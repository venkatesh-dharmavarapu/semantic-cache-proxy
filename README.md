Markdown
# ⚡ Semantic Cache Proxy for Local LLMs (Ollama + Redis Stack)

A high-performance, drop-in semantic caching reverse proxy designed for local LLM workloads. By indexing prompt embeddings into a Redis HNSW vector graph, identical and semantically paraphrased queries return cached completions in **< 15 ms**, eliminating local GPU/CPU inference bottlenecks.

100% local, free-tier, and private. Requires zero paid API keys.

---

## Architecture Flow

Incoming Client Request (OpenAI /v1 SDK, curl, LangChain)
│
▼
FastAPI Proxy (:8000/v1)
│
▼
[CacheKeyHasher: SHA-256] ──► Hashes System Prompt & Params
│
▼
[Local Ollama Embeddings]  ──► nomic-embed-text (768 dims)
│
▼
[Redis Stack HNSW Vector Index]
│
┌────────────┴────────────┐
▼                         ▼
HIT (Score ≥ 0.88)        MISS / BYPASS
│                         │
Serve Cached Payload      Forward to Ollama (:11434/v1)
(Latency: < 15ms)                │
Store Vector + Payload (TTL)
│
Return Response + Telemetry Headers


---

## Features

- **Drop-In Compatibility:** Matches standard `/v1/chat/completions` specifications.
- **Parametric Partitioning:** SHA-256 tagging ensures queries with distinct temperatures, models, or system prompts never cross-contaminate.
- **Custom Response Headers:** Inspect `X-Cache-Lookup` (`HIT`, `MISS`, `BYPASS`), `X-Cache-Similarity`, and `X-Lookup-Latency-ms`.
- **Atomic Telemetry:** Redis-backed counters track aggregate hit ratios and compute time saved.
- **Interactive UI:** Built-in Streamlit dashboard for real-time monitoring and inspection.

---

## Quickstart

### 1. Prerequisites
- Docker & Docker Desktop
- Ollama installed locally

```powershell
# Pull required local models
ollama pull nomic-embed-text
ollama pull llama3.2:1b
2. Start Redis Stack
PowerShell
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
3. Start the FastAPI Proxy Server
PowerShell
uvicorn app.main:app --reload --port 8000
Open Swagger docs: http://localhost:8000/docs

4. Launch the Streamlit Telemetry Dashboard
PowerShell
streamlit run app/dashboard.py
5. Run the Test Suite
PowerShell
python -m pytest -v tests/
6. Run the Latency Benchmark
PowerShell
python scripts/benchmark.py