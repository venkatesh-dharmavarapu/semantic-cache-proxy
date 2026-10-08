# Semantic Cache Proxy for LLM APIs

A high-performance semantic caching reverse proxy that sits between applications and LLM providers. By mapping semantically equivalent queries using vector embeddings, it serves cached responses with near-zero latency while reducing redundant API costs by 30–60%.

---

## Architecture: Phase 1 (Vector Engine & Similarity Index)

Incoming Request
│
▼
[CacheKeyHasher] ──► Extracts System Prompt & Param Hashes (SHA-256)
│
▼
[EmbeddingService] ──► Generates 1536-dim vector via text-embedding-3-small
│
▼
[CacheReader] ────► HNSW Vector Cosine Search (RedisVL)
│
┌───┴───────────────┐
│                   │
[HIT (≥ 0.95)]     [MISS (< 0.95)]
│                   │
Return Cached Response Forward to LLM Provider
│
[CacheWriter] ──► Store vector + payload with TTL


### Key Technical Mechanisms

1. **HNSW Cosine Vector Search**: Leverages Hierarchical Navigable Small World graphs via RedisVL for sub-millisecond approximate nearest neighbor retrieval.
2. **Deterministic Partition Tagging**: System prompt text and hyperparameters (`temperature`, `max_tokens`) are hashed into SHA-256 partition tags. Queries with distinct temperatures or system prompts never cross-contaminate.
3. **Adaptive TTL & Expiration**: Native Redis key expiration guarantees stale conversational data is automatically purged.

---

## Running Locally

### 1. Start Redis Stack
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest