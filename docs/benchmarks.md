# Engram Benchmark Results

> Generated: 2026-06-10 15:38 UTC
> 100 operations per measurement · 768-dim embeddings · synthetic data

## Vector Store Latency

| Backend         | Scale  | add p50  | add p95  | search p50 | search p95 |
|-----------------|--------|----------|----------|------------|------------|
| SQLite (numpy)  |  1,000 |   0.2 ms |   0.2 ms |   104.9 ms |   106.3 ms |
| FAISS           |  1,000 |   0.2 ms |   0.2 ms |     0.5 ms |     0.5 ms |
| SQLite (numpy)  | 10,000 |   0.2 ms |   0.2 ms |  1007.2 ms |  1034.5 ms |
| FAISS           | 10,000 |   0.2 ms |   0.2 ms |     7.1 ms |    10.3 ms |

**Key insight:** SQLite search is O(n) — it loads all embeddings and computes cosine similarity in numpy on every query. FAISS uses an optimized C++ index in memory, so search latency stays flat as scale grows.

## Graph Store Latency

| Backend   | Scale  | upsert p50 | upsert p95 | neighbor p50 | neighbor p95 |
|-----------|--------|------------|------------|--------------|---------------|
| NetworkX  |  1,000 |     0.1 ms |     0.1 ms |       0.3 ms |        0.3 ms |
| Neo4j     |  1,000 |     1.1 ms |     2.3 ms |       5.6 ms |       12.2 ms |
| NetworkX  | 10,000 |     0.7 ms |     0.7 ms |       0.3 ms |        0.4 ms |
| Neo4j     | 10,000 |     0.5 ms |     1.6 ms |      17.1 ms |       22.2 ms |

**Key insight:** NetworkX is pure Python in-memory. Neo4j uses Cypher over a socket, so it has higher latency per operation but scales to billions of edges and supports distributed deployments.

## Notes

- All measurements are wall-clock time on a single machine (no I/O to disk for in-memory stores).
- Numbers are p50 and p95 over 100 operations after the store is seeded.
- Embeddings are randomly generated — not from a real embedding model.
- For production workloads, latency will vary based on disk speed, memory pressure, and network (for Neo4j).
