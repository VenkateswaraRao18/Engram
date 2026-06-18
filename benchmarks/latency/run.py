"""
Engram Latency Benchmark
========================
Measures p50/p95 latency for add() and search() across vector backends,
and upsert/neighborhood latency for graph backends.

Scales tested: 1K and 10K memories.
No LLM or embedding API calls — all vectors are synthetically generated.

Usage:
    python benchmarks/latency/run.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from engram.models import Entity, Memory, MemoryType, Relation
from engram.stores.graph.networkx_store import NetworkXGraphStore
from engram.stores.vector.faiss_store import FAISSVectorStore
from engram.stores.vector.sqlite_store import SQLiteVectorStore

DIMENSIONS = 768
USER_ID = "bench"
N_OPS = 100  # operations per timed measurement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_embedding(seed: int) -> list[float]:
    rng = np.random.RandomState(seed % (2**31))
    vec = rng.randn(DIMENSIONS).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


def make_memory(i: int) -> Memory:
    return Memory(
        user_id=USER_ID,
        content=f"Benchmark memory number {i}",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        embedding=random_embedding(i),
    )


def pct(data: list[float], p: float) -> float:
    """Return percentile in milliseconds."""
    return float(np.percentile(data, p)) * 1000


def fmt(ms: float) -> str:
    return f"{ms:.1f} ms"


# ---------------------------------------------------------------------------
# Vector store benchmark
# ---------------------------------------------------------------------------

def run_vector_benchmark(store, scale: int) -> dict:
    # Seed the store with `scale` pre-built memories
    print(f"    seeding {scale:,} memories...", end=" ", flush=True)
    seed_memories = [make_memory(i) for i in range(scale)]
    for m in seed_memories:
        store.add([m])
    print("done")

    # Benchmark: add
    add_times = []
    for i in range(N_OPS):
        m = make_memory(scale + i)
        t0 = time.perf_counter()
        store.add([m])
        add_times.append(time.perf_counter() - t0)

    # Benchmark: search
    query_emb = random_embedding(999_999)
    search_times = []
    for _ in range(N_OPS):
        t0 = time.perf_counter()
        store.search(USER_ID, query_emb, k=5)
        search_times.append(time.perf_counter() - t0)

    return {
        "add_p50":    pct(add_times, 50),
        "add_p95":    pct(add_times, 95),
        "search_p50": pct(search_times, 50),
        "search_p95": pct(search_times, 95),
    }


# ---------------------------------------------------------------------------
# Graph store benchmark
# ---------------------------------------------------------------------------

def run_graph_benchmark(store, scale: int) -> dict:
    print(f"    seeding {scale:,} entities + relations...", end=" ", flush=True)

    # Create a chain: user -> place_0, place_1, ..., place_scale
    user = Entity(user_id=USER_ID, name="BenchUser", entity_type="PERSON")
    id_map = store.upsert_entities([user])
    user_id_canonical = id_map[user.id]

    places = [Entity(user_id=USER_ID, name=f"Place{i}", entity_type="PLACE")
              for i in range(scale)]

    # Batch upsert in chunks
    chunk = 200
    place_ids = []
    for i in range(0, len(places), chunk):
        batch = places[i:i+chunk]
        m = store.upsert_entities(batch)
        place_ids.extend(m[p.id] for p in batch)

    # Add LIVES_IN relations for the first 100 (rest are just entities)
    rels = [
        Relation(
            user_id=USER_ID,
            source_entity_id=user_id_canonical,
            relation_type="KNOWS",
            target_entity_id=place_ids[i],
            memory_id=f"mem_{i}",
        )
        for i in range(min(100, scale))
    ]
    store.upsert_relations(rels)
    print("done")

    # Benchmark: upsert_entities (single entity)
    upsert_times = []
    for i in range(N_OPS):
        e = Entity(user_id=USER_ID, name=f"NewPlace{scale+i}", entity_type="PLACE")
        t0 = time.perf_counter()
        store.upsert_entities([e])
        upsert_times.append(time.perf_counter() - t0)

    # Benchmark: neighborhood traversal
    nbr_times = []
    for _ in range(N_OPS):
        t0 = time.perf_counter()
        store.neighborhood(USER_ID, ["BenchUser"], hops=2)
        nbr_times.append(time.perf_counter() - t0)

    return {
        "upsert_p50": pct(upsert_times, 50),
        "upsert_p95": pct(upsert_times, 95),
        "nbr_p50":    pct(nbr_times, 50),
        "nbr_p95":    pct(nbr_times, 95),
    }


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------

def vector_table(results: dict) -> str:
    header = (
        "| Backend         | Scale  | add p50  | add p95  | search p50 | search p95 |\n"
        "|-----------------|--------|----------|----------|------------|------------|\n"
    )
    rows = ""
    for scale, backends in sorted(results.items()):
        for name, r in backends.items():
            rows += (
                f"| {name:<15} | {scale:>6,} | {fmt(r['add_p50']):>8} | "
                f"{fmt(r['add_p95']):>8} | {fmt(r['search_p50']):>10} | "
                f"{fmt(r['search_p95']):>10} |\n"
            )
    return header + rows


def graph_table(results: dict) -> str:
    header = (
        "| Backend   | Scale  | upsert p50 | upsert p95 | neighbor p50 | neighbor p95 |\n"
        "|-----------|--------|------------|------------|--------------|---------------|\n"
    )
    rows = ""
    for scale, backends in sorted(results.items()):
        for name, r in backends.items():
            rows += (
                f"| {name:<9} | {scale:>6,} | {fmt(r['upsert_p50']):>10} | "
                f"{fmt(r['upsert_p95']):>10} | {fmt(r['nbr_p50']):>12} | "
                f"{fmt(r['nbr_p95']):>13} |\n"
            )
    return header + rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Engram Latency Benchmark")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    scales = [1_000, 10_000]
    vector_results: dict = {}
    graph_results: dict = {}

    # -- Vector stores -------------------------------------------------------
    print("\n[Vector Stores]")

    vector_backends = {
        "SQLite (numpy)": lambda: SQLiteVectorStore(db_path=":memory:"),
        "FAISS": lambda: FAISSVectorStore(db_path=":memory:", dimensions=DIMENSIONS),
    }

    for scale in scales:
        vector_results[scale] = {}
        for name, factory in vector_backends.items():
            print(f"\n  {name} @ {scale:,}")
            store = factory()
            r = run_vector_benchmark(store, scale)
            vector_results[scale][name] = r
            print(f"    add    → p50={fmt(r['add_p50'])}  p95={fmt(r['add_p95'])}")
            print(f"    search → p50={fmt(r['search_p50'])}  p95={fmt(r['search_p95'])}")

    # -- Graph stores --------------------------------------------------------
    print("\n[Graph Stores]")

    graph_backends: dict = {
        "NetworkX": lambda: NetworkXGraphStore(),
    }

    # Try Neo4j
    try:
        from engram.stores.graph.neo4j_store import Neo4jGraphStore
        probe = Neo4jGraphStore()
        probe._driver.verify_connectivity()
        probe.close()
        graph_backends["Neo4j"] = lambda: Neo4jGraphStore()
        print("  Neo4j detected — including in benchmark")
    except Exception:
        print("  Neo4j not reachable — skipping Neo4j graph benchmark")

    for scale in scales:
        graph_results[scale] = {}
        for name, factory in graph_backends.items():
            print(f"\n  {name} @ {scale:,}")
            store = factory()
            r = run_graph_benchmark(store, scale)
            graph_results[scale][name] = r
            print(f"    upsert    → p50={fmt(r['upsert_p50'])}  p95={fmt(r['upsert_p95'])}")
            print(f"    neighbor  → p50={fmt(r['nbr_p50'])}  p95={fmt(r['nbr_p95'])}")

    # -- Save results --------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "../../docs")
    os.makedirs(out_dir, exist_ok=True)

    # JSON raw results
    raw = {
        "date": datetime.utcnow().isoformat(),
        "n_ops": N_OPS,
        "dimensions": DIMENSIONS,
        "vector": {str(k): v for k, v in vector_results.items()},
        "graph": {str(k): v for k, v in graph_results.items()},
    }
    json_path = os.path.join(out_dir, "latency_results.json")
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)

    # Markdown
    md = f"""# Engram Benchmark Results

> Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
> {N_OPS} operations per measurement · {DIMENSIONS}-dim embeddings · synthetic data

## Vector Store Latency

{vector_table(vector_results)}
**Key insight:** SQLite search is O(n) — it loads all embeddings and computes cosine similarity in numpy on every query. FAISS uses an optimized C++ index in memory, so search latency stays flat as scale grows.

## Graph Store Latency

{graph_table(graph_results)}
**Key insight:** NetworkX is pure Python in-memory. Neo4j uses Cypher over a socket, so it has higher latency per operation but scales to billions of edges and supports distributed deployments.

## Notes

- All measurements are wall-clock time on a single machine (no I/O to disk for in-memory stores).
- Numbers are p50 and p95 over {N_OPS} operations after the store is seeded.
- Embeddings are randomly generated — not from a real embedding model.
- For production workloads, latency will vary based on disk speed, memory pressure, and network (for Neo4j).
"""

    md_path = os.path.join(out_dir, "benchmarks.md")
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n{'=' * 60}")
    print(f"  Results saved to:")
    print(f"    {json_path}")
    print(f"    {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
