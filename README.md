# Engram

[![PyPI version](https://img.shields.io/pypi/v/engram-ltm.svg)](https://pypi.org/project/engram-ltm/)
[![Python](https://img.shields.io/pypi/pyversions/engram-ltm.svg)](https://pypi.org/project/engram-ltm/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/VenkateswaraRao18/Engram/actions/workflows/ci.yml/badge.svg)](https://github.com/VenkateswaraRao18/Engram/actions)

**Long-term memory for LLM applications** — hybrid vector store + temporal knowledge graph with automatic supersession.

---

## The Problem

Most memory systems are append-only. When a user says they live in Tampa and later says they moved to Austin, both facts sit side by side in the store with no relationship between them. The next search returns both, and your LLM gets conflicting context.

Engram solves this by tracking not just *what was said*, but *what is currently true*. When a new fact contradicts an existing one, the old memory is automatically marked as superseded and becomes invisible to future searches.

---

## Key Features

- **Temporal supersession** — single-valued facts (location, job, relationship status) update in place; stale versions are hidden automatically
- **Hybrid retrieval** — vector search and graph traversal fused via Reciprocal Rank Fusion (RRF), boosted by recency and importance
- **Tense-aware extraction** — distinguishes "I moved to Austin" (completed) from "I'm moving to Austin next month" (planned); only completed moves trigger supersession
- **Zero-infrastructure default** — SQLite + NetworkX, no external services required
- **Colab-ready** — works with `sentence-transformers` (no API key for embeddings) and Gemini (free tier)
- **Swappable backends** — SQLite → FAISS, NetworkX → Neo4j, Ollama → Gemini
- **Fully open source** — Apache 2.0, reproducible benchmarks included

---

## Install

```bash
pip install engram-ltm
```

**For Google Colab or cloud environments** (no Ollama required):

```bash
pip install engram-ltm sentence-transformers google-genai
```

**For local use with Ollama:**

```bash
pip install engram-ltm
ollama pull gemini-2.5-flash   # or any supported model
ollama pull nomic-embed-text
```

Requires Python 3.9+

---

## Quick Start

### Local (Ollama)

```python
from engram import Engram

memory = Engram.local("./memdb")

memory.add(
    messages=[{"role": "user", "content": "I live in Tampa and work at Google."}],
    user_id="alice"
)

memory.add(
    messages=[{"role": "user", "content": "I just moved to Austin and joined Stripe."}],
    user_id="alice"
)

context = memory.get_context("Where does Alice work?", user_id="alice")
# Returns: Austin and Stripe only — Tampa and Google are superseded
```

### Cloud / Google Colab (Gemini + sentence-transformers)

```python
from engram import Engram, EngramConfig

memory = Engram(EngramConfig(
    embedding_provider="sentence-transformers",
    llm_provider="gemini",
    llm_model="gemini-2.5-flash",
    gemini_api_key="YOUR_KEY",
))

memory.add(messages=[...], user_id="u1")
context = memory.get_context("query", user_id="u1")
```

A full interactive demo is available in [`examples/colab_demo.ipynb`](examples/colab_demo.ipynb) — open directly in Google Colab.

---

## How It Works

```
User message
     │
     ▼
┌─────────────────┐
│ Extraction LLM  │  → memories + entities + relations (JSON)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Consolidation Engine        │
│  • Deduplication (cosine)       │
│  • Conflict detection           │
│  • Supersession (graph)         │
└────────┬──────────────┬─────────┘
         │              │
         ▼              ▼
  ┌─────────────┐  ┌──────────────┐
  │ Vector Store│  │ Graph Store  │
  │ (SQLite /   │  │ (NetworkX /  │
  │  FAISS)     │  │  Neo4j)      │
  └──────┬──────┘  └──────┬───────┘
         │                │
         └───────┬─────────┘
                 ▼
         ┌───────────────┐
         │ Hybrid RRF    │  ← recency + importance boost
         │ Retriever     │
         └───────┬───────┘
                 ▼
          Ranked results
```

**Write path:** conversation → LLM extracts structured facts → consolidation deduplicates and detects conflicts → both stores updated atomically.

**Read path:** query → vector top-20 + graph neighborhood traversal → RRF fusion → recency and importance boost → ranked `SearchResult` list.

---

## Supersession

The core behavior that separates Engram from append-only systems.

When a new fact arrives for a single-valued relation (`LIVES_IN`, `WORKS_AT`, `RELATIONSHIP_STATUS`, etc.), Engram:

1. Finds any existing active relation of the same type for the same entity
2. Sets `valid_until` on the old graph edge
3. Marks the old memory `superseded_by` the new memory ID
4. Both become invisible to all future searches automatically

Tense matters: `"I'm planning to move to Austin next month"` stores the intention as an episodic memory but does **not** emit a `LIVES_IN` relation, so the current city is preserved until the move is confirmed.

---

## Benchmarks

### Memory QA Accuracy

25 hand-crafted examples across five question types (LongMemEval-style stress benchmark). Evaluated with **Gemini 2.5 Flash** as both extractor and answer model.

| System | knowledge\_update | temporal\_chain | single\_session | multi\_session | abstained | **Overall** |
|--------|:-----------------:|:---------------:|:---------------:|:--------------:|:---------:|:-----------:|
| **Engram** | **80%** | **80%** | 80% | **80%** | **100%** | **84%** |
| VectorOnly | 40% | 60% | 80% | 80% | 100% | 72% |
| NaiveRAG | 100% | 40% | 80% | 60% | 100% | 76% |

- **knowledge\_update** — fact updated across 7–8 sessions with 5 noise sessions in between and no temporal language. VectorOnly returns stale values; Engram supersedes correctly.
- **temporal\_chain** — same fact updated 3× in sequence (e.g. Tampa → Austin → Denver). VectorOnly returns "I don't know" when all three coexist in context. Engram chains supersession and returns only the latest.
- **abstained\_response** — all systems correctly refuse questions about facts never mentioned.

Reproduction scripts: `benchmarks/longmemeval/run.py`

### Latency

| Backend | Scale | search p50 | search p95 |
|---------|------:|:----------:|:----------:|
| SQLite (numpy) | 1 000 | 104.9 ms | 106.3 ms |
| FAISS | 1 000 | 0.5 ms | 0.5 ms |
| SQLite (numpy) | 10 000 | 1 007 ms | 1 034 ms |
| FAISS | 10 000 | 7.1 ms | 10.3 ms |

SQLite search is O(n). Switch to FAISS with `vector_store="faiss"` in `EngramConfig` for a 142× speedup at 10K scale.

---

## Comparison

| Feature | Engram | Mem0 | Zep | LangMem |
|---------|:------:|:----:|:---:|:-------:|
| Vector search | ✓ | ✓ | ✓ | ✓ |
| Knowledge graph | ✓ | partial | ✓ | ✗ |
| Temporal supersession | ✓ | ✗ | ✗ | ✗ |
| Hybrid retrieval (RRF) | ✓ | ✗ | ✗ | ✗ |
| Tense-aware extraction | ✓ | ✗ | ✗ | ✗ |
| Colab / cloud ready | ✓ | ✗ | ✗ | partial |
| Reproducible benchmarks | ✓ | ✗ | ✗ | ✗ |
| Framework-agnostic | ✓ | ✓ | ✓ | ✗ |
| Open source | ✓ | partial | partial | ✓ |

---

## Configuration

All tunables live in `EngramConfig`. Sensible defaults for everything.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_provider` | `"ollama"` | `"ollama"` or `"gemini"` |
| `llm_model` | `"gemini-2.5-flash"` | Model name passed to the provider |
| `embedding_provider` | `"ollama"` | `"ollama"` or `"sentence-transformers"` |
| `embedding_model` | `"nomic-embed-text"` | Model name (ST: `"all-MiniLM-L6-v2"`) |
| `vector_store` | `"sqlite"` | `"sqlite"` or `"faiss"` |
| `graph_store` | `"networkx"` | `"networkx"` or `"neo4j"` |
| `dedup_similarity_threshold` | `0.92` | Cosine threshold for deduplication |
| `recency_half_life_days` | `30.0` | Half-life for recency decay |
| `rrf_k` | `60` | RRF fusion constant |

---

## API Reference

```python
from engram import Engram, EngramConfig

memory = Engram(EngramConfig(...))

# Store memories from a conversation turn
memories = memory.add(messages=[...], user_id="u1")

# Search — returns ranked SearchResult list
results = memory.search("query", user_id="u1", k=5)

# Get a formatted context string ready for LLM injection
context = memory.get_context("query", user_id="u1", max_tokens=800)

# Delete a specific memory by ID
memory.forget(memory_id="...")

# Delete all memories for a user
memory.forget_user(user_id="u1")

# Usage statistics
memory.stats(user_id="u1")
```

---

## Running Tests

```bash
# Clone the repository
git clone https://github.com/VenkateswaraRao18/Engram.git
cd Engram

# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests (73 tests, no external services needed)
pytest tests/unit -x -q

# Latency benchmarks (synthetic data, no Ollama required)
python benchmarks/latency/run.py

# QA accuracy benchmark — stress dataset
python benchmarks/longmemeval/run.py --data benchmarks/longmemeval/data/sample_stress.json
```

---

## Roadmap

**v0.1** — Core library
- Hybrid vector + graph store, temporal supersession, RRF retrieval, 44 unit tests

**v0.2** — Multiple backends + Colab support *(current — v0.2.9)*
- FAISS and Neo4j backends
- AsyncEngram wrapper
- Gemini provider (direct REST, no SDK version issues)
- `sentence-transformers` embedder (Colab-ready, no API key)
- Tense-aware extraction (future moves don't supersede current facts)
- Supersession correctness fixes (memory-to-relation linking)
- LongMemEval-style stress benchmark (25 examples, 5 categories, 3 systems)
- CI/CD via GitHub Actions

**v1.0** — Production-ready
- HuggingFace extraction provider
- MkDocs documentation site
- Full LongMemEval evaluation (500 examples)
- PyPI trusted publishing

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Author

**Venky Jannegorla**
- GitHub: [@VenkateswaraRao18](https://github.com/VenkateswaraRao18)
- Email: venkyjannegorla@gmail.com
- PyPI: [engram-ltm](https://pypi.org/project/engram-ltm/)
