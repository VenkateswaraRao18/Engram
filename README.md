# Engram

Long-term memory for LLM applications, built on a hybrid vector store and temporal knowledge graph.

---

## Why I built this

Every memory library I looked at had the same blind spot: they treat memory as an append-only log. If a user tells your assistant they live in Tampa, then six sessions later says they moved to Austin, you get two facts sitting side by side in the store with no relationship between them. The next search returns both, and you ship conflicting context to your LLM.

The root problem is that most systems store *what was said*, not *what is currently true*. Facts have a lifespan. Engram tracks that.

---

## How it works

Two stores, always in sync:

**Vector store** (SQLite + numpy) holds the raw memory text and embeddings. Fast cosine search for semantic similarity.

**Knowledge graph** (NetworkX) holds typed relationships between entities — things like `LIVES_IN`, `WORKS_AT`, `PREFERS` — each with a validity window. When a new fact contradicts an existing one (same entity, same relation type), the old relation's `valid_until` gets set and the backing memory is marked `superseded_by`. Both become invisible to future searches automatically.

On every `search()` call, both stores are queried in parallel. Results are fused using **Reciprocal Rank Fusion (RRF)**, then boosted by recency and importance score. You get a single ranked list that reflects both semantic similarity and graph structure.

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │           Engram Client             │
                    │   add()  search()  get_context()      │
                    │   forget()  forget_user()  stats()    │
                    └───────┬──────────────────┬───────────┘
                            │ write            │ read
              ┌─────────────▼──────┐   ┌───────▼──────────────────┐
              │  Extraction Engine  │   │     Hybrid Retriever      │
              │  LLM → structured  │   │  vector top-20  ──┐       │
              │  JSON → memories,  │   │  graph traversal ─┤ RRF   │
              │  entities,         │   │  recency boost  ──┘       │
              │  relations         │   └───────┬──────────────────┘
              └──┬──────────┬──────┘           │ ranked SearchResults
                 │          │                  │
        ┌────────▼───┐  ┌───▼──────────┐       │
        │   Vector   │  │    Graph     │◄──────┘
        │   Store    │  │    Store     │
        │  (SQLite   │  │ (NetworkX    │
        │  + numpy)  │  │  + JSON)     │
        └────────────┘  └──────────────┘
                 ▲            ▲
          ┌──────┴────────────┴──────┐
          │   Consolidation Engine   │
          │  dedup · supersession    │
          │  entity resolution       │
          └──────────────────────────┘
```

**Write path:** message → LLM extraction → candidate memories + graph triples → consolidation (dedup check, conflict detection) → persist to both stores.

**Read path:** query → embed → vector top-20; in parallel, extract entity names → graph neighborhood traversal → RRF fusion → recency + importance boost → ranked results.

---

## The supersession scenario

This is the core behavior that separates Engram from append-only systems.

```python
from engram import Engram

mw = Engram.local("./mydb")

# Session 1
mw.add(
    messages=[{"role": "user", "content": "Hi, I'm Alex. I live in Tampa, Florida."}],
    user_id="alex"
)
# Graph: LIVES_IN(Alex → Tampa) [valid]

# Session 4, two weeks later
mw.add(
    messages=[{"role": "user", "content": "I just moved to Austin last month."}],
    user_id="alex"
)
# Graph: LIVES_IN(Alex → Tampa) [invalidated at 2026-06-09]
#        LIVES_IN(Alex → Austin) [valid]
# Memory "Alex lives in Tampa" → superseded_by: <Austin memory id>

results = mw.search("Where does Alex live?", user_id="alex")
# Returns: Austin memory only. Tampa is gone from results.

ctx = mw.get_context("Where does Alex live?", user_id="alex")
# "- [episodic] User moved to Austin last month"
# "- [semantic] User is a software engineer"
```

**Demo output (real, running locally with llama3.1 + nomic-embed-text):**

```
========= Step 1: Alex introduces himself (Tampa) =========
Stored 2 memories:
  [episodic] (importance=0.85) User lives in Tampa
  [semantic] (importance=0.75) User is a software engineer

Graph: User --[LIVES_IN]--> Tampa  [VALID]

=============== Step 2: Alex moves to Austin ===============
Stored 1 memory:
  [episodic] (importance=0.85) User moved to Austin last month

Supersession detected!
  Tampa memory → superseded_by: Austin memory

Active relations:
  User --[MOVED_TO]--> Austin
  User --[LIVES_IN]--> Austin

Invalidated:
  User --[LIVES_IN]--> Tampa  [INVALIDATED at 2026-06-09 22:00:52]

============ Step 3: Search "Where does Alex live?" ============
Results (Tampa excluded — superseded):
  1. User moved to Austin last month  [VALID]
  2. User is a software engineer      [VALID]

Stats: 3 total memories, 2 currently valid
```

---

## Install

```bash
git clone https://github.com/yourusername/engram
cd engram
pip install -e ".[local]"
```

Requirements: Python 3.9+, [Ollama](https://ollama.com) running locally with `llama3.1` and `nomic-embed-text` pulled.

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

---

## Quick reference

```python
from engram import Engram

mw = Engram.local("./memdb")       # zero-config, everything local

# Store memories from a conversation
memories = mw.add(
    messages=[{"role": "user", "content": "..."}],
    user_id="u1"
)

# Semantic + graph search
results = mw.search("query", user_id="u1", k=5)
for r in results:
    print(r.score, r.memory.content, r.retrieval_path)

# Formatted context string for prompt injection
ctx = mw.get_context("query", user_id="u1", max_tokens=800)

# Delete a specific memory
mw.forget(memory_id="...")

# GDPR-style full wipe
mw.forget_user(user_id="u1")

# Usage stats
mw.stats(user_id="u1")
```

---

## How memories are classified

Every extracted memory gets a type:

| Type | What it captures | Example |
|---|---|---|
| `semantic` | Stable facts about the user | "User is vegetarian" |
| `episodic` | Events and experiences | "User asked about flights to Tokyo" |
| `procedural` | Preferences and habits | "User prefers answers in bullet points" |

Importance scores are assigned by the LLM during extraction (identity facts ~0.9, location ~0.85, transient preferences ~0.4). Both type and importance feed into retrieval ranking.

---

## How RRF fusion works

Instead of picking one retrieval strategy, both run in parallel and scores are merged:

```
rrf_score(memory) = 1/(60 + rank_vector) + 1/(60 + rank_graph)
```

A memory that ranks 3rd in vector search and 5th in graph traversal scores higher than one that ranks 1st in only one list. The constant 60 prevents high-ranked results from dominating.

Final score applies recency and importance on top:

```
final = rrf × (1 + 0.3 × recency_decay) × (1 + 0.2 × importance)
```

Recency uses an exponential decay with a 30-day half-life (configurable).

---

## How it compares

| Feature | Engram | Mem0 | Zep | LangMem |
|---|---|---|---|---|
| Vector search | ✓ | ✓ | ✓ | ✓ |
| Knowledge graph | ✓ | partial | ✓ | ✗ |
| Temporal supersession | ✓ | ✗ | ✗ | ✗ |
| Hybrid retrieval (RRF) | ✓ | ✗ | ✗ | ✗ |
| Fully local (no API keys) | ✓ | ✗ | partial | ✓ |
| Published benchmarks | planned (v0.3) | ✗ | ✗ | ✗ |
| Framework-agnostic | ✓ | ✓ | ✓ | ✗ |
| Open source | ✓ | partial | partial | ✓ |

The main gap I wanted to close: Mem0 is the most widely used but has no temporal reasoning. Zep has a graph but doesn't do hybrid fusion or supersession. LangMem is tied to LangChain. None of them publish reproducible benchmark numbers.

---

## Project structure

```
engram/
├── src/engram/
│   ├── client.py              # Public API: Engram class
│   ├── config.py              # All tunables in one place
│   ├── models.py              # Memory, Entity, Relation, SearchResult
│   ├── extraction/
│   │   ├── llm_extractor.py   # LLM call → structured JSON → typed objects
│   │   └── prompts.py         # Extraction prompt templates
│   ├── embeddings/
│   │   └── ollama_embedder.py # nomic-embed-text via Ollama
│   ├── stores/
│   │   ├── vector/
│   │   │   └── sqlite_store.py    # SQLite + numpy cosine similarity
│   │   └── graph/
│   │       └── networkx_store.py  # NetworkX + JSON persistence
│   ├── retrieval/
│   │   └── hybrid.py          # RRF fusion, recency/importance boost
│   └── consolidation/
│       └── engine.py          # Dedup, supersession, entity resolution
├── tests/
│   ├── conftest.py            # FakeLLM, FakeEmbedder fixtures
│   └── unit/                  # 44 tests, all passing
└── demo.py                    # Tampa → Austin scenario end-to-end
```

---

## Configuration

Everything tunable lives in `EngramConfig`. No magic numbers in the codebase.

```python
from engram import Engram, EngramConfig

mw = Engram(EngramConfig(
    llm_model="llama3.1",
    embedding_model="nomic-embed-text",
    embedding_dimensions=768,
    rrf_k=60,                       # RRF constant
    recency_weight=0.3,             # boost weight for recent memories
    importance_weight=0.2,          # boost weight for high-importance memories
    recency_half_life_days=30.0,    # recency decay rate
    dedup_similarity_threshold=0.92 # cosine similarity to trigger dedup
))
```

---

## Running tests

```bash
pytest tests/unit -x -q
```

44 tests covering: models, extraction with fake LLM, vector store CRUD + cosine search, graph store entity resolution + conflict detection + persistence, RRF math, consolidation dedup + supersession.

---

## Limitations (honest)

**This is v0.** A few things to know before using it:

- **Extraction quality depends on your local LLM.** llama3.1 works well for straightforward facts but occasionally misses structured relations in long messages. A bigger model (or Anthropic/OpenAI) will do better.
- **Vector search is brute-force.** We compute cosine similarity across all memories for every query. Fine up to a few thousand memories per user. Above that, you'll want FAISS or Qdrant (planned for v0.2).
- **Graph store is embedded NetworkX.** Documented limit: ~100K edges. Beyond that, Neo4j backend is the path (planned for v0.2).
- **No async client yet.** Everything is synchronous. Async wrapper is on the roadmap.
- **No benchmarks published yet.** Numbers against Mem0, Zep, and naive RAG baselines are planned for v0.3. I'll publish honestly regardless of outcome.

---

## Roadmap

**v0.1 (current)** — Core working library
- SQLite vector store (numpy cosine)
- NetworkX graph store (JSON persistence)
- Hybrid RRF retrieval
- Temporal supersession
- 44 unit tests

**v0.2** — Multiple backends + async
- FAISS and Qdrant vector backends
- Neo4j graph backend
- AsyncEngram wrapper
- Shared backend contract test suite

**v0.3** — Benchmarks
- LongMemEval harness (Engram vs Mem0 vs Zep vs naive RAG)
- LoCoMo QA accuracy
- Latency suite at 1K / 10K / 100K memories
- Published to docs/ regardless of whether we win or lose each category

**v1.0** — Production-ready
- PyPI trusted publishing
- MkDocs documentation site
- Full API reference
- CI/CD matrix (Python 3.10, 3.11, 3.12)

---

## License

Apache 2.0
