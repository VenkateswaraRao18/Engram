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
pip install engram-ltm
```

Or from source:

```bash
git clone https://github.com/venkyjannegorla/engram
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

## Benchmarks

All numbers were generated on a single machine. Reproduction scripts are in `benchmarks/`.

### Latency (synthetic 768-dim embeddings, 100 ops each)

| Backend        | Scale  | search p50 | search p95 |
|----------------|--------|------------|------------|
| SQLite (numpy) |  1,000 |   104.9 ms |   106.3 ms |
| FAISS          |  1,000 |     0.5 ms |     0.5 ms |
| SQLite (numpy) | 10,000 |  1,007.2 ms |  1,034.5 ms |
| FAISS          | 10,000 |     7.1 ms |    10.3 ms |

SQLite search is O(n) — brute-force cosine over every embedding on every query. FAISS stays flat. Switch with `vector_store="faiss"` in `EngramConfig`.

| Backend  | Scale  | neighbor p50 | neighbor p95 |
|----------|--------|--------------|--------------|
| NetworkX |  1,000 |      0.3 ms  |      0.3 ms  |
| Neo4j    |  1,000 |      5.6 ms  |     12.2 ms  |
| NetworkX | 10,000 |      0.3 ms  |      0.4 ms  |
| Neo4j    | 10,000 |     17.1 ms  |     22.2 ms  |

NetworkX is pure Python in-memory. Neo4j adds network overhead but scales to billions of edges.

### Memory QA accuracy (LongMemEval-style stress benchmark, 25 examples)

25 hand-crafted QA examples across five question types. Results with **Gemini 2.5 Flash** as both extractor and answer model:

| System | knowledge_update | temporal_chain | single_session | multi_session | abstained | Overall |
|--------|-----------------|----------------|----------------|---------------|-----------|---------|
| **Engram (full)** | **80%** | **80%** | 80% | **80%** | **100%** | **84%** |
| VectorOnly (no supersession) | 40% | 60% | 80% | 80% | 100% | 72% |
| NaiveRAG (baseline) | 100% | 40% | 80% | 60% | 100% | 76% |

**`knowledge_update`** (5 examples, 7–8 sessions each with noise between old and new fact): VectorOnly scores 40% because without supersession both facts coexist — the LLM sees "User lives in Tampa" alongside "User lives in Austin" with no way to determine recency and answers wrong or refuses. Engram's supersession removes the stale fact, leaving only the current value.

**`temporal_chain`** (5 examples, same fact updated 3× without temporal cues): the hardest category. VectorOnly and NaiveRAG both degrade below 60% because neither can resolve three conflicting versions of the same fact. Engram chains the supersession — Tampa → Austin → Denver leaves only Denver — and scores 80%.

**`multi_session`** shows the extraction advantage: Engram's structured graph edges connect entities across sessions (brother Alex → Amazon), while NaiveRAG misses some multi-hop connections.

**`abstained_response`** tests correct refusal for never-mentioned facts. All systems score 100%.

The stress benchmark is specifically designed to expose the weaknesses of append-only systems. Each `knowledge_update` example adds 5–6 unrelated noise sessions between the old and new fact, and removes explicit temporal language ("I just moved") that would otherwise let any LLM infer recency from phrasing alone.

**Concrete VectorOnly failure examples:**
- `ku_001`: returns "Tampa" (stale) instead of "Austin" (current)
- `ku_005`: returns "computer science" (stale) instead of "data science" (current)
- `tc_001`: returns "I don't know" — confused by Tampa + Austin + Denver all in context
- `tc_003`: returns "I don't know" — confused by Python + Go + Rust all in context

Full results and raw JSON in `benchmarks/longmemeval/results/`.

---

## How it compares

| Feature | Engram | Mem0 | Zep | LangMem |
|---|---|---|---|---|
| Vector search | ✓ | ✓ | ✓ | ✓ |
| Knowledge graph | ✓ | partial | ✓ | ✗ |
| Temporal supersession | ✓ | ✗ | ✗ | ✗ |
| Hybrid retrieval (RRF) | ✓ | ✗ | ✗ | ✗ |
| Fully local (no API keys) | ✓ | ✗ | partial | ✓ |
| Published benchmarks | ✓ | ✗ | ✗ | ✗ |
| Framework-agnostic | ✓ | ✓ | ✓ | ✗ |
| Open source | ✓ | partial | partial | ✓ |

The main gap I wanted to close: Mem0 is the most widely used but has no temporal reasoning. Zep has a graph but doesn't do hybrid fusion or supersession. LangMem is tied to LangChain. None of them publish reproducible benchmark numbers.

---

## Project structure

```
engram/
├── src/engram/
│   ├── client.py              # Public API: Engram class
│   ├── async_client.py        # AsyncEngram wrapper
│   ├── config.py              # All tunables in one place
│   ├── models.py              # Memory, Entity, Relation, SearchResult
│   ├── extraction/
│   │   ├── llm_extractor.py   # LLM call → structured JSON → typed objects
│   │   └── prompts.py         # Extraction prompt templates
│   ├── embeddings/
│   │   └── ollama_embedder.py # nomic-embed-text via Ollama
│   ├── stores/
│   │   ├── vector/
│   │   │   ├── sqlite_store.py    # SQLite + numpy (default)
│   │   │   └── faiss_store.py     # FAISS IndexFlatIP (fast)
│   │   └── graph/
│   │       ├── networkx_store.py  # NetworkX + JSON persistence (default)
│   │       └── neo4j_store.py     # Neo4j via Cypher (scale)
│   ├── retrieval/
│   │   └── hybrid.py          # RRF fusion, recency/importance boost
│   └── consolidation/
│       └── engine.py          # Dedup, supersession, entity resolution
├── benchmarks/
│   ├── latency/run.py         # Vector/graph store latency at 1K and 10K scale
│   └── longmemeval/
│       ├── run.py             # QA accuracy benchmark (3 systems, 4 question types)
│       ├── adapters/          # Engram, VectorOnly, NaiveRAG adapters
│       ├── data/sample_20.json  # 20 hand-crafted evaluation examples
│       └── results/           # JSON + markdown outputs
├── tests/
│   ├── conftest.py            # FakeLLM, FakeEmbedder fixtures
│   └── unit/                  # 44 tests, all passing
├── docs/
│   ├── benchmarks.md          # Latency benchmark results
│   └── latency_results.json   # Raw latency numbers
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

## Running tests and benchmarks

```bash
# Unit tests
pytest tests/unit -x -q

# Latency benchmarks (no Ollama needed — uses synthetic embeddings)
python benchmarks/latency/run.py

# QA accuracy benchmark (Ollama + llama3.1 required, ~15 minutes)
python benchmarks/longmemeval/run.py

# Stress benchmark (noise sessions + temporal_chain category, ~25 minutes)
python benchmarks/longmemeval/run.py --data benchmarks/longmemeval/data/sample_stress.json

# Quick mode: knowledge_update + temporal_chain only (~8 minutes)
python benchmarks/longmemeval/run.py --quick --data benchmarks/longmemeval/data/sample_stress.json
```

44 tests covering: models, extraction with fake LLM, vector store CRUD + cosine search, graph store entity resolution + conflict detection + persistence, RRF math, consolidation dedup + supersession.

---

## Limitations (honest)

**This is v0.2.** A few things to know before using it:

- **Extraction quality depends on your local LLM.** llama3.1 works well for straightforward facts but occasionally misses structured relations in long messages or generates JSON that doesn't parse cleanly. A bigger model (Llama 3.3 70B or Anthropic/OpenAI) will do better.
- **Dedup has a blind spot for template-structured facts.** "User lives in Tampa" and "User lives in Austin" get cosine similarity ≈ 1.0 from nomic-embed-text because they share semantic structure. Engram handles this by bypassing dedup for memories linked to functional relations (LIVES_IN, WORKS_AT, etc.) and letting supersession do the work instead.
- **SQLite search is O(n).** Fine to a few thousand memories per user. Above ~5K, switch to `vector_store="faiss"` in `EngramConfig` for a 142× speedup at 10K scale.
- **NetworkX graph limit: ~100K edges.** Beyond that, switch to `graph_store="neo4j"`.
- **Multi-session reasoning is harder than single-session.** The LongMemEval benchmark shows 20% accuracy on multi-session questions — cross-session entity co-reference is a hard NLP problem independent of memory architecture.

---

## Roadmap

**v0.1** — Core working library
- SQLite vector store (numpy cosine)
- NetworkX graph store (JSON persistence)
- Hybrid RRF retrieval
- Temporal supersession
- 44 unit tests

**v0.2 (current)** — Multiple backends + async + benchmarks
- FAISS vector backend (142× faster at 10K scale)
- Neo4j graph backend
- AsyncEngram wrapper
- Latency benchmarks (1K and 10K scale)
- LongMemEval-style QA benchmark (25 examples, 5 categories, 3 systems)
- Stress benchmark with noise sessions and temporal_chain category
- Extended functional relation types: STUDIES, USES_LANGUAGE, DOES_EXERCISE, RELATIONSHIP_STATUS
- Fixed dedup false-positive for functional facts

**v1.0** — Production-ready
- PyPI trusted publishing
- MkDocs documentation site
- Full API reference
- CI/CD matrix (Python 3.10, 3.11, 3.12)
- LoCoMo / full LongMemEval (500 examples) evaluation

---

## License

Apache 2.0
