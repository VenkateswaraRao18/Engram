# MemWeave — Project Specification & Build Documentation

> **Purpose of this document:** Complete technical specification for building MemWeave, an open-source long-term memory engine for LLM assistants. This document is designed to be used with Claude Code as the source-of-truth build plan. Work through it phase by phase (see §14).

---

## 1. Project Overview

**MemWeave** is a Python library (distributed on PyPI) that gives any LLM application persistent, long-term memory using a **hybrid storage architecture**: a vector store for semantic recall and a temporal knowledge graph for structured facts about users and entities.

**One-line pitch:** "Mem0-class long-term memory with first-class knowledge-graph reasoning, transparent benchmarks, and an optional fully-local mode."

**Why it exists (differentiators vs. Mem0 / Zep / LangMem):**
1. **Hybrid retrieval by default** — every query is answered by fusing vector search + graph traversal via Reciprocal Rank Fusion (RRF), not vector-only with graph bolted on.
2. **Temporal fact resolution** — facts carry validity intervals; contradictions are resolved by recency + confidence ("user moved from Tampa to Austin" invalidates the old fact rather than coexisting with it).
3. **Transparent, reproducible benchmarks** — the repo ships a benchmark harness for LongMemEval and LoCoMo with published numbers vs. Mem0/Zep/naive-RAG baselines.
4. **Local-first option** — runs with zero cloud dependencies (local embeddings via sentence-transformers, SQLite-backed vector store, embedded graph store) for privacy-sensitive and on-device-adjacent use cases.

**Naming note:** Before first release, verify the name `memweave` is free on PyPI (`pip index versions memweave` or check pypi.org). If taken, fallback candidates: `memloom`, `weavemem`, `recallweave`. The rest of this doc uses `memweave`.

---

## 2. Success Metrics (what "done and good" means)

| Metric | Target |
|---|---|
| LongMemEval accuracy (hybrid mode) | ≥ Mem0 published numbers on same setup; report honestly either way |
| LoCoMo QA accuracy | Beat naive long-context baseline; competitive with Mem0/Zep |
| p95 retrieval latency (local mode, 10K memories) | < 150 ms |
| p95 end-to-end `add()` latency (excluding LLM extraction call) | < 100 ms |
| Token cost per query vs. full-context stuffing | ≥ 80% reduction |
| Test coverage on core modules | ≥ 85% |
| PyPI release | v0.1.0 with docs, CI badge, benchmark table in README |

---

## 3. Tech Stack

- **Language:** Python 3.10+ (use `pyproject.toml`, hatchling or setuptools build backend)
- **Core deps (keep minimal):**
  - `pydantic` v2 — data models
  - `numpy` — vector math
  - `sqlite-vec` (or `faiss-cpu` as alternative backend) — local vector index
  - `networkx` — in-memory/embedded graph store (default); `neo4j` driver as optional extra
  - `sentence-transformers` — optional extra for local embeddings
  - `openai`, `anthropic` — optional extras for LLM extraction + embeddings
  - `tenacity` — retries on LLM calls
- **Optional extras (in `pyproject.toml`):** `memweave[neo4j]`, `memweave[qdrant]`, `memweave[local]`, `memweave[openai]`, `memweave[anthropic]`, `memweave[all]`
- **Dev/test:** `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff` (lint + format), `mypy`
- **CI/CD:** GitHub Actions (test matrix on 3.10/3.11/3.12, lint, type-check, publish-on-tag to PyPI via trusted publishing)
- **Docs:** MkDocs Material, deployed to GitHub Pages

---

## 4. Architecture Overview

```
                          ┌─────────────────────────────────────┐
                          │            MemWeave Client           │
                          │  add() · search() · get_context() ·  │
                          │  forget() · consolidate() · stats()  │
                          └───────┬─────────────────────┬───────┘
                                  │ write path           │ read path
                  ┌───────────────▼──────────┐   ┌──────▼───────────────┐
                  │     Extraction Engine     │   │   Hybrid Retriever    │
                  │ (LLM-based fact/entity/   │   │  vector top-k  ───┐   │
                  │  relation extraction)     │   │  graph traversal ─┤RRF│
                  └──────┬──────────┬─────────┘   │  recency boost ───┘   │
                         │          │             └──────┬───────────────┘
              ┌──────────▼───┐  ┌───▼──────────┐         │ reranked memories
              │ Vector Store │  │ Graph Store   │◄────────┘
              │ (sqlite-vec /│  │ (networkx /   │
              │  FAISS /     │  │  Neo4j)       │
              │  Qdrant)     │  │ temporal facts│
              └──────────────┘  └───────────────┘
                         ▲          ▲
                  ┌──────┴──────────┴─────────┐
                  │   Consolidation Engine     │
                  │ dedup · contradiction      │
                  │ resolution · decay ·       │
                  │ summarization of old mems  │
                  └────────────────────────────┘
```

**Write path:** raw message(s) → extraction engine (LLM) → list of candidate `Memory` objects + graph triples → consolidation check against existing memories (duplicate? contradiction? update?) → persist to vector store + graph store.

**Read path:** query → embed → vector top-k; in parallel, extract entities from query → graph neighborhood traversal → merge both candidate lists with RRF → recency/importance boost → return ranked `Memory` objects (and optionally a formatted context string).

---

## 5. Repository Structure

```
memweave/
├── pyproject.toml
├── README.md                  # pitch, quickstart, benchmark table, comparison table
├── LICENSE                    # Apache-2.0
├── CLAUDE.md                  # instructions for Claude Code (see §15)
├── .github/
│   └── workflows/
│       ├── ci.yml             # lint + typecheck + tests on PR/push
│       └── publish.yml        # PyPI trusted publishing on git tag
├── src/memweave/
│   ├── __init__.py            # public exports: Memory, MemWeave, configs
│   ├── client.py              # MemWeave class — the public API facade
│   ├── config.py              # pydantic Settings for all components
│   ├── models.py              # Memory, Fact, Entity, Relation, SearchResult
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── base.py            # Extractor protocol
│   │   ├── llm_extractor.py   # LLM-based extraction (prompt templates inside)
│   │   └── prompts.py         # all extraction/consolidation prompts
│   ├── embeddings/
│   │   ├── base.py            # Embedder protocol
│   │   ├── openai_embedder.py
│   │   ├── local_embedder.py  # sentence-transformers
│   ├── stores/
│   │   ├── vector/
│   │   │   ├── base.py        # VectorStore protocol
│   │   │   ├── sqlite_vec.py  # default local backend
│   │   │   ├── faiss_store.py
│   │   │   └── qdrant_store.py
│   │   └── graph/
│   │       ├── base.py        # GraphStore protocol
│   │       ├── networkx_store.py  # default embedded backend (with JSON persistence)
│   │       └── neo4j_store.py
│   ├── retrieval/
│   │   ├── hybrid.py          # RRF fusion, recency/importance boosting
│   │   └── rerank.py          # optional cross-encoder rerank (extra)
│   ├── consolidation/
│   │   ├── engine.py          # dedup, contradiction resolution, decay
│   │   └── policies.py        # configurable decay/importance policies
│   └── llm/
│       ├── base.py            # LLMProvider protocol
│       ├── openai_provider.py
│       └── anthropic_provider.py
├── benchmarks/
│   ├── README.md              # how to reproduce all numbers
│   ├── longmemeval/
│   │   ├── run.py
│   │   └── adapters/          # memweave, mem0, zep, naive_rag adapters
│   ├── locomo/
│   │   └── run.py
│   └── latency/
│       └── run.py             # p50/p95 at 1K/10K/100K memories
├── tests/
│   ├── unit/                  # per-module tests, LLM calls mocked
│   ├── integration/           # end-to-end with local backends
│   └── conftest.py
├── docs/                      # MkDocs
│   ├── index.md
│   ├── quickstart.md
│   ├── concepts.md            # memory types, temporal facts, consolidation
│   ├── backends.md
│   ├── benchmarks.md
│   └── api.md
└── examples/
    ├── chatbot_with_memory.py
    ├── multi_session_demo.py
    └── neo4j_backend.py
```

---

## 6. Core Data Models (`models.py`)

All models are Pydantic v2.

```python
class MemoryType(str, Enum):
    EPISODIC = "episodic"        # "User asked about flights to Tokyo on 2026-03-01"
    SEMANTIC = "semantic"        # "User is vegetarian"
    PROCEDURAL = "procedural"    # "User prefers answers in bullet points"

class Memory(BaseModel):
    id: str                      # uuid4
    user_id: str
    content: str                 # natural-language statement of the memory
    memory_type: MemoryType
    embedding: list[float] | None = None   # not serialized in API responses
    importance: float = 0.5      # 0..1, set by extractor, used in ranking & decay
    created_at: datetime
    last_accessed_at: datetime | None = None
    access_count: int = 0
    valid_from: datetime         # temporal validity
    valid_until: datetime | None = None    # None = currently valid
    superseded_by: str | None = None       # memory id that replaced this one
    source_message_ids: list[str] = []
    metadata: dict[str, Any] = {}

class Entity(BaseModel):
    id: str
    user_id: str
    name: str                    # canonical name, e.g. "Tampa"
    entity_type: str             # PERSON, PLACE, ORG, PREFERENCE, EVENT, OTHER
    aliases: list[str] = []

class Relation(BaseModel):
    id: str
    user_id: str
    source_entity_id: str
    relation_type: str           # LIVES_IN, WORKS_AT, PREFERS, DISLIKES, ...
    target_entity_id: str
    memory_id: str               # provenance — which memory asserted this
    valid_from: datetime
    valid_until: datetime | None = None
    confidence: float = 0.8

class SearchResult(BaseModel):
    memory: Memory
    score: float                 # fused RRF score
    vector_score: float | None
    graph_score: float | None
    retrieval_path: str          # "vector" | "graph" | "both"
```

**Key invariants:**
- A `Relation` is always backed by a `Memory` (provenance). Deleting a memory invalidates its relations.
- At most one *currently valid* relation per (source, relation_type) for functional relations (LIVES_IN, etc.) — enforced by the consolidation engine, configurable via `policies.py`.
- Embeddings are stored only in the vector store, never duplicated in the graph.

---

## 7. Component Specifications

### 7.1 Extraction Engine (`extraction/`)

**Input:** one or more chat messages (`role`, `content`, optional `timestamp`) + `user_id`.
**Output:** `ExtractionResult(memories: list[Memory], entities: list[Entity], relations: list[Relation])`.

Implementation: a single structured LLM call. Prompt requirements (put in `prompts.py`):
- Extract only **durable, user-relevant** facts; ignore small talk and assistant boilerplate.
- Classify each memory as episodic / semantic / procedural.
- Assign `importance` (0–1) with rubric in the prompt (identity facts ≈ 0.9; transient preferences ≈ 0.4).
- Emit entities + typed relations as JSON conforming to a schema; instruct the model to return **only JSON**, then parse with Pydantic. On parse failure, retry once with the validation error appended.
- Support batching: extraction over a window of N messages (default 6) rather than per-message, to reduce cost.

The `LLMProvider` protocol abstracts OpenAI/Anthropic; default model configurable, e.g. `claude-haiku-4-5` or `gpt-4o-mini` class models for cost.

### 7.2 Vector Store (`stores/vector/`)

Protocol:

```python
class VectorStore(Protocol):
    def add(self, memories: list[Memory]) -> None: ...
    def search(self, user_id: str, query_embedding: list[float],
               k: int = 20, filter_valid: bool = True) -> list[tuple[str, float]]: ...
    def delete(self, memory_ids: list[str]) -> None: ...
    def update_metadata(self, memory_id: str, **fields) -> None: ...
    def count(self, user_id: str) -> int: ...
```

- **Default backend: `sqlite-vec`** — single-file, zero-server, fits the local-first story. Store the full memory record in a `memories` table alongside the vec index; all queries filtered by `user_id` and validity.
- FAISS backend: index per user_id namespace + sidecar SQLite for metadata.
- Qdrant backend: payload filters for user_id/validity.

### 7.3 Graph Store (`stores/graph/`)

Protocol:

```python
class GraphStore(Protocol):
    def upsert_entities(self, entities: list[Entity]) -> None: ...
    def upsert_relations(self, relations: list[Relation]) -> None: ...
    def invalidate_relation(self, relation_id: str, at: datetime) -> None: ...
    def neighborhood(self, user_id: str, entity_names: list[str],
                     hops: int = 2, valid_at: datetime | None = None
                     ) -> list[Relation]: ...
    def find_conflicting(self, relation: Relation) -> list[Relation]: ...
```

- **Default backend: `networkx`** MultiDiGraph with JSON-on-disk persistence (atomic write). Good to ~100K edges; documented as such.
- **Neo4j backend** (`memweave[neo4j]`): same protocol over Cypher; entity merge on `(user_id, name)`; temporal validity as relationship properties. This is the backend to highlight for the Samsung use case.
- Entity resolution on upsert: case-insensitive name + alias match; optional embedding-similarity match (threshold 0.9) as a config flag.

### 7.4 Hybrid Retriever (`retrieval/hybrid.py`)

Algorithm for `search(user_id, query, k)`:
1. Embed query → vector store top-20 candidates with scores.
2. Extract entities from query (cheap LLM call OR keyword/noun-phrase fallback when `llm_query_entities=False`) → graph `neighborhood()` → backing memories of returned relations become graph candidates, scored by `confidence × hop_decay (0.7^(hop-1))`.
3. **Reciprocal Rank Fusion:** `score(m) = Σ 1/(60 + rank_i(m))` across the two lists.
4. Boosts: `final = rrf × (1 + w_recency × recency_decay(last_accessed_or_created)) × (1 + w_importance × importance)` with defaults `w_recency=0.3`, `w_importance=0.2`, exponential recency half-life 30 days (all configurable).
5. Optional cross-encoder rerank of top-10 (extra dependency, off by default).
6. Update `access_count`/`last_accessed_at` on returned memories (write-behind, non-blocking).

### 7.5 Consolidation Engine (`consolidation/`)

Runs inline on every `add()` (against top-5 similar existing memories) and as a batch job via `consolidate()`:

- **Dedup:** cosine similarity > 0.92 AND LLM-confirmed same fact → merge (keep earliest `created_at`, max importance, union sources).
- **Contradiction resolution:** new fact conflicts with existing valid fact about same entity/relation → close old fact (`valid_until = now`, `superseded_by = new_id`), keep both for history. Conflicts detected via `find_conflicting()` for graph-backed facts and an LLM judgment for free-text semantic memories.
- **Decay:** effective importance decays as `importance × exp(-λ · days_since_access)`; memories below threshold (default 0.05) are archived (excluded from retrieval, not deleted). λ derived from per-type half-lives in `policies.py` (episodic 30d, semantic 365d, procedural 180d — configurable).
- **Summarization:** batch job clusters archived episodic memories (by entity + time window) and replaces clusters of ≥5 with one LLM-written summary memory.

### 7.6 Public API (`client.py`)

```python
from memweave import MemWeave

mw = MemWeave.local(path="./memdb")          # zero-config local mode
# or fully configured:
mw = MemWeave(config=MemWeaveConfig(
    embedder="openai/text-embedding-3-small",
    llm="anthropic/claude-haiku-4-5",
    vector_store=QdrantConfig(...),
    graph_store=Neo4jConfig(uri=..., user=..., password=...),
))

mw.add(messages=[{"role": "user", "content": "btw I moved to Austin last month"}],
       user_id="u123")                        # → list[Memory] created/updated

results = mw.search("where does the user live?", user_id="u123", k=5)
ctx = mw.get_context("travel plans", user_id="u123", max_tokens=800)
                                              # → formatted string for prompt injection
mw.forget(memory_id=...)                      # hard delete + relation cleanup
mw.forget_user(user_id="u123")                # GDPR-style full wipe
report = mw.consolidate(user_id="u123")       # batch dedup/decay/summarize
stats = mw.stats(user_id="u123")              # counts, store sizes, type breakdown
```

Both sync and async (`AsyncMemWeave`) clients; implement async first, generate sync via thin wrappers.

---

## 8. Configuration (`config.py`)

Single `MemWeaveConfig` Pydantic settings object, env-var overridable (`MEMWEAVE_` prefix). Every tunable mentioned above (RRF k, boost weights, half-lives, similarity thresholds, extraction window, model names) lives here with documented defaults. No magic numbers in code.

---

## 9. Benchmarks (`benchmarks/`) — this is the resume payload, treat as first-class

1. **LongMemEval** (interactive long-term memory QA benchmark): adapter pattern — each system under test implements `add_session(messages)` + `answer(question)`. Adapters for: MemWeave (vector-only), MemWeave (hybrid), Mem0, Zep, naive RAG (chunk + embed all history), full-context stuffing. Report: accuracy per question type (single-session, multi-session, temporal reasoning, knowledge update), total tokens, mean/p95 latency.
2. **LoCoMo** (long conversational memory QA): same adapter pattern; report QA accuracy + token cost.
3. **Latency suite:** synthetic memories at 1K/10K/100K scale; report add() and search() p50/p95 per backend.
4. Output everything as JSON + a generated Markdown table committed to `docs/benchmarks.md` and embedded in README.
5. `benchmarks/README.md` must allow full reproduction with pinned versions and fixed seeds. Honest numbers only — if MemWeave loses a category, publish it with analysis.

(Claude Code note: download benchmark datasets from their official GitHub repos; check licenses before redistributing — link, don't vendor, if license is unclear.)

---

## 10. Testing Strategy

- **Unit tests** (no network): mock `LLMProvider` and `Embedder` with deterministic fakes; cover models, RRF math, decay math, temporal invalidation logic, entity resolution, each store backend against its protocol via a shared parametrized test suite (`tests/unit/test_store_contract.py`).
- **Integration tests:** full add→search→consolidate flow on local backends with fake LLM; one optional live-LLM test gated behind `MEMWEAVE_LIVE_TESTS=1`.
- **Property tests** (hypothesis, nice-to-have): temporal invariant — at any timestamp, at most one valid functional relation per (entity, type).
- Coverage gate in CI: 85% on `src/memweave`.

---

## 11. CI/CD (GitHub Actions)

- `ci.yml`: on push/PR → ruff check + format check, mypy, pytest with coverage on Python 3.10–3.12 matrix, upload coverage badge.
- `publish.yml`: on tag `v*` → build sdist+wheel, publish to PyPI via **trusted publishing** (no API token secrets), create GitHub release with changelog.
- Dependabot for dependency bumps.

---

## 12. Documentation Requirements

- **README:** 30-second pitch, animated terminal GIF of quickstart, comparison table (MemWeave vs Mem0 vs Zep vs LangMem: hybrid retrieval, temporal facts, local mode, benchmarks published, license), benchmark table, install matrix of extras.
- **MkDocs site:** quickstart, concepts (memory types, temporal facts, consolidation lifecycle diagram), backend guides (incl. Neo4j setup), benchmark methodology, full API reference (mkdocstrings).
- Docstrings on every public symbol (Google style).

---

## 13. Non-Goals (v0.x)

- Multi-tenant auth/server mode (library only; no hosted REST service yet — note as v1.0 idea).
- Image/audio memories.
- Automatic PII redaction (document as user responsibility; add `forget_user` for compliance).
- Windows-specific optimization (CI tests Linux + macOS; Windows best-effort).

---

## 14. Build Phases (work through these in order with Claude Code)

**Phase 0 — Scaffold (½ day):** repo structure, pyproject with extras, ruff/mypy/pytest config, CI workflow, empty protocols, `MemWeave.local()` stub that round-trips a hardcoded memory. *Exit: CI green, `pip install -e .` works.*

**Phase 1 — Core write path (1–2 days):** models, config, sqlite-vec store, networkx graph store, LLM extractor with mocked tests, `add()` end-to-end. *Exit: integration test stores extracted memories + relations from a fake conversation.*

**Phase 2 — Read path (1–2 days):** embedders, vector search, graph neighborhood traversal, RRF fusion, boosts, `search()` + `get_context()`. *Exit: hybrid beats vector-only on a small handcrafted eval set committed to tests.*

**Phase 3 — Consolidation (1–2 days):** dedup, contradiction resolution with temporal invalidation, decay, `consolidate()`, `forget()`/`forget_user()`. *Exit: "moved from Tampa to Austin" scenario test passes — old fact superseded, queries return Austin.*

**Phase 4 — Backends & async (1–2 days):** Qdrant + FAISS + Neo4j backends passing the shared contract suite; `AsyncMemWeave`. *Exit: contract suite green on all backends (Neo4j via docker-compose in CI service container).*

**Phase 5 — Benchmarks (2–3 days):** LongMemEval + LoCoMo harness with all adapters, latency suite, generated results tables. *Exit: reproducible numbers committed to docs.*

**Phase 6 — Polish & ship (1–2 days):** README with tables, MkDocs site, examples, CHANGELOG, tag v0.1.0, PyPI trusted publish. *Exit: `pip install memweave` works from PyPI.*

---

## 15. CLAUDE.md (copy this into the repo root for Claude Code)

```markdown
# MemWeave — Claude Code Instructions

## What this project is
Open-source Python library: long-term memory for LLM apps via hybrid
vector + temporal knowledge graph storage. Full spec in MEMWEAVE_SPEC.md —
read the relevant section before implementing any module.

## Conventions
- Python 3.10+, src layout, Pydantic v2 models, full type hints, mypy clean.
- ruff for lint + format (line length 100). Run `ruff check --fix && ruff format`.
- Google-style docstrings on all public symbols.
- No magic numbers — every tunable goes in config.py with a documented default.
- Async-first internals; sync client wraps async.
- Never log memory contents at INFO level (privacy); DEBUG only.

## Commands
- Install: `pip install -e ".[all,dev]"`
- Test: `pytest -x -q` (unit only: `pytest tests/unit -x -q`)
- Lint/type: `ruff check . && mypy src/memweave`
- Docs preview: `mkdocs serve`

## Testing rules
- Every new module gets unit tests in the same PR. Mock all LLM/network calls
  in unit tests (fixtures in tests/conftest.py: FakeLLM, FakeEmbedder).
- Store backends must pass tests/unit/test_store_contract.py.
- Keep coverage ≥ 85% on src/memweave.

## Workflow
- Follow build phases in MEMWEAVE_SPEC.md §14; do not skip ahead.
- Small commits, conventional commit messages (feat:, fix:, test:, docs:).
- After each phase, run the full test suite + lint before moving on.
```

---

## 16. Resume Bullets This Project Should Produce (keep these in sight)

- "Created MemWeave, an open-source long-term memory engine for LLM assistants combining vector search with a temporal knowledge graph (Neo4j), fused via Reciprocal Rank Fusion."
- "Achieved X% accuracy on LongMemEval — outperforming vector-only retrieval by Y% on multi-session and temporal-reasoning questions — at Z ms p95 latency over 10K memories."
- "Reduced per-query token cost by N% vs. full-context baselines while maintaining answer accuracy on LoCoMo."
- "Shipped with 85%+ test coverage, GitHub Actions CI/CD, and PyPI trusted publishing; reached [downloads] in first month."

Fill in real numbers from `benchmarks/` output — never estimate.
