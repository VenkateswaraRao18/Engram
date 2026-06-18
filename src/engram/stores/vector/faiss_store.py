from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Optional

import faiss
import numpy as np

from ...models import Memory, MemoryType
from .sqlite_store import _parse_dt, _serialize_dt

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    superseded_by TEXT,
    source_message_ids TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding TEXT,
    faiss_id INTEGER
)
"""


class FAISSVectorStore:
    """SQLite metadata + FAISS IndexFlatIP for fast cosine similarity search."""

    def __init__(self, db_path: str = ":memory:", dimensions: int = 768):
        self._db_path = db_path
        self._dimensions = dimensions
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._id_to_faiss: dict[str, int] = {}
        self._faiss_to_id: dict[int, str] = {}
        self._faiss_counter = 0
        self._build_index()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_TABLE_SQL)

    def _build_index(self) -> None:
        """Build FAISS index from all stored embeddings."""
        base = faiss.IndexFlatIP(self._dimensions)
        self._index = faiss.IndexIDMap2(base)
        self._id_to_faiss.clear()
        self._faiss_to_id.clear()
        self._faiss_counter = 0

        rows = self._conn.execute(
            "SELECT id, embedding, faiss_id FROM memories "
            "WHERE embedding IS NOT NULL AND faiss_id IS NOT NULL"
        ).fetchall()

        if not rows:
            return

        vecs, ids = [], []
        for row in rows:
            fid = row["faiss_id"]
            vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vecs.append(vec / norm)
            ids.append(fid)
            self._id_to_faiss[row["id"]] = fid
            self._faiss_to_id[fid] = row["id"]
            self._faiss_counter = max(self._faiss_counter, fid + 1)

        if vecs:
            self._index.add_with_ids(
                np.array(vecs, dtype=np.float32),
                np.array(ids, dtype=np.int64),
            )

    def add(self, memories: list[Memory]) -> None:
        with self._conn:
            for mem in memories:
                faiss_id = None
                if mem.embedding:
                    vec = np.array(mem.embedding, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        faiss_id = self._faiss_counter
                        self._faiss_counter += 1
                        self._index.add_with_ids(
                            (vec / norm).reshape(1, -1).astype(np.float32),
                            np.array([faiss_id], dtype=np.int64),
                        )
                        self._id_to_faiss[mem.id] = faiss_id
                        self._faiss_to_id[faiss_id] = mem.id

                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (id, user_id, content, memory_type, importance,
                     created_at, last_accessed_at, access_count,
                     valid_from, valid_until, superseded_by,
                     source_message_ids, metadata, embedding, faiss_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mem.id, mem.user_id, mem.content, mem.memory_type.value,
                        mem.importance, _serialize_dt(mem.created_at),
                        _serialize_dt(mem.last_accessed_at), mem.access_count,
                        _serialize_dt(mem.valid_from), _serialize_dt(mem.valid_until),
                        mem.superseded_by, json.dumps(mem.source_message_ids),
                        json.dumps(mem.metadata),
                        json.dumps(mem.embedding) if mem.embedding else None,
                        faiss_id,
                    ),
                )

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 20,
        filter_valid: bool = True,
    ) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []

        where = "user_id = ? AND faiss_id IS NOT NULL"
        if filter_valid:
            where += " AND valid_until IS NULL AND superseded_by IS NULL"

        rows = self._conn.execute(
            f"SELECT id, faiss_id FROM memories WHERE {where}", (user_id,)
        ).fetchall()

        if not rows:
            return []

        valid_faiss_ids = {row["faiss_id"] for row in rows}
        faiss_to_mem = {row["faiss_id"]: row["id"] for row in rows}

        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = (q / norm).reshape(1, -1)

        # Over-fetch since FAISS doesn't know about user_id boundaries
        fetch_k = min(max(k * 5, 50), self._index.ntotal)
        scores, faiss_ids = self._index.search(q, fetch_k)

        results = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid == -1:
                continue
            if fid in valid_faiss_ids:
                results.append((faiss_to_mem[fid], float(score)))
                if len(results) >= k:
                    break

        return results

    def delete(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" * len(memory_ids))
        with self._conn:
            self._conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", memory_ids
            )
        self._build_index()

    def update_metadata(self, memory_id: str, **fields) -> None:
        if not fields:
            return
        set_parts, values = [], []
        for field, value in fields.items():
            set_parts.append(f"{field} = ?")
            if isinstance(value, datetime):
                value = value.isoformat()
            values.append(value)
        values.append(memory_id)
        with self._conn:
            self._conn.execute(
                f"UPDATE memories SET {', '.join(set_parts)} WHERE id = ?", values
            )

    def get(self, memory_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_all_valid(self, user_id: str) -> list[Memory]:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE user_id = ? "
            "AND valid_until IS NULL AND superseded_by IS NULL",
            (user_id,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["cnt"]

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            importance=row["importance"],
            created_at=_parse_dt(row["created_at"]),
            last_accessed_at=_parse_dt(row["last_accessed_at"]),
            access_count=row["access_count"],
            valid_from=_parse_dt(row["valid_from"]),
            valid_until=_parse_dt(row["valid_until"]),
            superseded_by=row["superseded_by"],
            source_message_ids=json.loads(row["source_message_ids"]),
            metadata=json.loads(row["metadata"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] is not None else None,
        )
