from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ...models import Memory, MemoryType

CREATE_TABLE_SQL = """
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
    embedding TEXT
)
"""


def _serialize_dt(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO string or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO string to datetime or None."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


class SQLiteVectorStore:
    """Pure SQLite + numpy cosine similarity vector store."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(CREATE_TABLE_SQL)

    def add(self, memories: list[Memory]) -> None:
        """Insert memories into the store."""
        with self._conn:
            for mem in memories:
                emb_json = json.dumps(mem.embedding) if mem.embedding is not None else None
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (id, user_id, content, memory_type, importance,
                     created_at, last_accessed_at, access_count,
                     valid_from, valid_until, superseded_by,
                     source_message_ids, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mem.id,
                        mem.user_id,
                        mem.content,
                        mem.memory_type.value,
                        mem.importance,
                        _serialize_dt(mem.created_at),
                        _serialize_dt(mem.last_accessed_at),
                        mem.access_count,
                        _serialize_dt(mem.valid_from),
                        _serialize_dt(mem.valid_until),
                        mem.superseded_by,
                        json.dumps(mem.source_message_ids),
                        json.dumps(mem.metadata),
                        emb_json,
                    ),
                )

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 20,
        filter_valid: bool = True,
    ) -> list[tuple[str, float]]:
        """Return top-k (memory_id, cosine_similarity) pairs."""
        if filter_valid:
            sql = """
                SELECT id, embedding FROM memories
                WHERE user_id = ?
                  AND embedding IS NOT NULL
                  AND valid_until IS NULL
                  AND superseded_by IS NULL
            """
        else:
            sql = """
                SELECT id, embedding FROM memories
                WHERE user_id = ?
                  AND embedding IS NOT NULL
            """
        rows = self._conn.execute(sql, (user_id,)).fetchall()
        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scored: list[tuple[str, float]] = []
        for row in rows:
            emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            emb = emb / norm
            score = float(np.dot(q, emb))
            scored.append((row["id"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def delete(self, memory_ids: list[str]) -> None:
        """Delete memories by ID."""
        if not memory_ids:
            return
        placeholders = ",".join("?" * len(memory_ids))
        with self._conn:
            self._conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", memory_ids
            )

    def update_metadata(self, memory_id: str, **fields) -> None:
        """Update arbitrary fields on a memory record."""
        if not fields:
            return
        set_parts = []
        values = []
        for k, v in fields.items():
            set_parts.append(f"{k} = ?")
            # Serialize datetime values
            if isinstance(v, datetime):
                v = v.isoformat()
            values.append(v)
        values.append(memory_id)
        sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE id = ?"
        with self._conn:
            self._conn.execute(sql, values)

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def get_all_valid(self, user_id: str) -> list[Memory]:
        """Get all non-superseded, non-expired memories for a user."""
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ?
              AND valid_until IS NULL
              AND superseded_by IS NULL
            """,
            (user_id,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self, user_id: str) -> int:
        """Count total memories (including invalid) for a user."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["cnt"]

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Deserialize a database row into a Memory object."""
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
