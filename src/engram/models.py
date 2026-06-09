from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content: str
    memory_type: MemoryType
    embedding: Optional[list[float]] = None
    importance: float = 0.5
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    superseded_by: Optional[str] = None
    source_message_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    entity_type: str  # PERSON, PLACE, ORG, PREFERENCE, EVENT, OTHER
    aliases: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    source_entity_id: str
    relation_type: str
    target_entity_id: str
    memory_id: str
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    confidence: float = 0.8


class SearchResult(BaseModel):
    memory: Memory
    score: float
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    retrieval_path: str  # "vector" | "graph" | "both"


class ExtractionResult(BaseModel):
    memories: list[Memory] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
