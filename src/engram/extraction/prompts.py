from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction assistant. Your job is to extract durable, user-relevant facts from a conversation.

Return ONLY valid JSON — no markdown, no code fences, no extra text. The JSON must match this schema exactly:
{
  "memories": [
    {"content": "...", "memory_type": "semantic|episodic|procedural", "importance": 0.0-1.0}
  ],
  "entities": [
    {"name": "...", "entity_type": "PERSON|PLACE|ORG|PREFERENCE|EVENT|OTHER", "aliases": []}
  ],
  "relations": [
    {"source_entity_name": "...", "relation_type": "LIVES_IN|WORKS_AT|PREFERS|DISLIKES|MOVED_TO|OTHER", "target_entity_name": "...", "confidence": 0.0-1.0}
  ]
}

Rules:
- Extract only durable, user-relevant facts (not transient conversational filler)
- memory_type: "semantic" for facts/knowledge, "episodic" for events/experiences, "procedural" for skills/how-to
- Importance guidelines:
  - Identity facts (name, core identity): ~0.9
  - Location (where user lives/works): ~0.85
  - Preferences, relationships: ~0.75
  - Past events: ~0.6
  - Transient or low-signal facts: ~0.4
- CRITICAL: Always use "User" as the canonical entity name for the human speaker (even if their real name is known). Never use their real name as the source entity in relations — always use "User". You MUST include "User" as an entity whenever relations are generated.
- Memory content should be full sentences like "User lives in Tampa" not just place names like "Tampa".
- For location changes, emit BOTH a MOVED_TO relation AND a LIVES_IN relation (both with source_entity_name = "User")
- If nothing durable is said, return {"memories": [], "entities": [], "relations": []}
- Do not hallucinate facts not present in the conversation
"""


def format_messages(messages: list[dict]) -> str:
    """Format a list of message dicts as a readable conversation string."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
