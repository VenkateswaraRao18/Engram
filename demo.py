from __future__ import annotations

"""
Engram demo: Tampa -> Austin supersession scenario.
Uses real Ollama LLM (llama3.1) and embeddings (nomic-embed-text).
"""

import os
import shutil

from engram import Engram


def separator(title: str = "") -> None:
    width = 60
    if title:
        padding = (width - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def main():
    # Clean up previous demo_db
    if os.path.exists("./demo_db"):
        shutil.rmtree("./demo_db")

    separator("Engram v0 Demo")
    print("Initializing Engram with Ollama (llama3.1 + nomic-embed-text)...")
    mw = Engram.local("./demo_db")
    print("Ready.\n")

    user_id = "alex"

    # ---- Message 1: Tampa ----
    separator("Step 1: Alex introduces himself (Tampa)")
    msg1 = [
        {"role": "user", "content": "Hi, I'm Alex. I live in Tampa, Florida. I'm a software engineer."}
    ]
    print(f"User: {msg1[0]['content']}\n")
    print("Extracting memories...")
    stored1 = mw.add(msg1, user_id=user_id)
    if stored1:
        print(f"Stored {len(stored1)} memory/memories:")
        for m in stored1:
            print(f"  [{m.memory_type.value}] (importance={m.importance:.2f}) {m.content}")
            print(f"    id: {m.id}")
    else:
        print("  No new memories extracted.")

    # Show graph relations
    rels = mw._graph_store.neighborhood(user_id, ["User", "Alex"], hops=2)
    if rels:
        print("\nGraph relations:")
        for r in rels:
            src_name = mw._graph_store._entities.get(r.source_entity_id, None)
            tgt_name = mw._graph_store._entities.get(r.target_entity_id, None)
            src = src_name.name if src_name else r.source_entity_id
            tgt = tgt_name.name if tgt_name else r.target_entity_id
            status = "VALID" if r.valid_until is None else "INVALIDATED"
            print(f"  {src} --[{r.relation_type}]--> {tgt}  [{status}]")
    print()

    # ---- Message 2: Austin (move) ----
    separator("Step 2: Alex moves to Austin")
    msg2 = [
        {"role": "user", "content": "I just moved to Austin, Texas last month. Loving it here! The tech scene is amazing."}
    ]
    print(f"User: {msg2[0]['content']}\n")
    print("Extracting memories...")
    stored2 = mw.add(msg2, user_id=user_id)
    if stored2:
        print(f"Stored {len(stored2)} memory/memories:")
        for m in stored2:
            print(f"  [{m.memory_type.value}] (importance={m.importance:.2f}) {m.content}")
            print(f"    id: {m.id}")
    else:
        print("  No new memories extracted.")

    # Check if Tampa memory was superseded
    if stored1:
        tampa_mem = mw._vector_store.get(stored1[0].id)
        if tampa_mem and tampa_mem.superseded_by:
            print(f"\nSupersession detected!")
            print(f"  Tampa memory (id: {tampa_mem.id[:8]}...) -> superseded_by: {tampa_mem.superseded_by[:8]}...")
        elif tampa_mem and tampa_mem.valid_until:
            print(f"\nTampa memory invalidated at: {tampa_mem.valid_until}")

    # Show updated graph relations
    rels2 = mw._graph_store.neighborhood(user_id, ["User", "Alex"], hops=2)
    if rels2:
        print("\nActive graph relations:")
        for r in rels2:
            src_ent = mw._graph_store._entities.get(r.source_entity_id)
            tgt_ent = mw._graph_store._entities.get(r.target_entity_id)
            src = src_ent.name if src_ent else r.source_entity_id
            tgt = tgt_ent.name if tgt_ent else r.target_entity_id
            print(f"  {src} --[{r.relation_type}]--> {tgt}  (confidence={r.confidence:.2f})")

    # Also show all relations including invalidated ones
    all_rels = list(mw._graph_store._relations.values())
    invalidated = [r for r in all_rels if r.valid_until is not None]
    if invalidated:
        print("\nInvalidated relations:")
        for r in invalidated:
            src_ent = mw._graph_store._entities.get(r.source_entity_id)
            tgt_ent = mw._graph_store._entities.get(r.target_entity_id)
            src = src_ent.name if src_ent else r.source_entity_id
            tgt = tgt_ent.name if tgt_ent else r.target_entity_id
            print(f"  {src} --[{r.relation_type}]--> {tgt}  [INVALIDATED at {r.valid_until}]")
    print()

    # ---- Search ----
    separator("Step 3: Search for Alex's location")
    query = "Where does Alex live?"
    print(f"Query: {query}\n")
    results = mw.search(query, user_id=user_id, k=5)
    if results:
        print(f"Found {len(results)} result(s):")
        for i, r in enumerate(results, 1):
            mem = r.memory
            # Check validity labels
            label = "VALID"
            if mem.superseded_by:
                label = f"SUPERSEDED -> {mem.superseded_by[:8]}..."
            elif mem.valid_until:
                label = f"EXPIRED at {mem.valid_until}"
            print(f"  {i}. [{r.retrieval_path}] score={r.score:.4f} [{label}]")
            print(f"     {mem.content}")
            if r.vector_score is not None:
                print(f"     vector_score={r.vector_score:.4f}", end="")
            if r.graph_score is not None:
                print(f"  graph_score={r.graph_score:.4f}", end="")
            print()
    else:
        print("  No results found.")
    print()

    # ---- get_context ----
    separator("Step 4: get_context()")
    context = mw.get_context("Tell me about Alex", user_id=user_id, max_tokens=800)
    if context:
        print("Context string for LLM injection:")
        print(context)
    else:
        print("  No context available.")
    print()

    # ---- Stats ----
    separator("Step 5: Stats")
    s = mw.stats(user_id)
    print(f"Total memories in store: {s['total_memories']}")
    valid = mw._vector_store.get_all_valid(user_id)
    print(f"Currently valid memories: {len(valid)}")
    print("\nAll valid memories:")
    for m in valid:
        print(f"  [{m.memory_type.value}] {m.content}")

    separator("Demo complete")


if __name__ == "__main__":
    main()
