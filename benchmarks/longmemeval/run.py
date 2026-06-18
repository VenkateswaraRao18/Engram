"""
Engram LongMemEval-style Benchmark
====================================
Evaluates three memory system variants on 20 hand-crafted examples that mirror
the LongMemEval taxonomy (knowledge_update, single_session, multi_session,
abstained_response).

Usage:
    python benchmarks/longmemeval/run.py
    python benchmarks/longmemeval/run.py --quick          # knowledge_update only
    python benchmarks/longmemeval/run.py --judge llm      # LLM judge (slower)
    python benchmarks/longmemeval/run.py --data path.json # custom dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Load .env if present (GEMINI_API_KEY etc.)
_env_path = os.path.join(os.path.dirname(__file__), "../../.env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from adapters import BaseAdapter, EngramAdapter, NaiveRAGAdapter, VectorOnlyAdapter
from score import accuracy, score_example

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DEFAULT = os.path.join(os.path.dirname(__file__), "data", "sample_20.json")

QUESTION_TYPES = [
    "knowledge_update",
    "temporal_chain",
    "single_session",
    "multi_session",
    "abstained_response",
]


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_adapter(
    adapter,
    examples: list[dict],
    use_llm_judge: bool = False,
) -> dict:
    """Run a single adapter on all examples. Returns per-example results."""
    results = []

    for i, ex in enumerate(examples):
        user_id = f"user_{ex['id']}"
        adapter.reset(user_id)

        # Ingest all sessions
        t_add_start = time.perf_counter()
        for session in ex["sessions"]:
            adapter.add_session(session["messages"], user_id, session["session_id"])
        t_add = time.perf_counter() - t_add_start

        # Answer the question
        t_ans_start = time.perf_counter()
        predicted = adapter.answer(ex["question"], user_id)
        t_ans = time.perf_counter() - t_ans_start

        correct = score_example(
            ex["question_type"],
            ex["question"],
            ex["answer"],
            predicted,
            use_llm_judge=use_llm_judge,
        )

        results.append(
            {
                "id": ex["id"],
                "question_type": ex["question_type"],
                "question": ex["question"],
                "ground_truth": ex["answer"],
                "predicted": predicted,
                "correct": correct,
                "add_ms": round(t_add * 1000, 1),
                "answer_ms": round(t_ans * 1000, 1),
            }
        )

        status = "PASS" if correct else "FAIL"
        print(
            f"    [{i+1:2d}/{len(examples)}] {status}  {ex['id']:<10} "
            f"gt={ex['answer']!r:<20}  pred={predicted[:60]!r}"
        )

    return results


def print_summary(adapter_name: str, results: list[dict]) -> dict:
    by_type: dict = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["correct"])

    overall = accuracy([r["correct"] for r in results])
    type_accs = {t: accuracy(by_type[t]) for t in QUESTION_TYPES if t in by_type}

    print(f"\n  {adapter_name}")
    print(f"  {'─' * 50}")
    for t, acc in type_accs.items():
        n = len(by_type[t])
        print(f"    {t:<25} {acc*100:5.1f}%  ({sum(by_type[t])}/{n})")
    print(f"    {'Overall':<25} {overall*100:5.1f}%  ({sum(r['correct'] for r in results)}/{len(results)})")

    return {"overall": overall, "by_type": type_accs}


def save_results(all_results: dict, summary: dict) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    raw_path = os.path.join(RESULT_DIR, f"longmemeval_{ts}.json")
    with open(raw_path, "w") as f:
        json.dump({"timestamp": ts, "results": all_results, "summary": summary}, f, indent=2)

    # Markdown table
    md_lines = [
        "# LongMemEval Results\n",
        f"> Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n",
        "",
        "## Accuracy by Question Type\n",
        "| System | knowledge_update | temporal_chain | single_session | multi_session | abstained_response | **Overall** |",
        "|--------|-----------------|----------------|----------------|---------------|-------------------|-------------|",
    ]
    for system, s in summary.items():
        bt = s["by_type"]
        row = (
            f"| {system} "
            f"| {bt.get('knowledge_update', 0)*100:.0f}% "
            f"| {bt.get('temporal_chain', 0)*100:.0f}% "
            f"| {bt.get('single_session', 0)*100:.0f}% "
            f"| {bt.get('multi_session', 0)*100:.0f}% "
            f"| {bt.get('abstained_response', 0)*100:.0f}% "
            f"| **{s['overall']*100:.0f}%** |"
        )
        md_lines.append(row)

    md_lines += [
        "",
        "## Notes",
        "- Dataset: 20 hand-crafted examples across 4 question types",
        "- Scoring: substring match for factual; abstention marker check for abstained_response",
        "- `knowledge_update` is Engram's primary differentiator (temporal supersession)",
    ]

    md_path = os.path.join(RESULT_DIR, f"longmemeval_{ts}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nResults saved to:\n  {raw_path}\n  {md_path}")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Only run knowledge_update examples (fastest)")
    parser.add_argument("--judge", choices=["substring", "llm"], default="substring",
                        help="Scoring method (default: substring match)")
    parser.add_argument("--data", default=DATA_DEFAULT,
                        help="Path to dataset JSON")
    parser.add_argument("--adapters", nargs="+",
                        choices=["engram", "vector_only", "naive_rag"],
                        default=["engram", "vector_only", "naive_rag"],
                        help="Which adapters to run")
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    if args.quick:
        dataset = [ex for ex in dataset if ex["question_type"] in ("knowledge_update", "temporal_chain")]
        print(f"Quick mode: {len(dataset)} knowledge_update + temporal_chain examples")

    use_llm_judge = args.judge == "llm"

    adapter_map = {
        "engram": EngramAdapter,
        "vector_only": VectorOnlyAdapter,
        "naive_rag": NaiveRAGAdapter,
    }

    # Wire Gemini key into adapters if available
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    BaseAdapter.gemini_api_key = gemini_key
    llm_backend = "gemini-2.5-flash" if gemini_key else "llama3.1 (local)"

    print("=" * 65)
    print("  Engram LongMemEval Benchmark")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  {len(dataset)} examples  ·  judge={args.judge}  ·  llm={llm_backend}")
    print("=" * 65)

    all_results: dict = {}
    summary: dict = {}

    for key in args.adapters:
        AdapterClass = adapter_map[key]
        adapter = AdapterClass()
        print(f"\n[{adapter.name}]")
        results = run_adapter(adapter, dataset, use_llm_judge=use_llm_judge)
        all_results[adapter.name] = results
        summary[adapter.name] = print_summary(adapter.name, results)

    # Final comparison table
    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    header = f"  {'System':<30} {'ku':>5} {'tc':>5} {'ss':>5} {'ms':>5} {'ab':>5} {'ALL':>6}"
    print(header)
    print("  " + "─" * 65)
    for system, s in summary.items():
        bt = s["by_type"]
        print(
            f"  {system:<30} "
            f"{bt.get('knowledge_update', 0)*100:4.0f}% "
            f"{bt.get('temporal_chain', 0)*100:4.0f}% "
            f"{bt.get('single_session', 0)*100:4.0f}% "
            f"{bt.get('multi_session', 0)*100:4.0f}% "
            f"{bt.get('abstained_response', 0)*100:4.0f}% "
            f"{s['overall']*100:5.0f}%"
        )
    print("  ku=knowledge_update  tc=temporal_chain  ss=single_session  ms=multi_session  ab=abstained")

    save_results(all_results, summary)


if __name__ == "__main__":
    main()
