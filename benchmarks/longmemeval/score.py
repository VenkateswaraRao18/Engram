from __future__ import annotations

import ollama

ABSTAIN_MARKERS = [
    "i don't know",
    "i do not know",
    "not mentioned",
    "no information",
    "cannot determine",
    "not provided",
    "unknown",
    "not available",
    "not stated",
    "not sure",
]


def score_factual(predicted: str, ground_truth: str) -> bool:
    """
    True if the prediction contains the ground truth answer.
    - Exact phrase match (fast path)
    - Single-word answers: substring match
    - Multi-word answers: ALL keywords (len > 3) must appear in prediction
      to avoid false positives like "science" matching "computer science" when
      the answer is "data science"
    """
    pred_lower = predicted.lower()
    gt_lower = ground_truth.lower()
    if gt_lower in pred_lower:
        return True
    keywords = [w for w in gt_lower.split() if len(w) > 3]
    if not keywords:
        return False
    if len(keywords) == 1:
        return keywords[0] in pred_lower
    # Multi-keyword: require all to match to avoid generic-word false positives
    return all(kw in pred_lower for kw in keywords)


def score_abstained(predicted: str) -> bool:
    """
    True if the response correctly abstains from answering.
    """
    pred_lower = predicted.lower()
    return any(marker in pred_lower for marker in ABSTAIN_MARKERS)


def llm_judge(question: str, ground_truth: str, predicted: str, model: str = "llama3.1") -> bool:
    """
    Use an LLM as a judge for complex answers that can't be scored by substring match.
    Returns True if the prediction is semantically correct.
    """
    prompt = (
        f"Question: {question}\n"
        f"Correct answer: {ground_truth}\n"
        f"System answer: {predicted}\n\n"
        "Does the system answer correctly address the question with the right information? "
        "Respond with exactly one word: CORRECT or INCORRECT."
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    verdict = response.message.content.strip().upper()
    return "CORRECT" in verdict


def score_example(
    question_type: str,
    question: str,
    ground_truth: str,
    predicted: str,
    use_llm_judge: bool = False,
) -> bool:
    if question_type == "abstained_response":
        return score_abstained(predicted)
    # temporal_chain and knowledge_update are both factual — same scoring
    if use_llm_judge:
        return llm_judge(question, ground_truth, predicted)
    return score_factual(predicted, ground_truth)


def accuracy(results: list[bool]) -> float:
    if not results:
        return 0.0
    return sum(results) / len(results)
