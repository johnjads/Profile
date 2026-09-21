from __future__ import annotations

from collections.abc import Iterable


def coverage(answered: Iterable[bool]) -> float:
    values = list(answered)
    return sum(values) / len(values) if values else 0.0


def selective_accuracy(correct: Iterable[bool], answered: Iterable[bool]) -> float:
    pairs = [(c, a) for c, a in zip(correct, answered) if a]
    return sum(c for c, _ in pairs) / len(pairs) if pairs else 0.0


def claim_precision(supported_claims: int, factual_claims: int) -> float:
    return supported_claims / factual_claims if factual_claims else 0.0


def claim_recall(supported_required_claims: int, required_claims: int) -> float:
    return supported_required_claims / required_claims if required_claims else 0.0


def brier_score(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    pairs = [(float(p), int(y)) for p, y in zip(probabilities, outcomes)]
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else 0.0


def risk_at_threshold(confidences: Iterable[float], errors: Iterable[bool], threshold: float) -> float:
    selected = [e for c, e in zip(confidences, errors) if c >= threshold]
    return sum(selected) / len(selected) if selected else 0.0
