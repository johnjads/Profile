from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    query: str
    expected_route: str
    expected_app: str | None
    answered: bool
    correct: bool


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    rows: list[BenchmarkCase] = []
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows.append(BenchmarkCase(
                case_id=r['case_id'],
                query=r['query'],
                expected_route=r['expected_route'],
                expected_app=r['expected_app'] or None,
                answered=r.get('answered', '1') == '1',
                correct=r.get('correct', '1') == '1',
            ))
    return rows


def evaluate_router(cases: list[BenchmarkCase], route_fn: Callable[[str], object]) -> dict[str, float]:
    if not cases:
        return {"route_accuracy": 0.0}
    matched = 0
    for case in cases:
        decision = route_fn(case.query)
        route = getattr(decision, 'route', None)
        app = getattr(decision, 'app_key', None)
        matched += int(route == case.expected_route and app == case.expected_app)
    return {"route_accuracy": matched / len(cases)}
