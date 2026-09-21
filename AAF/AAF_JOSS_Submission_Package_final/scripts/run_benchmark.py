from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aaf_reference.benchmark import evaluate_router, load_cases
from aaf_reference.router import deterministic_route

KEYWORDS = {
    "lecturer ai": "lecturer",
    "ide ai": "ide",
    "property": "property",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluation/benchmark_template.csv")
    args = parser.parse_args()
    cases = load_cases(Path(args.input))
    metrics = evaluate_router(cases, lambda q: deterministic_route(q, KEYWORDS))
    for name, value in metrics.items():
        print(f"{name}={value:.4f}")


if __name__ == "__main__":
    main()
