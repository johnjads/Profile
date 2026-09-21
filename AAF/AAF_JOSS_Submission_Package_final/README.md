# AAF Assistant: Reliability-Centered LLMOps Research Software

This repository is the open research/reproducibility companion for the AAF Assistant study and is intended to support a JOSS submission. The production system under study is deployed within [AI Agent Folio](https://aiagentfolio.com/); the authenticated dashboard is `https://auth.aiagentfolio.com/dashboard`.

## Author-completion fields

- `Johnson Ang D. S.`
- 'SUSS, Singapore, ST Engineering, Singapore, Aston University, England, Birmingham City University, England, University of Essex, England, University of Portsmouth, England, Murdoch University, Australia'
- `0009-0005-9149-1742`
- `https://github.com/johnjads/Profile/tree/master/`
- `https://github.com/johnjads/Profile/tree/master/AAF`
- `initial folder commit: AAF~52e4233ed014124fe90276aafd28df802a58d9e4`
- `deepseek-r1:7b / qwen2.5:7b, Q4_K_M`
- ``0.30.10`
- `4 OCPU, 24GB RAM, Ubuntu 22.04, aarch64/ARM64`
- `- Total set size: 500 labeled examples
    Strata: stratified by task category (e.g. code-gen, Q&A, summarization) 
    and by difficulty tier (easy/medium/hard), roughly balanced per cell
    Split:
     Development: 60% (300 examples) — used for prompt/config iteration
     Holdout: 30% (150 examples) — held back, single evaluation pass only
     Security/red-team: 10% (50 examples) — adversarial/edge-case prompts, 
        never used for tuning, reserved for final sign-off
     Split method: stratified random sampling, fixed seed for reproducibility`
- `Research impact statement. AAF is deployed as part of the AI Agent Folio platform and serves as the operational software substrate for research into reliability-centered LLMOps. The implementation is used to investigate deterministic routing, context grounding, feedback-driven prompt adaptation, local-model inference, verification, and bounded tool execution under resource-constrained deployment. The associated research artifacts provide executable implementations, tests, trace structures, benchmark definitions, and reproducibility materials. `
- `<ARTIFACT_DOI or not yet minted>`
- `NONE`
- `https://auth.aiagentfolio.com/dashboard (bottom corner right chat bubble)`

## Important provenance boundary

This package must not be described as the private production AAF source tree unless the canonical production repository is actually public and the release is identical. Production implementation claims should be mapped to `https://github.com/johnjads/Profile/tree/master//AAF` and `AAF~52e4233ed014124fe90276aafd28df802a58d9e4` when available.

## Local development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e . pytest
pytest -q
python scripts/run_benchmark.py --input evaluation/benchmark_template.csv
```

## JOSS paper

The authoritative JOSS article is `paper/paper.md` with `paper/paper.bib`. The Word/PDF research manuscript belongs in the supporting research package, not as the JOSS article.

## Citation

The final repository should update `CITATION.cff` with the final authors, version, and DOI. `<ARTIFACT_DOI or not yet minted>` should be replaced after an archival release is created.
