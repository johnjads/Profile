# Publication figure construction specification

Use this specification when rebuilding the figures manually in draw.io, Visio, Illustrator, Inkscape, or Word.

## Global style

- Monochrome: black strokes and black text on white background.
- No color coding, gradients, shadows, 3-D effects, icons, or decorative fills.
- Stroke width: approximately 0.8-1.1 pt at final journal size.
- Box corners: mildly rounded (about 6-10 pt radius), or square if the venue prefers strict engineering notation.
- Arrowheads: **small**. Target arrowhead length about 3-5 mm at final printed width and no more than roughly 40-50% of a node height. Use a simple filled triangular or open V arrowhead; do not use oversized PowerPoint arrows.
- Connectors: orthogonal/horizontal/vertical by default; avoid long diagonal crossings.
- Font: Arial/Helvetica or the journal's permitted sans-serif. Use 8.5-10.5 pt body labels at final size; 11-12 pt for figure titles if titles are included inside the artwork.
- Keep at least one text-line height of internal padding inside each box.
- Avoid text touching borders or arrows.
- Prefer one-way flow from left to right or top to bottom.
- If feedback creates a loop, route the return path around the outside of the main flow rather than through the center of other nodes.

## Figure 1: current runtime

User/browser -> AAF gateway -> deterministic application routing -> trusted context -> deepseek-r1:7b -> Qwen2.5-7B/Ollama -> answer.

Feedback path should be a separate lower loop: answer -> feedback record -> nightly critique -> persistent prompt addendum -> next startup.

## Figure 2: proposed reliability-centered architecture

Normalize/classify -> reliability router -> one of three paths: verified FAQ, evidence retrieval/validation, bounded agent. All paths converge on generation/action, then verification/policy gate, then final disposition. A shared control plane should sit beneath these paths and connect policy, authorization, provenance, observability, versioning, and rollback. The offline adaptation loop should connect feedback -> critique -> candidate -> regression/safety gate -> promotion.

## Figure 3: feedback adaptation

Production interaction -> feedback/trace capture -> teacher-LLM critique -> candidate artifact -> sanitization/policy checks -> frozen regression + safety suite -> versioned canary -> production.

Feedback should enter a queue; it must not directly mutate the live prompt.

## Figure 4: bounded agent

User goal -> planner/router -> policy/risk gate -> schema-valid call -> tool adapter/executor -> tool-result validation -> confirm/deny/escalate -> final answer/abstention. Add an explicit bounded retry loop and show MCP, if used, as an interoperability layer rather than the policy engine.

## Output formats

Retain: SVG, PDF, and PNG. For editable office workflows, also provide EMF where supported. Keep the DOT/Mermaid source beside each figure so reviewers can reconstruct the layout.
