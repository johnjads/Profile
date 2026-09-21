# Production code provenance 

The production AAF Assistant is hosted within AI Agent Folio. To make implementation claims reproducible, replace the placeholders below with the actual public source release, or explicitly document why the production source cannot be released.

- Repository: `https://github.com/johnjads/Profile/tree/master/AAF`
- Commit/tag: `initial folder commit: AAF~52e4233ed014124fe90276aafd28df802a58d9e4`
- Runtime/model: `deepseek-r1:7b / qwen2.5:7b` / `0.30.10`
- Hardware: `4 OCPU, 24GB RAM, Ubuntu 22.04, aarch64/ARM64`
- Quantization: `Q4_K_M`
- Relevant source paths: `auth_app/main.py`, `aaf_distill.py`, `auth.db` schema, `evolved_chatbot_addendum.txt`
- Deployment evidence: `https://auth.aiagentfolio.com/dashboard (bottom corner right chat bubble)`
