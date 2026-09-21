from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.request import Request, urlopen

@dataclass(frozen=True)
class OllamaResponse:
    text: str
    raw: dict


def generate(base_url: str, model: str, prompt: str, timeout_s: float = 60.0) -> OllamaResponse:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = Request(
        base_url.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as response:
        raw = json.loads(response.read().decode("utf-8"))
    return OllamaResponse(text=str(raw.get("response", "")), raw=raw)
