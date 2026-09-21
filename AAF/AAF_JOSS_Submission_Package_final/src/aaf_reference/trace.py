from dataclasses import dataclass, asdict
import json

@dataclass
class TraceRecord:
    interaction_id: str
    route: str
    route_confidence: float
    risk_tier: str
    model_id: str
    prompt_version: str
    verification_outcome: str
    latency_ms: float
    final_disposition: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
