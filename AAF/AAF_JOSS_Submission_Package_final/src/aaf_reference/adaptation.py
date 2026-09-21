from dataclasses import dataclass

@dataclass(frozen=True)
class CandidateArtifact:
    version: str
    guidance: str
    source_feedback_ids: tuple[str, ...]

@dataclass(frozen=True)
class RegressionDecision:
    promote: bool
    reason: str


def sanitize_guidance(guidance: str) -> str:
    """Treat feedback as data: remove blank lines and obvious prompt-control directives."""
    banned = ("ignore previous", "reveal system prompt", "disable safety")
    cleaned = " ".join(line.strip() for line in guidance.splitlines() if line.strip())
    lowered = cleaned.casefold()
    if any(term in lowered for term in banned):
        raise ValueError("candidate contains a prohibited control directive")
    return cleaned


def build_candidate(version: str, guidance: str, feedback_ids: list[str]) -> CandidateArtifact:
    clean = sanitize_guidance(guidance)
    if not clean:
        raise ValueError('guidance must not be empty')
    return CandidateArtifact(version, clean, tuple(feedback_ids))


def regression_gate(baseline_ok: bool, candidate_ok: bool, security_ok: bool, latency_ok: bool) -> RegressionDecision:
    checks = {"baseline": baseline_ok, "candidate": candidate_ok, "security": security_ok, "latency": latency_ok}
    if all(checks.values()):
        return RegressionDecision(True, "all release gates passed")
    failed = ", ".join(k for k, v in checks.items() if not v)
    return RegressionDecision(False, f"release gate failed: {failed}")
