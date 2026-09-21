from dataclasses import dataclass

@dataclass(frozen=True)
class ClaimCheck:
    supported: bool
    evidence_id: str | None
    reason: str


def verify_claim(claim: str, evidence: dict[str, str]) -> ClaimCheck:
    c = claim.casefold().strip()
    for evidence_id, text in evidence.items():
        if c and c in text.casefold():
            return ClaimCheck(True, evidence_id, 'claim text found in evidence')
    return ClaimCheck(False, None, 'no direct lexical support found')
