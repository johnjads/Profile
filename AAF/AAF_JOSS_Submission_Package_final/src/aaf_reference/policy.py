from dataclasses import dataclass

@dataclass(frozen=True)
class PolicyDecision:
    disposition: str
    allowed: bool
    reason: str

READ_ONLY_TOOLS = {'get_app_health', 'get_app_metadata', 'get_current_user_profile'}


def authorize_tool(risk_tier: str, tool_name: str, confirmed: bool = False) -> PolicyDecision:
    if tool_name in READ_ONLY_TOOLS:
        return PolicyDecision('allow', True, 'read-only tool')
    if risk_tier == 'high' and not confirmed:
        return PolicyDecision('deny', False, 'high-risk action requires explicit confirmation')
    if not confirmed:
        return PolicyDecision('confirm', False, 'state-changing action requires confirmation')
    return PolicyDecision('allow', True, 'confirmed state-changing action')
