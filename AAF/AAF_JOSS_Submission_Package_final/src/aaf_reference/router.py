from dataclasses import dataclass

@dataclass(frozen=True)
class RouteDecision:
    route: str
    app_key: str | None
    confidence: float
    reason: str


def deterministic_route(query: str, keyword_map: dict[str, str], threshold: float = 1.0) -> RouteDecision:
    q = query.casefold()
    matches = [(k, app) for k, app in keyword_map.items() if k.casefold() in q]
    if not matches:
        return RouteDecision('evidence', None, 0.0, 'no deterministic keyword match')
    keyword, app = sorted(matches, key=lambda x: (-len(x[0]), x[0]))[0]
    return RouteDecision('faq', app, float(threshold), f'matched keyword: {keyword}')
