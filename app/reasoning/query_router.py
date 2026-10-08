"""Deterministic query routing that selects retrieval and reasoning strategy."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRoute:
    strategy: str
    reason: str
    top_k: int = 5
    requires_multiple_hops: bool = False


_MULTI_HOP = re.compile(r"\b(compare|both|between|relationship|how does .+ affect|across .+ and|first .+ then)\b", re.I)
_KEYWORD_HEAVY = re.compile(r"\b(section|clause|article|§|\d+(?:\.\d+)+|exact wording|defined term)\b", re.I)


def route_query(question: str) -> QueryRoute:
    """Route a query without an extra model call; always grounded in document search."""
    text = (question or "").strip()
    if _MULTI_HOP.search(text):
        return QueryRoute("multi_hop", "The question asks for a comparison or linked facts.", 8, True)
    if _KEYWORD_HEAVY.search(text):
        return QueryRoute("lexical_hybrid", "The question names a clause, section, or exact term.", 8)
    return QueryRoute("semantic_hybrid", "Use the standard dense and lexical hybrid search.", 5)
