from __future__ import annotations

from app.pipelines.graph.state import AgentState
from app.pipelines.graph.node_type2 import should_use_type2_fast_path


def route_by_type(state: AgentState) -> str:
    """Deterministic routing function based on query type."""
    qtype = state.get("qtype", "type1")
    if qtype == "type2":
        if should_use_type2_fast_path(state.get("query", "")):
            return "fast_path"
        return "type2"
    return "type1"
