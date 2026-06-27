from __future__ import annotations

from typing import Any, Dict

from app.pipelines.graph.state import AgentState
from app.utils.sandbox import last_stdout_value


def check_type2_node(state: AgentState) -> Dict[str, Any]:
    """Node that evaluates Type 2 sandbox execution and decides on fallback."""
    sandbox_ok = state.get("sandbox_ok", False)
    if sandbox_ok:
        ans = last_stdout_value(state.get("sandbox_output", "")).strip()
        if ans:
            return {
                "llm_answer": ans,
                "needs_fallback": False,
            }

    return {
        "needs_fallback": True,
    }


def route_after_check_type2(state: AgentState) -> str:
    """Routing function after Check Type 2 node."""
    if state.get("needs_fallback", False):
        return "llm_type2_direct"
    return "formatter"
