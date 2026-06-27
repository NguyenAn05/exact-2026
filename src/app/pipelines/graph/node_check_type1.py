from __future__ import annotations

import time
from typing import Any, Dict

from app.pipelines.graph.state import AgentState
from app.utils.sandbox import last_stdout_value


def check_type1_node(state: AgentState) -> Dict[str, Any]:
    """Node that evaluates Type 1 sandbox execution and prepares retry history if needed."""
    z3_ans = last_stdout_value(state.get("sandbox_output", "")).strip()
    llm_ans = state.get("llm_answer", "").strip()
    sandbox_ok = state.get("sandbox_ok", False)

    # We match case-insensitively
    mismatch = sandbox_ok and (llm_ans.lower() != z3_ans.lower())
    attempts = state.get("attempts_history") or []
    retry_count = state.get("retry_count", 0)

    if (not sandbox_ok) or mismatch:
        if not sandbox_ok:
            feedback = f"Z3 code failed: {state.get('sandbox_error', 'empty stdout or compilation error')}"
        else:
            feedback = f"Mismatch: LLM got '{llm_ans}' but Z3 solver got '{z3_ans}'."

        new_attempts = list(attempts) + [
            {"code": state.get("generated_code", ""), "feedback": feedback}
        ]
        return {
            "retry_count": retry_count + 1,
            "attempts_history": new_attempts,
            "needs_retry": True,
        }

    return {
        "needs_retry": False,
    }


def route_after_check_type1(state: AgentState) -> str:
    """Routing function after Check Type 1 node."""
    elapsed = time.time() - state.get("start_time", time.time())
    needs_retry = state.get("needs_retry", False)
    retry_count = state.get("retry_count", 0)

    # Allow at most 2 retries (3 runs total: initial + 2 retries), and under 35s budget
    if needs_retry and retry_count < 3 and elapsed < 35.0:
        return "llm_type1"

    return "formatter"
