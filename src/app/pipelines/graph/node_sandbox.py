from __future__ import annotations

from typing import Any, Dict

from app.pipelines.graph.state import AgentState
from app.utils.sandbox import run_python_code


TYPE1_SANDBOX_TIMEOUT_SECONDS = 3.0
TYPE2_SANDBOX_TIMEOUT_SECONDS = 5.0


def sandbox_node(state: AgentState) -> Dict[str, Any]:
    """Sandbox execution node that runs the generated python code (Z3 or SymPy)."""
    code = state.get("generated_code", "").strip()
    if not code:
        return {
            "sandbox_ok": False,
            "sandbox_output": "",
            "sandbox_error": "No generated code found in state",
        }

    sandbox_type = "type2" if state.get("qtype") == "type2" else "type1"
    timeout_seconds = TYPE2_SANDBOX_TIMEOUT_SECONDS if sandbox_type == "type2" else TYPE1_SANDBOX_TIMEOUT_SECONDS
    result = run_python_code(code, sandbox_type=sandbox_type, timeout_seconds=timeout_seconds)

    return {
        "sandbox_ok": result.ok,
        "sandbox_output": result.stdout,
        "sandbox_error": result.stderr if not result.ok else "",
    }
