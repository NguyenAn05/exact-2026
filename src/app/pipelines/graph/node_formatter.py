from __future__ import annotations

from typing import Any, Dict

from app.pipelines.graph.state import AgentState
from app.schemas import PredictRequest
from app.utils.normalize import coerce_response


def formatter_node(state: AgentState) -> Dict[str, Any]:
    """Node that normalizes and formats the final answer/unit using coerce_response."""
    req = PredictRequest(
        query_id=state["query_id"],
        type=state["qtype"],
        query=state["query"],
        premises=state["premises"],
        options=state["options"],
    )

    raw = {
        "answer": state.get("llm_answer", ""),
        "unit": state.get("unit", ""),
        "explanation": state.get("explanation", ""),
        "premises_used": state.get("premises_used", []),
        "reasoning": state.get("reasoning", {}),
    }

    # Coerce using our normalization helpers
    coerced = coerce_response(req, raw)

    return {
        "final_answer": coerced.answer,
        "final_unit": coerced.unit,
        "final_explanation": coerced.explanation,
        "final_premises_used": coerced.premises_used,
        "final_reasoning": coerced.reasoning.model_dump() if hasattr(coerced.reasoning, "model_dump") else coerced.reasoning,
    }
