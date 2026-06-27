from __future__ import annotations

from app.schemas import PredictRequest, PredictResponseItem
from app.utils.normalize import coerce_response
from app.pipelines.graph.node_type2 import deterministic_physics_solver


async def solve_type1_mock(req: PredictRequest) -> PredictResponseItem:
    raw = {
        "answer": "Uncertain" if "Uncertain" in req.options else (req.options[0] if req.options else "Uncertain"),
        "unit": "",
        "explanation": "MOCK_MODE response. Connect vLLM to generate real logic answers.",
        "premises_used": list(range(len(req.premises))),
        "reasoning": {"type": "mock", "steps": ["Mock response generated for API-format testing."]},
    }
    return coerce_response(req, raw)


async def solve_type2_mock(req: PredictRequest) -> PredictResponseItem:
    fast = deterministic_physics_solver(req.query)
    if fast is not None:
        return coerce_response(req, fast)

    raw = {
        "answer": "0",
        "unit": "",
        "explanation": "MOCK_MODE response. Connect vLLM to generate real physics answers.",
        "premises_used": [],
        "reasoning": {"type": "mock", "steps": ["Mock response generated for API-format testing."]},
    }
    return coerce_response(req, raw)

