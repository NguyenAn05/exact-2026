from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.hardware_monitor import snapshot as hw_snapshot
from app.llm_client import VLLMClient
from app.logging_utils import log_entry
from app.pipelines.graph import agent_graph
from app.pipelines.mock import solve_type1_mock, solve_type2_mock
from app.schemas import PredictRequest, PredictResponseItem, fallback_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.type1_llm = VLLMClient(
        settings,
        base_url=settings.resolved_type1_base_url(),
        model_name=settings.resolved_type1_model_name(),
    )
    app.state.type2_llm = VLLMClient(
        settings,
        base_url=settings.resolved_type2_base_url(),
        model_name=settings.resolved_type2_model_name(),
    )
    app.state.agent_graph = agent_graph
    yield


app = FastAPI(
    title="EXACT 2026 Submission Server",
    version="0.3.0-langgraph",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "mock_mode": settings.mock_mode,
        "type1_vllm_base_url": settings.resolved_type1_base_url(),
        "type1_model_name": settings.resolved_type1_model_name() or "auto-from-/v1/models",
        "type2_vllm_base_url": settings.resolved_type2_base_url(),
        "type2_model_name": settings.resolved_type2_model_name() or "auto-from-/v1/models",
        "type2_fallback_to_llm": settings.type2_fallback_to_llm,
    }


def _ensure_runtime_clients() -> None:
    """Initialize vLLM clients if lifespan has not run yet.

    This keeps tests and some ASGI runners safe; normal uvicorn startup uses lifespan.
    """
    settings = get_settings()
    if not hasattr(app.state, "type1_llm"):
        app.state.type1_llm = VLLMClient(
            settings,
            base_url=settings.resolved_type1_base_url(),
            model_name=settings.resolved_type1_model_name(),
        )
    if not hasattr(app.state, "type2_llm"):
        app.state.type2_llm = VLLMClient(
            settings,
            base_url=settings.resolved_type2_base_url(),
            model_name=settings.resolved_type2_model_name(),
        )
    if not hasattr(app.state, "agent_graph"):
        app.state.agent_graph = agent_graph


@app.post("/predict", response_model=list[PredictResponseItem])
async def predict(payload: PredictRequest) -> list[PredictResponseItem]:
    t0 = time.time()
    hw_before = hw_snapshot()
    settings = get_settings()
    _ensure_runtime_clients()

    if settings.mock_mode:
        if payload.type == "type1":
            item = await solve_type1_mock(payload)
        elif payload.type == "type2":
            item = await solve_type2_mock(payload)
        else:
            item = fallback_response(payload, f"Unsupported type: {payload.type}")
    else:
        try:
            # Reconstruct initial state payload for LangGraph
            state_input = {
                "query_id": payload.query_id,
                "qtype": payload.type,
                "query": payload.query,
                "premises": payload.premises,
                "options": payload.options,
                "retry_count": 0,
                "attempts_history": [],
                "start_time": time.time(),
            }
            # Execute graph dynamically passing the LLM clients config
            config = {
                "configurable": {
                    "type1_llm": app.state.type1_llm,
                    "type2_llm": app.state.type2_llm,
                }
            }
            state_output = await app.state.agent_graph.ainvoke(state_input, config)

            # Map compiled state final outputs to predictable PredictResponseItem
            item = PredictResponseItem(
                query_id=payload.query_id,
                answer=state_output.get("final_answer", ""),
                unit=state_output.get("final_unit", ""),
                explanation=state_output.get("final_explanation", ""),
                premises_used=state_output.get("final_premises_used", []),
                reasoning=state_output.get("final_reasoning", {"type": "cot", "steps": []}),
            )
        except Exception as exc:
            item = fallback_response(payload, f"LangGraph execution failed safely: {exc}")

    latency_s = round(time.time() - t0, 3)
    hw_after = hw_snapshot()
    log_entry({
        "event": "predict",
        "query_id": payload.query_id,
        "type": payload.type,
        "request": {
            "query": payload.query,
            "premises": payload.premises,
            "options": payload.options,
        },
        "response": {
            "answer": item.answer,
            "unit": item.unit,
            "explanation": item.explanation,
            "premises_used": item.premises_used,
        },
        "latency_s": latency_s,
        "hw_before": hw_before,
        "hw_after": hw_after,
        "model": {
            "reasoning_type": getattr(item.reasoning, "type", None) if hasattr(item, "reasoning") else (item.reasoning.get("type") if isinstance(item.reasoning, dict) else None),
        },
    })

    # Competition requires a list even for one input query.
    return [item]


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    # Do not leak stack traces to the evaluator. Validation errors still use FastAPI defaults.
    return JSONResponse(status_code=500, content={"detail": f"Unhandled server error: {exc}"})
