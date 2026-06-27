from __future__ import annotations

from typing import Any, Literal, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    query_id: str
    type: Literal["type1", "type2"]
    query: str
    premises: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)

    @field_validator("query_id", "query")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class Reasoning(BaseModel):
    type: str
    steps: list[str]


class PredictResponseItem(BaseModel):
    query_id: str
    answer: str
    unit: str
    explanation: str
    premises_used: list[int]
    reasoning: Optional[Dict[str, Any]] = None


def fallback_response(req: PredictRequest, message: str) -> PredictResponseItem:
    """Return a valid competition-shaped response even if the model or parser fails."""
    if req.type == "type1":
        if req.options:
            answer = "Uncertain" if "Uncertain" in req.options else req.options[0]
        else:
            answer = "Uncertain"
        return PredictResponseItem(
            query_id=req.query_id,
            answer=answer,
            unit="",
            explanation=message,
            premises_used=[],
            reasoning={"type": "fallback", "steps": [message]},
        )

    return PredictResponseItem(
        query_id=req.query_id,
        answer="0",
        unit="",
        explanation=message,
        premises_used=[],
        reasoning={"type": "fallback", "steps": [message]},
    )
