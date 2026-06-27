from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    # Input data
    query_id: str
    qtype: str  # 'type1' or 'type2'
    query: str
    premises: List[str]
    options: List[str]

    # Intermediate model outputs
    llm_answer: str
    unit: str
    explanation: str
    premises_used: List[int]
    reasoning: Dict[str, Any]
    generated_code: str  # z3_code or python_code

    # Sandbox execution status
    sandbox_ok: bool
    sandbox_output: str
    sandbox_error: str

    # Agent loop counters and budget
    retry_count: int
    attempts_history: List[Dict[str, Any]]
    start_time: float

    # Final outputs
    final_answer: str
    final_unit: str
    final_explanation: str
    final_premises_used: List[int]
    final_reasoning: Dict[str, Any]
    error_occurred: str

    # Control flow flags
    needs_retry: bool
    needs_fallback: bool
