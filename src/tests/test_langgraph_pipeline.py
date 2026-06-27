import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock

from app.schemas import PredictRequest
from app.pipelines.graph import agent_graph


@pytest.mark.asyncio
async def test_langgraph_logic_flow_success():
    # Input data
    state_input = {
        "query_id": "T1_LG_001",
        "qtype": "type1",
        "query": "Is Asha happy?",
        "premises": [
            "If Asha gets a gift, she is happy.",
            "Asha got a gift."
        ],
        "options": ["Yes", "No", "Uncertain"],
        "retry_count": 0,
        "attempts_history": [],
        "start_time": time.time(),
    }

    # Mock Type 1 LLM client
    type1_llm_mock = MagicMock()
    first_resp = {
        "answer": "Yes",
        "unit": "",
        "explanation": "Asha got a gift, so she is happy.",
        "premises_used": [0, 1],
        "reasoning": {"type": "fol", "steps": []},
        "z3_code": "print('Yes')"
    }
    type1_llm_mock.chat_json = AsyncMock(return_value=json.dumps(first_resp))

    config = {
        "configurable": {
            "type1_llm": type1_llm_mock,
            "type2_llm": MagicMock(),
        }
    }

    # Execute graph
    output = await agent_graph.ainvoke(state_input, config)

    assert output["final_answer"] == "Yes"
    assert output["final_unit"] == ""
    assert output["final_premises_used"] == [0, 1]
    assert type1_llm_mock.chat_json.call_count == 1


@pytest.mark.asyncio
async def test_langgraph_logic_flow_retry():
    state_input = {
        "query_id": "T1_LG_002",
        "qtype": "type1",
        "query": "Is Asha happy?",
        "premises": ["Asha got a gift."],
        "options": ["Yes", "No", "Uncertain"],
        "retry_count": 0,
        "attempts_history": [],
        "start_time": time.time(),
    }

    type1_llm_mock = MagicMock()
    # First response mismatch: LLM says Yes, but Z3 solver code prints Uncertain
    first_resp = {
        "answer": "Yes",
        "unit": "",
        "explanation": "Dummy",
        "premises_used": [0],
        "reasoning": {"type": "fol", "steps": []},
        "z3_code": "print('Uncertain')"
    }
    # Second response matching: LLM corrects and matches Z3 outputting Yes
    second_resp = {
        "answer": "Yes",
        "unit": "",
        "explanation": "Corrected",
        "premises_used": [0],
        "reasoning": {"type": "fol", "steps": []},
        "z3_code": "print('Yes')"
    }

    type1_llm_mock.chat_json = AsyncMock()
    type1_llm_mock.chat_json.side_effect = [
        json.dumps(first_resp),
        json.dumps(second_resp)
    ]

    config = {
        "configurable": {
            "type1_llm": type1_llm_mock,
            "type2_llm": MagicMock(),
        }
    }

    output = await agent_graph.ainvoke(state_input, config)

    # Verifies retry node was called
    assert type1_llm_mock.chat_json.call_count == 2
    assert output["final_answer"] == "Yes"


@pytest.mark.asyncio
async def test_langgraph_physics_flow_success():
    state_input = {
        "query_id": "T2_LG_001",
        "qtype": "type2",
        "query": "Calculate capacitance of a 10V capacitor storing 50uC.",
        "premises": [],
        "options": [],
        "retry_count": 0,
        "attempts_history": [],
        "start_time": time.time(),
    }

    type2_llm_mock = MagicMock()
    # Mock LLM to return valid python code that prints the numeric value
    type2_resp = {
        "python_code": "print(5)",
        "unit": "uF",
        "explanation": "C = Q / V"
    }
    type2_llm_mock.chat_json = AsyncMock(return_value=json.dumps(type2_resp))

    config = {
        "configurable": {
            "type1_llm": MagicMock(),
            "type2_llm": type2_llm_mock,
        }
    }

    output = await agent_graph.ainvoke(state_input, config)

    # 5 is coerced and returned with unit uF
    assert output["final_answer"] == "5"
    assert output["final_unit"] == "uF"
    assert type2_llm_mock.chat_json.call_count == 1


@pytest.mark.asyncio
async def test_langgraph_physics_flow_fallback():
    state_input = {
        "query_id": "T2_LG_002",
        "qtype": "type2",
        "query": "Calculate capacitance.",
        "premises": [],
        "options": [],
        "retry_count": 0,
        "attempts_history": [],
        "start_time": time.time(),
    }

    type2_llm_mock = MagicMock()
    # Code generation fails / returns invalid syntax
    type2_code_resp = {
        "python_code": "invalid python code",
        "unit": "uF",
        "explanation": "C = Q / V"
    }
    # Direct fallback returns direct answer
    type2_direct_resp = {
        "answer": "5",
        "unit": "uF",
        "explanation": "Fallback direct answer",
        "premises_used": [],
        "reasoning": {"type": "cot", "steps": ["Direct deduction"]}
    }

    type2_llm_mock.chat_json = AsyncMock()
    type2_llm_mock.chat_json.side_effect = [
        json.dumps(type2_code_resp),
        json.dumps(type2_direct_resp)
    ]

    config = {
        "configurable": {
            "type1_llm": MagicMock(),
            "type2_llm": type2_llm_mock,
        }
    }

    output = await agent_graph.ainvoke(state_input, config)

    # Fallback direct LLM node is called after code sandbox fails
    assert type2_llm_mock.chat_json.call_count == 2
    assert output["final_answer"] == "5"
    assert output["final_unit"] == "uF"
