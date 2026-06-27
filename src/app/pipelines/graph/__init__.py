from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.pipelines.graph.node_check_type1 import (
    check_type1_node,
    route_after_check_type1,
)
from app.pipelines.graph.node_check_type2 import (
    check_type2_node,
    route_after_check_type2,
)
from app.pipelines.graph.node_formatter import formatter_node
from app.pipelines.graph.node_router import route_by_type
from app.pipelines.graph.node_sandbox import sandbox_node
from app.pipelines.graph.node_type1 import llm_type1_node
from app.pipelines.graph.node_type2 import (
    llm_type2_direct_node,
    llm_type2_node,
    deterministic_physics_solver,
)
from app.pipelines.graph.state import AgentState
from typing import Any, Dict

# Initialize the StateGraph
workflow = StateGraph(AgentState)


async def fast_path_node(state: AgentState) -> Dict[str, Any]:
    fast = deterministic_physics_solver(state.get("query", ""))
    if fast:
        return {
            "llm_answer": fast["answer"],
            "unit": fast["unit"],
            "explanation": fast["explanation"],
            "premises_used": fast["premises_used"],
            "reasoning": fast["reasoning"],
        }
    return {}


# 1. Add all graph nodes
workflow.add_node("llm_type1", llm_type1_node)
workflow.add_node("llm_type2", llm_type2_node)
workflow.add_node("llm_type2_direct", llm_type2_direct_node)
workflow.add_node("fast_path", fast_path_node)
workflow.add_node("sandbox", sandbox_node)
workflow.add_node("check_type1", check_type1_node)
workflow.add_node("check_type2", check_type2_node)
workflow.add_node("formatter", formatter_node)

# 2. Add Start Routing conditional edge
workflow.add_conditional_edges(
    START,
    route_by_type,
    {
        "type1": "llm_type1",
        "type2": "llm_type2",
        "fast_path": "fast_path",
    },
)


# 3. Connect LLM Nodes to Sandbox
workflow.add_edge("llm_type1", "sandbox")
workflow.add_edge("llm_type2", "sandbox")


# 4. Connect Sandbox output based on query type
def route_after_sandbox(state: AgentState) -> str:
    if state.get("qtype") == "type2":
        return "check_type2"
    return "check_type1"


workflow.add_conditional_edges(
    "sandbox",
    route_after_sandbox,
    {
        "check_type1": "check_type1",
        "check_type2": "check_type2",
    },
)

# 5. Connect Check Type 1 with retry/formatter routing
workflow.add_conditional_edges(
    "check_type1",
    route_after_check_type1,
    {
        "llm_type1": "llm_type1",
        "formatter": "formatter",
    },
)

# 6. Connect Check Type 2 with fallback/formatter routing
workflow.add_conditional_edges(
    "check_type2",
    route_after_check_type2,
    {
        "llm_type2_direct": "llm_type2_direct",
        "formatter": "formatter",
    },
)

# 7. Connect direct LLM node to formatter
workflow.add_edge("llm_type2_direct", "formatter")
workflow.add_edge("fast_path", "formatter")

# 8. Connect Formatter to END
workflow.add_edge("formatter", END)

# Compile the final executable Graph
agent_graph = workflow.compile()
