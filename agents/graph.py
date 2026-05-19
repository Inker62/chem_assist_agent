"""多智能体图构建——通过 import agents.multiagent as _mam 访问 ToolNode 实例"""
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from config import CHECKPOINT_DB_PATH
from agents.state import MultiAgentState
from agents.nodes import _clean_messages, supervisor_node, chem_agent_node, literature_agent_node, summary_node
import agents.multiagent as _mam


def _route_supervisor(state: MultiAgentState):
    """主管条件边路由——both 时顺序执行 chem 再 lit，避免 Send 并行导致消息交错"""
    next_ = state["next"]
    if next_ == "both":
        return "chem_agent"
    return next_


def build_workflow():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("chem_agent", chem_agent_node)
    workflow.add_node("chem_tools", _mam.chem_tool_node)
    workflow.add_node("literature_agent", literature_agent_node)
    workflow.add_node("literature_tools", _mam.literature_tool_node)
    workflow.add_node("summary", summary_node)
    workflow.add_edge("summary", END)
    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {"chem_agent": "chem_agent", "literature_agent": "literature_agent", END: END}
    )

    # 化学专家：工具循环，或 → summary，或 → literature_agent（both 顺序链）
    workflow.add_conditional_edges(
        "chem_agent",
        lambda state: state["next"],
        {
            "chem_tools": "chem_tools",
            "summary": "summary",
            "literature_agent": "literature_agent",
            END: END,
        }
    )
    workflow.add_edge("chem_tools", "chem_agent")

    # 文献专家：工具循环，或 → summary
    workflow.add_conditional_edges(
        "literature_agent",
        lambda state: state["next"],
        {"literature_tools": "literature_tools", "summary": "summary", END: END}
    )
    workflow.add_edge("literature_tools", "literature_agent")

    return workflow


def initialize_multiagent():
    workflow = build_workflow()
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)
