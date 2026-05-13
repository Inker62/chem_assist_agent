import os
import operator
from typing import TypedDict, Annotated, Sequence, List, Literal
from dotenv import load_dotenv
from pandas.core.methods import describe
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from tools.chemspider_search import ChemSpiderTool
from tools.chem_memory import ChemicalMemoryTool
from tools.literature_search import PubmedSearchTool, CrossrefSearchTool

load_dotenv()
BASE_URL = "https://api.deepseek.com"

class MultiAgentState(TypedDict):
    """全局状态，用于在不同Agent之间共享信息"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str | None

class SupervisorDecision(BaseModel):
    """主管决策结构"""
    next: Literal["chem_agent", "literature_agent", "both", "finish"] = Field(
        description="下一步将把任务派往的专家Agent。'both'代表需要先调用'chem_agent',再调用'literature_agent'。'finish'代表已解决问题"
    )


def create_supervisor_llm():
    """创建主管专家Agent，用于总结和决策"""
    return ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=BASE_URL,
        extra_body={"thinking": {"type": "disabled"}}
    )

# 定义专家成员函数

chem_tools = [ChemSpiderTool(), ChemicalMemoryTool()]
literature_tools = [PubmedSearchTool(), CrossrefSearchTool()]

chem_tool_node = ToolNode(chem_tools)
literature_tool_node = ToolNode(literature_tools)

chem_model = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=BASE_URL,
        extra_body={"thinking": {"type": "disabled"}}
    ).bind_tools(chem_tools)

literature_model = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=BASE_URL,
        extra_body={"thinking": {"type": "disabled"}}
    ).bind_tools(literature_tools)

def chem_agent_node(state: MultiAgentState)-> dict:
    """化学专家Agent节点：处理化学物质查询"""
    response = chem_model.invoke(state["messages"])
    if response.tool_calls:
        return {"messages": [response], "next": "chem_tools"}
    return {"messages": [response], "next": "supervisor"}

def literature_agent_node(state: MultiAgentState) -> dict:
    """文献专家Agent节点：处理学术文献查询"""
    response = literature_model.invoke(state["messages"])
    if response.tool_calls:
        return {"messages": [response], "next": "literature_tools"}
    return {"messages": [response], "next": "supervisor"}

supervisor_llm = create_supervisor_llm()
structured_supervisor = supervisor_llm.with_structured_output(SupervisorDecision)

def supervisor_node(state: MultiAgentState) -> dict:
    """主管专家Agent节点：分析用户意图，并分派任务给合适的专家Agent"""
    system_prompt = (
        "你是ChemAssist的主管智能体。你的任务是根据用户的请求，将任务分派给合适的专家Agent。\n"
        "你可以调用以下两个专家Agent。\n"
        "1.'chem_agent':根据用户输入查询化学物质的结构与性质。\n"
        "2.'literature_agent':根据用户输入的关键词查询相关研究文献。\n"
        "如果你判断需要化学专家，请回复 `chem_agent`。如果需要文献专家，请回复 `literature_agent`。如果需要两者，请回复 `both`。如果问题已经解决，返回 'finish'。"
    )

    messages = [{"role": "system", "content": system_prompt}] + list(state["messages"])

    try:
        decision: SupervisorDecision = structured_supervisor.invoke(messages)
    except Exception:
        return {"next": "finish"}

    if decision.next == "both":
        return {"next": "chem_agent"}
    elif decision.next == "finish":
        return {"next": END}
    else:
        return {"next": decision.next}

# 构建Graph
def build_workflow():
    """构建多智能体工作流"""
    workflow = StateGraph(MultiAgentState)

    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("chem_agent", chem_agent_node)
    workflow.add_node("chem_tools", chem_tool_node)
    workflow.add_node("literature_agent", literature_agent_node)
    workflow.add_node("literature_tools", literature_tool_node)

    # 添加边
    workflow.add_edge(START, "supervisor")

    # 从主管到专家的调度
    workflow.add_conditional_edge(
        "supervisor",
        lambda x: x["next"],
        {
            "chem_agent": "chem_agent",
            "literature_agent": "literature_agent",
            END: END
        }
    )

    # 从专家到工具&从工具返回专家
    workflow.add_conditional_edge(
        "chem_agent",
        lambda x: x["next"],
        {
            "chem_tools": "chem_tools", "supervisor": "supervisor",
        }
    )
    workflow.add_edge("chem_tools", "chem_agent")

    workflow.add_conditional_edge(
        "literature_agent",
        lambda x: x["next"],
        {
            "literature_tools": "literature_tools", "supervisor": "supervisor",
        }
    )
    workflow.add_edge("literature_tools", "literature_agent")

    return workflow

def initialize_multiagent():
    """初始化 LangGraph 多智能体应用"""
    workflow = build_workflow()
    checkpointer = InMemorySaver()  # 提供短期记忆
    return workflow.compile(checkpointer=checkpointer)


