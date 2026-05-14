import os
import operator
from typing import TypedDict, Annotated, Sequence, List, Literal
from dotenv import load_dotenv
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
        #extra_body={"thinking": {"type": "disabled"}}
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


def chem_agent_node(state: MultiAgentState) -> dict:
    # 强制标准化：只保留 HumanMessage, AIMessage, ToolMessage
    clean = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            clean.append(HumanMessage(content=m.content))
        elif isinstance(m, AIMessage):
            # 保留 AIMessage，但可在此处做清洗
            clean.append(AIMessage(content=m.content, id=m.id))
        elif isinstance(m, ToolMessage):
            clean.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id, name=m.name))

    # 如果没有有效消息，返回错误提示
    if not clean:
        return {"messages": [AIMessage(content="错误：未收到有效查询。")], "next": END}

    # 注入化学专家身份
    system_msg = HumanMessage(content="你是化学信息学专家，使用工具查询物质结构数据。")
    messages = [system_msg] + clean

    response = chem_model.invoke(messages)
    if response.tool_calls:
        return {"messages": [response], "next": "chem_tools"}
    return {"messages": [response], "next": "supervisor"}


def literature_agent_node(state: MultiAgentState) -> dict:
    # 强制标准化
    clean = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            clean.append(HumanMessage(content=m.content))
        elif isinstance(m, AIMessage):
            clean.append(AIMessage(content=m.content, id=m.id))
        elif isinstance(m, ToolMessage):
            clean.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id, name=m.name))

    if not clean:
        return {"messages": [AIMessage(content="错误：未收到有效查询。")], "next": END}

    # 注入文献专家身份
    system_msg = HumanMessage(content="你是文献检索专家，使用工具获取学术论文信息。")
    messages = [system_msg] + clean

    response = literature_model.invoke(messages)
    if response.tool_calls:
        return {"messages": [response], "next": "literature_tools"}
    return {"messages": [response], "next": "supervisor"}

supervisor_llm = create_supervisor_llm()
structured_supervisor = supervisor_llm.with_structured_output(SupervisorDecision)


def supervisor_node(state: MultiAgentState) -> dict:
    # 强制标准化
    clean = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            clean.append(HumanMessage(content=m.content))
        elif isinstance(m, AIMessage):
            clean.append(AIMessage(content=m.content, id=m.id))
        elif isinstance(m, ToolMessage):
            clean.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id, name=m.name))

    system_prompt = (
        "你是 ChemAssist 的主管。你的唯一职责是根据用户请求，决定调用哪个专家。\n"
        "可选决策：\n"
        "- chem_agent：查询化学物质的结构、性质\n"
        "- literature_agent：检索学术文献\n"
        "- both：需要同时查询化学和文献（会先化学后文献）\n"
        "- finish：任务已由专家完成，需要你生成一个最终回复\n\n"
        "请只返回决策键，不要添加任何额外文字。"
    )
    messages = [HumanMessage(content=system_prompt)] + clean

    try:
        decision = structured_supervisor.invoke(messages)
    except Exception:
        return {"next": END}

    if decision.next == "both":
        return {"next": "chem_agent"}
    elif decision.next == "finish":
        # 任务完成，主管自己生成最终回复
        try:
            final_response = supervisor_llm.invoke(clean)
            return {"messages": [final_response], "next": END}
        except Exception:
            # LLM调用失败时的兜底
            return {"messages": [AIMessage(content="抱歉，处理过程中遇到错误，请稍后重试。")], "next": END}
    elif decision.next in ("chem_agent", "literature_agent"):
        return {"next": decision.next}
    else:
        return {"next": END}


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
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "chem_agent": "chem_agent",
            "literature_agent": "literature_agent",
            "finish": END,
            END: END
        }
    )

    # 从专家到工具&从工具返回专家
    workflow.add_conditional_edges(
        "chem_agent",
        lambda x: x["next"],
        {
            "chem_tools": "chem_tools", "supervisor": "supervisor",
        }
    )
    workflow.add_edge("chem_tools", "chem_agent")

    workflow.add_conditional_edges(
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


