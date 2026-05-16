import os
import operator
import json
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from tools.chemspider_search import ChemSpiderTool
from tools.chem_memory import ChemicalMemoryTool
from tools.literature_search import CrossrefSearchTool

load_dotenv()
BASE_URL = "https://api.deepseek.com"

# -------------------- 全局状态 --------------------
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str | None
    round_count: int  # 循环计数器，防止无限循环

# -------------------- 消息清洗函数 --------------------
def _clean_messages(messages: list) -> list:
    """只保留 HumanMessage 和不带 tool_calls 的 AIMessage，供最终总结使用"""
    clean = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            clean.append(msg)
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            clean.append(msg)
    return clean

# -------------------- LLM 实例 --------------------
# 主管决策专用 LLM（开启 JSON 输出）
decision_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=BASE_URL,
    extra_body={
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"}
    }
)

# 主管最终总结专用 LLM（无 JSON 限制）
summary_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=BASE_URL,
    extra_body={"thinking": {"type": "disabled"}}
)

# -------------------- 专家工具与模型 --------------------
chem_tools = [ChemSpiderTool(), ChemicalMemoryTool()]
literature_tools = [CrossrefSearchTool()]

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

# -------------------- 节点定义 --------------------
def supervisor_node(state: MultiAgentState) -> dict:
    print("🔥 supervisor_node 被调用")
    clean_history = _clean_messages(state["messages"])
    round_count = state.get("round_count", 0)

    # 防止无限循环
    if round_count >= 3:
        print("⛔ 达到最大循环轮次，强制结束")
        return {"messages": [AIMessage(content="抱歉，处理超时，请稍后重试。")], "next": END}

    # 自动检测：如果最后一条消息是 AIMessage 且无 tool_calls，说明专家已完成
    last_msg = state["messages"][-1] if state["messages"] else None
    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
        print("✅ 检测到专家已完工，直接 finish")
        # 追加格式化指令，使输出整齐
        format_instruction = HumanMessage(content="如果上面的对话中包含文献检索的结果，请将那些文献整理成一个清晰的表格，表格列包括：序号、标题、第一作者、期刊、出版年、DOI。只输出表格，不要添加任何额外的总结、综述或评论。如果没有文献，就简洁地总结化学信息，使用表格列出关键属性。")
        messages_for_summary = clean_history + [format_instruction]
        try:
            final = summary_llm.invoke(messages_for_summary)
            print(f"💬 主管回复: {final.content[:100]}")
            return {"messages": [final], "next": END}
        except Exception as e:
            print(f"❌ 主管生成回复失败: {e}")
            return {"messages": [AIMessage(content="抱歉，处理时遇到错误。")], "next": END}

    # 专家未完工，继续决策
    system_prompt = (
        "你是 ChemAssist 的主管。你的唯一职责是根据用户请求，决定调用哪个专家。\n"
        "你必须严格按照以下JSON格式输出你的决策，不要包含任何其他文字：\n\n"
        '{"next": "<决策词>"}\n\n'
        "决策词只能是以下四个之一：\n"
        '- "chem_agent"：查询化学物质的结构与性质\n'
        '- "literature_agent"：检索学术文献\n'
        '- "both"：需要同时查询化学和文献（会先化学后文献）\n'
        '- "finish"：任务已由专家完成，由你直接友好回复\n\n'
        "例如，如果用户问'苯甲酸的结构'，你应该输出：\n"
        '{"next": "chem_agent"}\n\n'
        "请输出JSON。"
    )

    messages = [HumanMessage(content=system_prompt)] + clean_history

    try:
        response = decision_llm.invoke(messages)
        print(f"🧠 主管原始回复: {response.content}")
        decision_json = json.loads(response.content)
        decision = decision_json.get("next", "finish")
        print(f"🧠 解析后决策: {decision}")
    except Exception as e:
        print(f"❌ 主管决策失败: {e}")
        return {"messages": [AIMessage(content=f"决策错误: {e}")], "next": END}

    # 路由，递增轮次
    if decision == "both":
        return {"next": "chem_agent", "round_count": round_count + 1}
    elif decision == "finish":
        # 直接总结（不加额外指令，因为这是主管主动认为任务完成）
        try:
            final = summary_llm.invoke(clean_history)
            print(f"💬 主管回复: {final.content[:100]}")
            return {"messages": [final], "next": END}
        except Exception as e:
            print(f"❌ 主管生成回复失败: {e}")
            return {"messages": [AIMessage(content="抱歉，处理时遇到错误。")], "next": END}
    elif decision in ("chem_agent", "literature_agent"):
        return {"next": decision, "round_count": round_count + 1}
    else:
        return {"messages": [AIMessage(content="抱歉，无法理解你的请求。")], "next": END}


def chem_agent_node(state: MultiAgentState) -> dict:
    print("⚗️ chem_agent_node 被调用")
    # 直接使用完整消息，让模型看到工具痕迹
    messages = [HumanMessage(content="你是化学信息学专家，使用工具查询物质结构数据。")] + list(state["messages"])
    response = chem_model.invoke(messages)
    if response.tool_calls:
        print(f"🔧 化学专家调用工具: {[tc['name'] for tc in response.tool_calls]}")
        return {"messages": [response], "next": "chem_tools"}
    print("💬 化学专家任务完成，返回主管")
    return {"messages": [response], "next": "supervisor"}


def literature_agent_node(state: MultiAgentState) -> dict:
    print("📚 literature_agent_node 被调用")
    messages = [HumanMessage(content="你是文献检索专家，使用工具获取学术论文信息。")] + list(state["messages"])
    response = literature_model.invoke(messages)
    if response.tool_calls:
        print(f"🔧 文献专家调用工具: {[tc['name'] for tc in response.tool_calls]}")
        return {"messages": [response], "next": "literature_tools"}
    print("💬 文献专家任务完成，返回主管")
    return {"messages": [response], "next": "supervisor"}


# -------------------- 构建图 --------------------
def build_workflow():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("chem_agent", chem_agent_node)
    workflow.add_node("chem_tools", chem_tool_node)
    workflow.add_node("literature_agent", literature_agent_node)
    workflow.add_node("literature_tools", literature_tool_node)

    workflow.add_edge(START, "supervisor")

    # 主管 → 专家
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "chem_agent": "chem_agent",
            "literature_agent": "literature_agent",
            END: END
        }
    )

    # 化学专家 ↔ 工具
    workflow.add_conditional_edges(
        "chem_agent",
        lambda state: state["next"],
        {
            "chem_tools": "chem_tools",
            "supervisor": "supervisor",
            END: END
        }
    )
    workflow.add_edge("chem_tools", "chem_agent")

    # 文献专家 ↔ 工具
    workflow.add_conditional_edges(
        "literature_agent",
        lambda state: state["next"],
        {
            "literature_tools": "literature_tools",
            "supervisor": "supervisor",
            END: END
        }
    )
    workflow.add_edge("literature_tools", "literature_agent")

    return workflow


def initialize_multiagent():
    workflow = build_workflow()
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)