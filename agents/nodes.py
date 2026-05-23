"""多智能体图节点函数——通过 import agents.multiagent as _mam 在调用时解析 LLM 引用"""
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import END
from agents.state import MultiAgentState

logger = logging.getLogger(__name__)


def _get_mam():
    """延迟导入避免与 multiagent.py 的循环导入——仅在各函数调用时执行"""
    import agents.multiagent as _mam
    return _mam


def _stream_summary(llm, messages):
    """MagicMock -> invoke; real LLM (streaming=True) -> invoke, LangGraph auto-captures token chunks"""
    if type(llm).__module__ == 'unittest.mock':
        return llm.invoke(messages)
    return llm.invoke(messages)


def _clean_messages(messages: list) -> list:
    """只保留 HumanMessage 和不带 tool_calls 的 AIMessage，供最终总结使用"""
    clean = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            clean.append(msg)
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            clean.append(msg)
    return clean


def _ensure_tool_pairs(messages: list) -> list:
    """
    确保消息列表中所有 AIMessage(tool_calls) 都有对应的 ToolMessage 紧随。
    并行 fan-out 可能导致其他 agent 的消息插入 tool_call/ToolMessage 之间，
    DeepSeek API 会拒绝这种格式。此函数移除那些响应不完整的 tool_call 消息。
    """
    result = []
    pending = {}  # tool_call_id -> index in result of the AIMessage

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            idx = len(result)
            result.append(msg)
            for tc in msg.tool_calls:
                pending[tc["id"]] = idx

        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id in pending:
                result.append(msg)
                del pending[msg.tool_call_id]
            # ToolMessage without matching tool_call → discard

        elif isinstance(msg, HumanMessage):
            # HumanMessage between tool_call and ToolMessage corrupts the pairing.
            # Remove any un-responded tool_call AIMessages, then add the HumanMessage.
            if pending:
                for idx in set(pending.values()):
                    result[idx] = None  # mark for removal
                pending.clear()
            result.append(msg)

        else:
            # Plain AIMessage or other types: only add if no pending tool calls
            if not pending:
                result.append(msg)
            # Otherwise drop it (interleaved message from other agent)

    # Final cleanup: remove None markers
    if None in result:
        result = [m for m in result if m is not None]

    return result


def supervisor_node(state: MultiAgentState) -> dict:
    logger.debug("supervisor_node 被调用")
    clean_history = _clean_messages(state["messages"])
    round_count = state.get("round_count", 0)

    if round_count >= 5:
        logger.warning("达到最大循环轮次，强制结束")
        return {"messages": [AIMessage(content="抱歉，处理超时，请稍后重试。")], "next": END}

    system_prompt = (
        "你是 ChemAssist 的主管。你的唯一职责是根据用户请求，决定调用哪个专家。\n"
        "你必须严格按照以下JSON格式输出你的决策，不要包含任何其他文字：\n\n"
        '{"next": "<决策词>"}\n\n'
        "决策词只能是以下四个之一：\n"
        '- "chem_agent"：查询化学物质的结构与性质\n'
        '- "literature_agent"：检索学术文献\n'
        '- "both"：需要同时查询化学物质相关信息和文献（并行处理）\n'
        '- "finish"：任务已由专家完成，或不需要专家，由你直接友好回复\n\n'

        "以下是一些示范案例，请参考:\n"
        "案例1. 用户输入：苯甲酸。 你应该输出：\n"
        '{"next": "chem_agent"}\n\n'

        "案例2. 用户输入：请查询AIE荧光分子合成的相关文献。 你应该输出：\n"
        '{"next": "literature_agent"}\n\n'

        "案例3. 用户输入：告诉我甲基丙烯酸甲酯的化学结构，及它可以用于哪些反应？ 你应该输出：\n"
        '{"next": "both"}\n\n'

        "案例4. 用户输入：今天天气真不错哈。 你应该输出：\n"
        '{"next": "finish"}\n\n'

        "案例5. 接受到专家agent返回的格式化化学物质信息或文献信息。 你应该输出：\n"
        '{"next": "finish"}\n\n'
    )

    messages = [HumanMessage(content=system_prompt)] + clean_history

    try:
        response = _get_mam().decision_llm.invoke(messages)
        logger.debug("主管原始回复: %s", response.content)
        decision_json = json.loads(response.content)
        decision = decision_json.get("next", "finish")
        logger.debug("解析后决策: %s", decision)
    except Exception as e:
        logger.warning("主管决策失败: %s", e)
        return {"messages": [AIMessage(content=f"决策错误: {e}")], "next": END}

    if decision == "both":
        logger.info("决策 both，顺序调度化学专家 → 文献专家")
        return {"next": "both", "round_count": round_count + 1, "both_active": True}
    elif decision == "finish":
        try:
            final = _stream_summary(_get_mam().summary_llm, clean_history)
            logger.debug("主管回复: %s", final.content[:100])
            return {"messages": [final], "next": END}
        except Exception as e:
            logger.warning("主管生成回复失败: %s", e)
            return {"messages": [AIMessage(content="抱歉，处理时遇到错误。")], "next": END}
    elif decision in ("chem_agent", "literature_agent"):
        return {"next": decision, "round_count": round_count + 1}
    else:
        return {"messages": [AIMessage(content="抱歉，无法理解你的请求。")], "next": END}


def chem_agent_node(state: MultiAgentState) -> dict:
    logger.debug("chem_agent_node 被调用")
    safe_messages = _ensure_tool_pairs(list(state["messages"]))
    messages = [HumanMessage(content=(
        "你是化学信息学专家，使用工具查询物质结构数据。遵循以下流程：\n"
        "1. 先调用 ChemicalMemory 检查知识库中是否已有该化合物。\n"
        "2. 如未命中，调用 ChemSpiderSearch（按名称）或 PubChemSearch（按 SMILES）获取化合物数据。\n"
        "3. 获得 SMILES 后，必须调用 ChemCalc 计算分子描述符并生成 2D 结构图。\n"
        "4. 如有结构分析需求，调用 StructureAnnotator 识别官能团、手性中心和核心骨架。\n"
        "重要：拿到 SMILES 就立刻调用 ChemCalc，不要等用户要求。"
    ))] + safe_messages
    response = _get_mam().chem_model.invoke(messages)
    if response.tool_calls:
        logger.debug("化学专家调用工具: %s", [tc['name'] for tc in response.tool_calls])
        return {"messages": [response], "next": "chem_tools"}
    # both 顺序链：chem 完成后直接路由到 literature_agent
    if state.get("both_active"):
        logger.debug("化学专家任务完成，顺序链到文献专家")
        return {"messages": [response], "next": "literature_agent", "both_active": False}
    logger.debug("化学专家任务完成，返回summary")
    return {"messages": [response], "next": "summary"}


def literature_agent_node(state: MultiAgentState) -> dict:
    logger.debug("literature_agent_node 被调用")
    safe_messages = _ensure_tool_pairs(list(state["messages"]))
    messages = [HumanMessage(content="你是文献检索专家，使用工具获取学术论文信息。")] + safe_messages
    response = _get_mam().literature_model.invoke(messages)
    if response.tool_calls:
        logger.debug("文献专家调用工具: %s", [tc['name'] for tc in response.tool_calls])
        return {"messages": [response], "next": "literature_tools"}
    logger.debug("文献专家任务完成，返回summary")
    return {"messages": [response], "next": "summary"}


def summary_node(state: MultiAgentState) -> dict:
    logger.debug("summary_node 被调用")
    clean_history = _clean_messages(state["messages"])
    format_instruction = HumanMessage(content=(
        "请根据上面的对话生成最终回复。严格遵循以下规则：\n"
        "1. 如果对话中有化合物的 SMILES 结构式，必须调用 ChemCalc 工具生成 2D 结构图。\n"
        "2. 如果需要整理文献表格，请输出 Markdown 表格。\n"
        "3. 如果是化学查询，请总结关键化学信息（名称、分子式、分子量、SMILES）。\n"
        "4. 如果用户绘制了分子结构，优先展示结构分析结果。\n"
        "5. 回复末尾主动询问用户是否需要进一步的分析（如官能团标注、结构比较）。"
    ))
    messages = clean_history + [format_instruction]
    try:
        final = _stream_summary(_get_mam().summary_llm, messages)
        logger.debug("最终回复: %s", final.content[:100])
        return {"messages": [final]}
    except Exception as e:
        logger.warning("汇总失败: %s", e)
        return {"messages": [AIMessage(content="抱歉，处理时遇到错误。")]}
