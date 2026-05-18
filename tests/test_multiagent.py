"""多智能体图节点和图结构单元测试"""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Send
from langgraph.graph import END


# ============================================================
# 辅助函数
# ============================================================
def _make_state(content="aspirin", round_count=0, extra_messages=None):
    state = {
        "messages": [HumanMessage(content=content)],
        "next": None,
        "round_count": round_count,
    }
    if extra_messages:
        state["messages"].extend(extra_messages)
    return state


def _mock_llm(mocker, path, invoke_return):
    """替换模块级 LLM 实例为 MagicMock（ChatOpenAI 是 Pydantic 模型，不能直接 patch 方法）"""
    mock = MagicMock()
    mock.invoke.return_value = invoke_return
    mocker.patch(path, mock)
    return mock


# ============================================================
# _clean_messages 测试
# ============================================================
class TestCleanMessages:
    def test_returns_empty_for_empty_list(self):
        from agents.multiagent import _clean_messages
        assert _clean_messages([]) == []

    def test_keeps_human_and_plain_ai_removes_toolcall_ai(self):
        from agents.multiagent import _clean_messages
        messages = [
            HumanMessage(content="你好"),
            AIMessage(content="回复1"),
            AIMessage(content="", tool_calls=[{"name": "tool", "args": {}, "id": "1"}]),
            AIMessage(content="回复2"),
        ]
        result = _clean_messages(messages)
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert result[1].content == "回复1"
        assert result[2].content == "回复2"

    def test_mixed_bag_filters_toolmessage(self):
        from agents.multiagent import _clean_messages
        messages = [
            HumanMessage(content="查询aspirin"),
            AIMessage(content="", tool_calls=[{"name": "ChemSpiderSearch", "args": {}, "id": "1"}]),
            ToolMessage(content="tool result", tool_call_id="1"),
            AIMessage(content="查到结果了"),
        ]
        result = _clean_messages(messages)
        # ToolMessage 不是 HumanMessage 也不是 AIMessage，应被过滤
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert result[1].content == "查到结果了"


# ============================================================
# supervisor_node 测试
# ============================================================
class TestSupervisorNode:
    def test_decides_chem_agent(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.decision_llm',
                  MagicMock(content='{"next": "chem_agent"}'))
        result = mam.supervisor_node(_make_state())
        assert result["next"] == "chem_agent"
        assert result["round_count"] == 1

    def test_decides_literature_agent(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.decision_llm',
                  MagicMock(content='{"next": "literature_agent"}'))
        result = mam.supervisor_node(_make_state("请查询AIE相关文献"))
        assert result["next"] == "literature_agent"

    def test_decides_both_returns_send_list(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.decision_llm',
                  MagicMock(content='{"next": "both"}'))
        result = mam.supervisor_node(_make_state("甲基丙烯酸甲酯的化学结构和相关文献"))
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], Send)
        assert isinstance(result[1], Send)
        assert result[0].node == "chem_agent"
        assert result[1].node == "literature_agent"
        assert result[0].arg["round_count"] == 1

    def test_decides_finish_calls_summary_llm(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.decision_llm',
                  MagicMock(content='{"next": "finish"}'))
        _mock_llm(mocker, 'agents.multiagent.summary_llm',
                  AIMessage(content="你好，有什么可以帮你的？"))
        result = mam.supervisor_node(_make_state("你好"))
        assert result["next"] == END
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "你好，有什么可以帮你的？"

    def test_max_rounds_forces_end(self, mocker):
        from agents import multiagent as mam
        result = mam.supervisor_node(_make_state(round_count=5))
        assert result["next"] == END
        assert "超时" in result["messages"][0].content

    def test_invalid_json_returns_error(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.decision_llm',
                  MagicMock(content="这不是合法的JSON"))
        result = mam.supervisor_node(_make_state())
        assert result["next"] == END
        assert "错误" in result["messages"][0].content

    def test_unknown_decision_returns_error(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.decision_llm',
                  MagicMock(content='{"next": "invalid_option"}'))
        result = mam.supervisor_node(_make_state())
        assert result["next"] == END
        assert "无法理解" in result["messages"][0].content

    def test_finish_llm_error_returns_fallback(self, mocker):
        from agents import multiagent as mam
        # decision_llm 返回 finish
        mock_decision = MagicMock()
        mock_decision.invoke.return_value = MagicMock(content='{"next": "finish"}')
        mocker.patch('agents.multiagent.decision_llm', mock_decision)
        # summary_llm 抛出异常
        mock_summary = MagicMock()
        mock_summary.invoke.side_effect = RuntimeError("LLM挂了")
        mocker.patch('agents.multiagent.summary_llm', mock_summary)

        result = mam.supervisor_node(_make_state())
        assert result["next"] == END
        assert "错误" in result["messages"][0].content


# ============================================================
# chem_agent_node 测试
# ============================================================
class TestChemAgentNode:
    def test_with_tool_calls_routes_to_chem_tools(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.chem_model',
                  AIMessage(content="",
                            tool_calls=[{"name": "ChemSpiderSearch", "args": {"query": "aspirin"}, "id": "call_1"}]))
        result = mam.chem_agent_node(_make_state())
        assert result["next"] == "chem_tools"

    def test_without_tool_calls_routes_to_summary(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.chem_model',
                  AIMessage(content="Aspirin的SMILES是..."))
        result = mam.chem_agent_node(_make_state())
        assert result["next"] == "summary"


# ============================================================
# literature_agent_node 测试
# ============================================================
class TestLiteratureAgentNode:
    def test_with_tool_calls_routes_to_literature_tools(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.literature_model',
                  AIMessage(content="",
                            tool_calls=[{"name": "CrossrefSearch", "args": {"query": "AIE"}, "id": "call_1"}]))
        result = mam.literature_agent_node(_make_state("请检索AIE文献"))
        assert result["next"] == "literature_tools"

    def test_without_tool_calls_routes_to_summary(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.literature_model',
                  AIMessage(content="找到3篇相关文献..."))
        result = mam.literature_agent_node(_make_state("请检索AIE文献"))
        assert result["next"] == "summary"


# ============================================================
# summary_node 测试
# ============================================================
class TestSummaryNode:
    def test_success_returns_final_message(self, mocker):
        from agents import multiagent as mam
        _mock_llm(mocker, 'agents.multiagent.summary_llm',
                  AIMessage(content="## 总结\nAspirin是一种..."))
        state = {
            "messages": [HumanMessage(content="aspirin"), AIMessage(content="查到Aspirin数据")],
            "next": None,
            "round_count": 2,
        }
        result = mam.summary_node(state)
        assert len(result["messages"]) == 1
        assert "总结" in result["messages"][0].content

    def test_llm_error_returns_fallback(self, mocker):
        from agents import multiagent as mam
        mock = MagicMock()
        mock.invoke.side_effect = RuntimeError("LLM挂了")
        mocker.patch('agents.multiagent.summary_llm', mock)
        state = {
            "messages": [HumanMessage(content="aspirin")],
            "next": None,
            "round_count": 1,
        }
        result = mam.summary_node(state)
        assert "错误" in result["messages"][0].content


# ============================================================
# build_workflow 测试
# ============================================================
class TestBuildWorkflow:
    def test_has_all_required_nodes(self):
        from agents.multiagent import build_workflow
        wf = build_workflow()
        nodes = wf.nodes
        required = {"supervisor", "chem_agent", "chem_tools", "literature_agent", "literature_tools", "summary"}
        missing = required - set(nodes.keys())
        assert not missing, f"缺少节点: {missing}"
