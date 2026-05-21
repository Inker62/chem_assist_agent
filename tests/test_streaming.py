"""_stream_summary 函数测试——验证 invoke 调用和 mock 回退"""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage


class TestStreamSummary:
    def test_calls_invoke_and_returns_result(self):
        from agents.nodes import _stream_summary
        mock = MagicMock()
        mock.invoke.return_value = AIMessage(content="streamed result")
        result = _stream_summary(mock, [HumanMessage(content="test")])
        assert result.content == "streamed result"
        mock.invoke.assert_called_once()

    def test_magicmock_bypasses_stream_check(self):
        """MagicMock should go directly to invoke without trying stream"""
        from agents.nodes import _stream_summary
        mock = MagicMock()
        mock.invoke.return_value = AIMessage(content="mock response")
        result = _stream_summary(mock, [HumanMessage(content="test")])
        assert result.content == "mock response"
        mock.invoke.assert_called_once()

    def test_non_mock_llm_uses_invoke(self):
        """Any non-MagicMock object should also use invoke directly"""
        from agents.nodes import _stream_summary

        class RealLLM:
            def invoke(self, messages):
                return AIMessage(content="real response")

        llm = RealLLM()
        result = _stream_summary(llm, [HumanMessage(content="test")])
        assert result.content == "real response"


class TestSummaryNodeStreaming:
    def test_uses_stream_summary_path(self, mocker):
        """summary_node calls summary_llm to produce final response"""
        from agents import nodes as nod
        from agents import multiagent as mam

        mock_summary_llm = MagicMock()
        mock_summary_llm.invoke.return_value = AIMessage(content="summary result")
        mocker.patch.object(mam, 'summary_llm', mock_summary_llm)

        state = {
            "messages": [HumanMessage(content="aspirin"), AIMessage(content="data found")],
            "next": None,
            "round_count": 2,
        }
        result = nod.summary_node(state)
        assert len(result["messages"]) == 1
        assert "summary result" in result["messages"][0].content


class TestSupervisorFinishStreaming:
    def test_finish_uses_stream_summary_path(self, mocker):
        """supervisor_node finish branch calls summary_llm to produce response"""
        from agents import nodes as nod
        from agents import multiagent as mam

        mock_decision = MagicMock()
        mock_decision.invoke.return_value = MagicMock(content='{"next": "finish"}')
        mocker.patch.object(mam, 'decision_llm', mock_decision)

        mock_summary_llm = MagicMock()
        mock_summary_llm.invoke.return_value = AIMessage(content="direct reply")
        mocker.patch.object(mam, 'summary_llm', mock_summary_llm)

        state = {
            "messages": [HumanMessage(content="hello")],
            "next": None,
            "round_count": 0,
        }
        result = nod.supervisor_node(state)
        assert result["next"] == "__end__"
        assert len(result["messages"]) == 1
        assert "direct reply" in result["messages"][0].content
