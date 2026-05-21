"""多智能体全图集成测试——验证整个 StateGraph 的条件路由是否正确"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


@pytest.fixture
def mock_all_llms():
    """替换所有 LLM 实例为 MagicMock，防止真实 API 调用"""
    import agents.multiagent as mam

    mock_decision = MagicMock()
    mock_summary = MagicMock()
    mock_chem = MagicMock()
    mock_lit = MagicMock()

    p1 = patch('agents.multiagent.decision_llm', mock_decision)
    p2 = patch('agents.multiagent.summary_llm', mock_summary)
    p3 = patch('agents.multiagent.chem_model', mock_chem)
    p4 = patch('agents.multiagent.literature_model', mock_lit)
    p1.start(); p2.start(); p3.start(); p4.start()

    yield type('Mocks', (), {
        'mam': mam,
        'decision': mock_decision,
        'summary': mock_summary,
        'chem': mock_chem,
        'lit': mock_lit,
        'patchers': (p1, p2, p3, p4),
    })

    p1.stop(); p2.stop(); p3.stop(); p4.stop()


def test_chem_agent_route_to_end(mock_all_llms):
    """Supervisor -> chem_agent(no tool_calls) -> summary -> END"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "chem_agent"}')
    m.chem.invoke.return_value = AIMessage(content="Aspirin: SMILES=..., MW=180.16")
    m.summary.invoke.return_value = AIMessage(content="## Summary\nAspirin...")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="aspirin")], "round_count": 0, "next": None},
        config={"configurable": {"thread_id": "test-1"}}
    )

    messages = result.get("messages", [])
    assert len(messages) > 0
    last_ai = [m for m in messages if isinstance(m, AIMessage) and m.content]
    assert len(last_ai) > 0
    assert m.chem.invoke.call_count == 1


def test_chem_agent_with_tool_loop(mock_all_llms):
    """Supervisor -> chem_agent(tool_call) -> chem_tools -> chem_agent(no tool_call) -> summary -> END"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "chem_agent"}')

    call1 = AIMessage(content="", tool_calls=[{"name": "ChemicalMemory", "args": {"query": "aspirin"}, "id": "call_1"}])
    call2 = AIMessage(content="Found Aspirin in memory: SMILES=CC(=O)Oc1ccccc1C(=O)O")
    m.chem.invoke.side_effect = [call1, call2]

    def fake_tool_node(state):
        return {"messages": [ToolMessage(content="Aspirin data JSON...", tool_call_id="call_1")]}
    tp = patch('agents.multiagent.chem_tool_node', fake_tool_node)
    tp.start()

    m.summary.invoke.return_value = AIMessage(content="## Result\n- SMILES: CC(=O)Oc1ccccc1C(=O)O")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="aspirin")], "round_count": 0, "next": None},
        config={"configurable": {"thread_id": "test-2"}}
    )

    tp.stop()

    messages = result.get("messages", [])
    assert len(messages) > 0
    assert m.chem.invoke.call_count == 2


def test_literature_agent_route_to_end(mock_all_llms):
    """Supervisor -> literature_agent(no tool_calls) -> summary -> END"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "literature_agent"}')
    m.lit.invoke.return_value = AIMessage(content="Found 3 papers")
    m.summary.invoke.return_value = AIMessage(content="## Literature Results")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="AIE fluorescent molecules")], "round_count": 0, "next": None},
        config={"configurable": {"thread_id": "test-3"}}
    )

    messages = result.get("messages", [])
    last_ai = [m for m in messages if isinstance(m, AIMessage) and m.content]
    assert len(last_ai) > 0


def test_finish_directly(mock_all_llms):
    """Supervisor directly finish -> END (no experts called)"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "finish"}')
    m.summary.invoke.return_value = AIMessage(content="Hello! How can I help?")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="hello")], "round_count": 0, "next": None},
        config={"configurable": {"thread_id": "test-4"}}
    )

    m.chem.invoke.assert_not_called()
    m.lit.invoke.assert_not_called()


def test_max_rounds_guard_stops_execution(mock_all_llms):
    """round_count=5 forces end"""
    m = mock_all_llms

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="aspirin")], "next": None, "round_count": 5},
        config={"configurable": {"thread_id": "test-5"}}
    )

    messages = result.get("messages", [])
    assert len(messages) > 0
    error_texts = [m.content for m in messages if isinstance(m, AIMessage) and "超时" in str(m.content)]
    assert len(error_texts) > 0


def test_both_sequential_chain(mock_all_llms):
    """Supervisor decides both -> chem -> lit -> summary (sequential chain)"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "both"}')
    m.chem.invoke.return_value = AIMessage(content="Aspirin: SMILES=CC(=O)Oc1ccccc1C(=O)O, MW=180.16")
    m.lit.invoke.return_value = AIMessage(content="Found 5 papers on Aspirin synthesis after 2020")
    m.summary.invoke.return_value = AIMessage(content="## Aspirin\n\nChem info + 5 literature references")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="Aspirin industrial synthesis, 5 papers after 2020")], "round_count": 0, "next": None},
        config={"configurable": {"thread_id": "test-both-1"}}
    )

    messages = result.get("messages", [])
    assert len(messages) > 0
    m.chem.invoke.assert_called()
    m.lit.invoke.assert_called()
    m.summary.invoke.assert_called()
