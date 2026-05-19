"""多智能体全图集成测试——验证整个 StateGraph 的条件路由是否正确"""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


@pytest.fixture
def mock_all_llms(mocker):
    """替换所有 LLM 实例为 MagicMock，防止真实 API 调用"""
    import agents.multiagent as mam

    mock_decision = MagicMock()
    mock_summary = MagicMock()
    mock_chem = MagicMock()
    mock_lit = MagicMock()

    mocker.patch('agents.multiagent.decision_llm', mock_decision)
    mocker.patch('agents.multiagent.summary_llm', mock_summary)
    mocker.patch('agents.multiagent.chem_model', mock_chem)
    mocker.patch('agents.multiagent.literature_model', mock_lit)

    return type('Mocks', (), {
        'mam': mam,
        'decision': mock_decision,
        'summary': mock_summary,
        'chem': mock_chem,
        'lit': mock_lit,
    })


def test_chem_agent_route_to_end(mocker, mock_all_llms):
    """Supervisor → chem_agent(无tool_calls) → summary → END"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "chem_agent"}')
    m.chem.invoke.return_value = AIMessage(content="Aspirin的化学结构是...")
    m.summary.invoke.return_value = AIMessage(content="## 总结\nAspirin(阿司匹林)...")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="aspirin")]},
        config={"configurable": {"thread_id": "test-1"}}
    )

    messages = result.get("messages", [])
    assert len(messages) > 0
    last_ai = [m for m in messages if isinstance(m, AIMessage) and m.content]
    assert len(last_ai) > 0
    assert "总结" in last_ai[-1].content


def test_chem_agent_with_tool_loop(mocker, mock_all_llms):
    """Supervisor → chem_agent(tool_call) → chem_tools → chem_agent(无tool_call) → summary → END"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "chem_agent"}')

    call1 = AIMessage(
        content="",
        tool_calls=[{"name": "ChemicalMemory", "args": {"query": "aspirin"}, "id": "call_1"}]
    )
    call2 = AIMessage(content="已在记忆库中找到Aspirin数据：SMILES=CC(=O)Oc1ccccc1C(=O)O")
    m.chem.invoke.side_effect = [call1, call2]

    # Mock ToolNode: 替换 chem_tool_node 为 MagicMock
    def fake_tool_node(state):
        return {"messages": [ToolMessage(content="Aspirin数据JSON...", tool_call_id="call_1")]}
    mocker.patch('agents.multiagent.chem_tool_node', fake_tool_node)

    m.summary.invoke.return_value = AIMessage(content="## 查询结果\n- SMILES: CC(=O)Oc1ccccc1C(=O)O")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="aspirin")]},
        config={"configurable": {"thread_id": "test-2"}}
    )

    messages = result.get("messages", [])
    assert len(messages) > 0
    assert m.chem.invoke.call_count == 2


def test_literature_agent_route_to_end(mock_all_llms):
    """Supervisor → literature_agent(无tool_calls) → summary → END"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "literature_agent"}')
    m.lit.invoke.return_value = AIMessage(content="找到3篇相关文献")
    m.summary.invoke.return_value = AIMessage(content="## 文献检索结果\n| 序号 | 标题 | ...")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="查询AIE荧光分子合成文献")]},
        config={"configurable": {"thread_id": "test-3"}}
    )

    messages = result.get("messages", [])
    last_ai = [m for m in messages if isinstance(m, AIMessage) and m.content]
    assert len(last_ai) > 0


def test_finish_directly(mock_all_llms):
    """Supervisor 直接决定 finish → END（不调用专家）"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "finish"}')
    m.summary.invoke.return_value = AIMessage(content="你好！有什么可以帮你的？")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="你好")]},
        config={"configurable": {"thread_id": "test-4"}}
    )

    # 确认专家模型从未被调用
    m.chem.invoke.assert_not_called()
    m.lit.invoke.assert_not_called()


def test_max_rounds_guard_stops_execution(mock_all_llms):
    """round_count=5 时强制结束"""
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


def test_both_parallel_dispatch(mock_all_llms):
    """Supervisor decides both -> parallel chem + lit -> summary -> END"""
    m = mock_all_llms
    m.decision.invoke.return_value = MagicMock(content='{"next": "both"}')
    m.chem.invoke.return_value = AIMessage(content="Aspirin: SMILES=CC(=O)Oc1ccccc1C(=O)O, MW=180.16")
    m.lit.invoke.return_value = AIMessage(content="Found 5 papers on Aspirin synthesis after 2020")
    m.summary.invoke.return_value = AIMessage(content="## Aspirin\n\nChem info + 5 literature references")

    from agents.multiagent import initialize_multiagent
    app = initialize_multiagent()

    result = app.invoke(
        {"messages": [HumanMessage(content="Aspirin industrial synthesis, 5 papers after 2020")]},
        config={"configurable": {"thread_id": "test-both-1"}}
    )

    messages = result.get("messages", [])
    assert len(messages) > 0
    # 两个专家都应该被调用
    m.chem.invoke.assert_called()
    m.lit.invoke.assert_called()
    # summary 应该生成最终回复
    m.summary.invoke.assert_called()
    # 最终回复应包含内容
    final_ai = [m for m in messages if isinstance(m, AIMessage) and "Aspirin" in str(m.content)]
    assert len(final_ai) > 0
