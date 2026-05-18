"""单智能体测试"""
import pytest


def test_initialize_agent_returns_compiled_graph():
    """initialize_agent() 返回可 invoke 的编译后图"""
    from agents.agent import initialize_agent
    agent = initialize_agent()
    assert agent is not None
    assert hasattr(agent, 'invoke')


def test_agent_tools_configured(mocker):
    """Agent 应配置 ChemicalMemoryTool 和 ChemSpiderTool"""
    from agents.agent import initialize_agent
    from tools.chem_memory import ChemicalMemoryTool
    from tools.chemspider_search import ChemSpiderTool

    # Mock create_agent 来截获传递给它的参数
    mock_create = mocker.patch('agents.agent.create_agent')
    initialize_agent()

    call_kwargs = mock_create.call_args.kwargs
    tool_classes = [type(t) for t in call_kwargs.get("tools", [])]
    assert ChemicalMemoryTool in tool_classes
    assert ChemSpiderTool in tool_classes
