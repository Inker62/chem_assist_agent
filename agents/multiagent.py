"""多智能体协同模块——LLM 实例 + 向后兼容 re-export

⚠️ 顺序约束：Section 1（LLM/ToolNode 定义）必须在 Section 2（子模块导入）之前。
nodes.py 通过 import agents.multiagent as _mam 在调用时解析 LLM 引用。
"""
import logging
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from tools.chemspider_search import ChemSpiderTool
from tools.chem_memory import ChemicalMemoryTool
from tools.chem_calc import ChemCalcTool
from tools.literature_search import CrossrefSearchTool

logger = logging.getLogger(__name__)

# ==================== Section 1: LLM 与 ToolNode 实例 ====================

decision_llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    extra_body={
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"}
    }
)

summary_llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    streaming=True,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}}
)

chem_tools = [ChemSpiderTool(), ChemicalMemoryTool(), ChemCalcTool()]
literature_tools = [CrossrefSearchTool()]

chem_tool_node = ToolNode(chem_tools)
literature_tool_node = ToolNode(literature_tools)

chem_model = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}}
).bind_tools(chem_tools)

literature_model = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}}
).bind_tools(literature_tools)


# ==================== Section 2: 子模块 Re-export ====================
from agents.state import MultiAgentState  # noqa: E402, F401
from agents.nodes import _clean_messages, supervisor_node, chem_agent_node, literature_agent_node, summary_node  # noqa: E402, F401
from agents.graph import build_workflow, initialize_multiagent  # noqa: E402, F401
