"""ChemAssist 工具注册中心

所有 Agent 工具在此统一注册。新增工具只需在此文件中添加一行 import 和一行实例化即可。
multiagent.py 从此处导入工具列表，不再逐个硬编码。
"""
from tools.chem_memory import ChemicalMemoryTool
from tools.chemspider_search import ChemSpiderTool
from tools.pubchem_search import PubChemTool
from tools.chem_calc import ChemCalcTool, StructureAnnotationTool
from tools.literature_search import CrossrefSearchTool

# 化学类工具——按调用优先级排列：先查本地缓存 → API 名称检索 → SMILES 解析 → 计算 → 标注
CHEM_TOOLS = [
    ChemicalMemoryTool(),
    ChemSpiderTool(),
    PubChemTool(),
    ChemCalcTool(),
    StructureAnnotationTool(),
]

# 文献类工具
LITERATURE_TOOLS = [
    CrossrefSearchTool(),
]
