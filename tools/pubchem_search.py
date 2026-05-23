"""PubChem 检索工具——通过 SMILES 或名称查询化合物身份信息

解决问题：ChemSpider 不支持 SMILES 检索，用户通过 Ketcher 绘制分子后
无法直接通过 SMILES 获取化合物名称。PubChem 的 REST API 免费、无需 API Key。

API 文档: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""
import json
import time
import logging
from typing import Optional, Type
import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from tools.chem_memory import add_to_memory

logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
REQUEST_DELAY = 0.3  # PubChem 建议 ≤5 requests/sec


def search_pubchem_by_smiles(smiles: str) -> Optional[dict]:
    """通过 SMILES 检索 PubChem，返回化合物身份信息（名称、IUPAC、分子式等）"""
    props = "Title,IUPACName,MolecularFormula,MolecularWeight,InChIKey,CanonicalSMILES"
    url = f"{PUBCHEM_BASE}/compound/smiles/{smiles}/property/{props}/JSON"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            logger.info("PubChem SMILES 未命中: %s", smiles[:30])
            return None
        resp.raise_for_status()
        data = resp.json()
        props_list = data.get("PropertyTable", {}).get("Properties", [])
        if not props_list:
            return None
        return props_list[0]
    except requests.exceptions.RequestException as e:
        logger.warning("PubChem SMILES 请求失败: %s", e)
        return None


def search_pubchem_by_name(name: str) -> Optional[dict]:
    """通过名称检索 PubChem，返回化合物身份信息"""
    props = "Title,IUPACName,MolecularFormula,MolecularWeight,InChIKey,CanonicalSMILES"
    url = f"{PUBCHEM_BASE}/compound/name/{name}/property/{props}/JSON"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            logger.info("PubChem 名称未命中: %s", name)
            return None
        resp.raise_for_status()
        data = resp.json()
        props_list = data.get("PropertyTable", {}).get("Properties", [])
        if not props_list:
            return None
        return props_list[0]
    except requests.exceptions.RequestException as e:
        logger.warning("PubChem 名称请求失败: %s", e)
        return None


def search_pubchem_synonyms(smiles: str, limit: int = 10) -> list:
    """通过 SMILES 获取化合物的同义词列表（中文名、商品名等）"""
    url = f"{PUBCHEM_BASE}/compound/smiles/{smiles}/synonyms/JSON"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        synonyms = data.get("InformationList", {}).get("Information", [])
        if not synonyms:
            return []
        all_syns = synonyms[0].get("Synonym", [])
        return all_syns[:limit]
    except requests.exceptions.RequestException as e:
        logger.warning("PubChem 同义词请求失败: %s", e)
        return []


class PubChemInput(BaseModel):
    smiles: str = Field(description="化合物的 SMILES 结构式，例如 'CC(=O)Oc1ccccc1C(=O)O'")
    get_synonyms: bool = Field(default=True, description="是否同时获取同义词列表")


class PubChemTool(BaseTool):
    """通过 PubChem REST API 将 SMILES 解析为化合物名称和身份信息"""
    name: str = "PubChemSearch"
    description: str = (
        "当用户提供了化合物的 SMILES 结构式（例如通过分子绘图工具），需要识别其化学名称、"
        "IUPAC 名、分子式、分子量等信息时，必须首先调用此工具。\n"
        "也支持通过化合物名称查询，但优先用于 SMILES→名称 的解析。\n"
        "输入：SMILES 字符串。\n"
        "返回：包含化合物名称（Title）、IUPAC 名、分子式、分子量、InChIKey、同义词列表的 JSON。"
    )
    args_schema: Type[BaseModel] = PubChemInput

    def _run(self, smiles: str, get_synonyms: bool = True) -> str:
        time.sleep(REQUEST_DELAY)

        result = search_pubchem_by_smiles(smiles)
        if result is None:
            return json.dumps({
                "status": "not_found",
                "message": f"PubChem 中未找到 SMILES '{smiles[:50]}' 对应的化合物。请检查结构是否正确。",
                "smiles": smiles
            }, ensure_ascii=False)

        title = result.get("Title", "")
        if get_synonyms and title:
            syns = search_pubchem_synonyms(smiles)
            result["Synonyms"] = syns  # noqa: E702  (keeping for readability)

        # 存入知识库，以 SMILES 为键
        memory_data = {
            "id": result.get("CID"),
            "commonName": title,
            "smiles": result.get("CanonicalSMILES", smiles),
            "molecularFormula": result.get("MolecularFormula", ""),
            "molecularWeight": result.get("MolecularWeight", ""),
            "iupacName": result.get("IUPACName", ""),
            "inchikey": result.get("InChIKey", ""),
            "source": "PubChem",
        }
        add_to_memory(title or smiles, memory_data)

        logger.info("PubChem 命中: %s (CID=%s)", title, result.get("CID"))
        return json.dumps(result, indent=2, ensure_ascii=False)

    def _arun(self, smiles: str, get_synonyms: bool = True):
        raise NotImplementedError
