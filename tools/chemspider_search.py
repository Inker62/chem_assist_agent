import os
import time
from typing import Optional, Type, Dict, Any
import json
import requests
from dotenv import load_dotenv
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from tools.chem_memory import add_to_memory


load_dotenv()
RSC_API_KEY = os.getenv("RSC_API_KEY")
BASE_URL = "https://api.rsc.org/compounds/v1"

def search_compound_by_name(compound_name: str) -> Optional[Dict[str, Any]]:
    """
    当用户需要根据化学物质的名称、俗称、系统命名等方式查询物质对应的SMILES结构式、InChiIKey等化学结构信息时调用此工具
    采用两步法调用：
        1.通过物质名称查询ChemSpiderID
        2.通过ChemSpiderID查询物质结构信息，如SMILES结构式、InChIKey等
    """
    headers = {"apikey": RSC_API_KEY,"Content-Type":"application/json","Accept":"application/json"}

    search_url = f"{BASE_URL}/filter/name"
    search_resp = requests.post(search_url, json={"name":compound_name}, headers=headers)
    #print("DEBUG: name search response:", search_resp.text)
    search_resp.raise_for_status()
    query_id=search_resp.json().get("queryId")
    if not query_id:
        return None

    status_url = f"{BASE_URL}/filter/{query_id}/status"
    max_retries = 15
    for i in range(max_retries):
        time.sleep(3)
        status_resp = requests.get(status_url, headers=headers)
        status_resp.raise_for_status()
        status = status_resp.json().get("status")
        if status == "Complete":
            break
    else:  # 超时
        return None


    results_url = f"{BASE_URL}/filter/{query_id}/results"
    results_resp = requests.get(results_url, headers=headers)
    results_resp.raise_for_status()
    results=results_resp.json().get("results",[])
    if not results:
        return None

    first_result = results[0]
    if isinstance(first_result, int):
        compound_id = first_result
    elif isinstance(first_result, dict):
        compound_id = first_result.get("id")
    else:
        return None

    detail_url = f"{BASE_URL}/records/{compound_id}/details"
    fields = "SMILES, Formula, InChI, InChIKey, StdInChI, StdInChIKey, AverageMass, MolecularWeight, MonoisotopicMass, NominalMass, CommonName, ReferenceCount, DataSourceCount, PubMedCount, RSCCount, Mol2D, Mol3D"
    detail_resp = requests.get(detail_url, headers=headers, params={"fields":fields})
    detail_resp.raise_for_status()

    return detail_resp.json()

class ChemSpiderInput(BaseModel):
    """ ChemSpider查询的输入参数    """
    query: str = Field(
        description="需要查询的化学物质名称，可以是系统命名、商品名或俗称，例如：'aspirin','caffeine','acetaminophen'。"
    )

class ChemSpiderTool(BaseTool):
    """通过RSC ChemSpider API检索化学物质的SMILES、InChIKey等信息"""
    name: str = "ChemSpiderSearch"
    description: str=(
        "当用户需要根据化学物质的名称、俗称或系统命名查询其 SMILES 结构式、"
        "StdInChIKey、ChemSpiderID、分子量等化学信息时，必须调用此工具。"
        "输入：一个化学物质名称字符串。"
        "返回：一个包含该物质各项化学属性与标识符的 JSON 字典。"
        "如果未找到结果，将返回相应的提示信息。"
    )
    args_schema: Type[BaseModel] = ChemSpiderInput

    def _run(self,query: str) -> str:
        """执行ChemSPider查询并格式化返回结果"""
        try:
            data = search_compound_by_name(query)
            #未查询到数据，返回提示信息
            if data is None:
                return (
                    f"在 ChemSpider 中未找到名为 '{query}' 的化合物。"
                    f"请检查拼写，或尝试使用更标准的 IUPAC 名称。"
                )
            #查询到数据，先后执行缓存、传给LLM操作
            print(f"即将存入记忆库，query={query}, data keys={list(data.keys())}")
            add_to_memory(query, data)
            print("存入成功")
            return json.dumps(data,indent=2,ensure_ascii=False)
        except requests.exceptions.HTTPError as e:
            return  f"API 请求失败 (HTTP 错误): {str(e)}"
        except Exception as e:
            return  f"查询过程中发生未预期的错误: {str(e)}"




