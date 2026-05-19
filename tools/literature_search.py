import requests
from typing import Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json

from config import CROSSREF_BASE_URL, CROSSREF_TIMEOUT, PUBMED_EMAIL

class CrossrefSearchInput(BaseModel):
    query: str = Field(description="CrossRef 检索关键词，支持引文标题、作者或 DOI")
    max_results: int = Field(default=5, description="返回的最大文献数量")

class CrossrefSearchTool(BaseTool):
    name: str = "CrossrefSearch"
    description: str = (
        "当用户需要检索学术文献时使用此工具。"
        "输入：中英文关键词,例如：请检索多巴胺合成相关的文献，此时请以【多巴胺】、【合成】为关键词进行检索。"
        "返回：一个包含文献标题、作者、期刊、出版年、DOI 和引用次数的 JSON 列表。"
        "调用此工具后，请直接将返回的文献列表以表格形式呈现给用户，表格列应为：序号、标题、第一作者、期刊、出版年、DOI。"
        "不要再要求用户提供额外的文献条目，也不要写文献综述。"
    )
    args_schema: Type[BaseModel] = CrossrefSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            params = {
                "query": query,
                "rows": max_results,
                "mailto": PUBMED_EMAIL
            }
            resp = requests.get(CROSSREF_BASE_URL, params=params, timeout=CROSSREF_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("message", {}).get("items", []):
                authors = item.get("author", [])
                results.append({
                    "title": (item.get("title") or ["N/A"])[0],
                    "authors": [a.get("family", "N/A") for a in authors],
                    "journal": (item.get("container-title") or ["N/A"])[0] or "N/A",
                    "year": (item.get("issued", {}).get("date-parts") or [[0]])[0][0],
                    "doi": item.get("DOI") or "N/A",
                    "citation_count": item.get("is-referenced-by-count", 0)
                })

            return json.dumps(results, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"CrossRef 检索失败: {str(e)}"




