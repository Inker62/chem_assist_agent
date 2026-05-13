import requests
import os
from typing import List, Type, Optional
from Bio import Entrez
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

load_dotenv()

PUBMED_EMAIL = os.getenv("PUBMED_EMAIL","")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY","")
CROSSREF_BASE_URL = "https://api.crossref.org/works/"
REQUEST_TIMEOUT = 15

Entrez.email = PUBMED_EMAIL
if PUBMED_API_KEY:
    Entrez.api_key = PUBMED_API_KEY

# PubMed Tool
class PubmedSearchInput(BaseModel):
    query: str = Field(description="用于PubMed检索文献的关键词/Keyword，支持Bool逻辑。例如：‘aspirin AND synthesis’")
    max_results: int = Field(default=5, description="返回的最大文献数量")

class PubmedSearchTool(BaseTool):
    name: str = "PubmedSearch"
    description: str = (
        "当用户需要检索化学、化学合成、药物合成、生物医学、药理知识等方面相关文献时使用此工具。"
        "输入：中文关键词或英文Keywords，支持常用Bool逻辑词，例如（AND、OR、NOT）"
        "输出：包含文献标题（title）、作者（authors）、期刊（journal）、发表年份（year）、PubMed编号（pmid）、期号（Volume）、卷号（Issue）、摘要（Abstract）、DOI号的文献信息列表"
    )
    args_schema: Type[BaseModel] = PubmedSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        """执行PubMed检索"""
        try:
            # esearch查找PMIDs
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            pmid_record = Entrez.read(handle)
            handle.close()
            id_list = pmid_record.get("IdList",[])

            if not id_list:
                return "未找到相关文献"

            # efetch批量获取文献XML记录
            handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="null", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            # 解析XML
            results = []
            for article in records:
                medline = article.get("MedlineCitation", {}).get("Article", {})
                pubmed_data = medline.get("PubmedData", {})

                # Authors
                authors = []
                author_list = medline.get("AuthorList",[])
                for author in author_list:
                    last = author.get("LastName","")
                    first = author.get("FirstName","")
                    if last and first:
                        authors.append(f"{first}{last} ".strip())

                # DOI
                doi = ""
                article_ids = pubmed_data.get("ArticleIDList",[])
                for article_id in article_ids:
                    if article_id.attributes.get("IDType") == "doi":
                        doi = str(article_id)
                        break

                # MeSH
                mesh_trems = []
                mesh_headings = article.get("MedlineCitation",{}).get("MeshHeadingList",[])
                for mesh_heading in mesh_headings:
                    descriptor = mesh_heading.get("DescriptorName")
                    if descriptor:
                        mesh_trems.append(str(descriptor))

                # Abstract
                abstract = medline.get("Abstract",{}).get("AbstractText",[""])[0]
                if isinstance(abstract, str):
                    abstract = abstract
                else:
                    abstract = ""

                results.append({
                    "pmid": str(article.get("MedlineCitation",{}).get("PMID","N/A")),
                    "title": str(medline.get("ArticleTitle","N/A")),
                    "authors": authors,
                    "journal": str(medline.get("Journal",{}).get("Title","N/A")),
                    "year": str(medline.get("Journal",{}).get("JournalIssue",{}).get("PubDate",{}).get("Year","N/A")),
                    "abstract": abstract,
                    "doi": doi,
                    "keywords": mesh_trems,
                    "publication_type": [str(pt) for pt in medline.get("PublicationTypeList",[])]
                })

                return  json.dumps(results, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"PubMed 检索失败: {str(e)}"


class CrossrefSearchInput(BaseModel):
    query: str = Field(description="CrossRef 检索关键词，支持引文标题、作者或 DOI")
    max_results: int = Field(default=5, description="返回的最大文献数量")

class CrossrefSearchTool(BaseTool):
    name: str = "CrossrefSearch"
    description: str = (
        "当用户需要检索期刊论文、会议论文等学术资源的简要信息时使用此工具。"
        "输入：中英文关键词、DOI 或作者名。"
        "返回：包含标题、DOI、期刊和引用信息的文献列表。"
    )
    args_schema: Type[BaseModel] = CrossrefSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            params = {
                "query": query,
                "rows": max_results,
                "mailto": PUBMED_EMAIL
            }
            resp = requests.get("https://api.crossref.org/works", params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("message", {}).get("items", []):
                authors = item.get("author", [])
                results.append({
                    "title": item.get("title", ["N/A"])[0],
                    "authors": [a.get("family", "N/A") for a in authors],
                    "journal": item.get("container-title", ["N/A"])[0] or "N/A",
                    "year": item.get("issued", {}).get("date-parts", [[0]])[0][0],
                    "doi": item.get("DOI", "N/A"),
                    "citation_count": item.get("is-referenced-by-count", 0)
                })

            return json.dumps(results, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"CrossRef 检索失败: {str(e)}"




