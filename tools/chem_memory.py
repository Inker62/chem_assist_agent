import os
import json
from typing import Optional, Type
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "local_data")
CACHE_DIR = os.path.join(DATA_DIR, "model_cache") #持久存储向量库
PERSIST_DIR = os.path.join(DATA_DIR, "chemical_memory_db") #模型缓存地址
os.makedirs(CACHE_DIR, exist_ok=True)  # 确保目录存在
#加载'all-MiniLM-L6-v2'本地模型，轻量化，生产384维向量
embedder = SentenceTransformer('all-MiniLM-L6-v2',cache_folder=CACHE_DIR)
#初始化chroma数据库实例，在磁盘上持久化一个客户端，存入指定路径
client = chromadb.PersistentClient(path=PERSIST_DIR)

collection = client.get_or_create_collection(
    name='chemical_cache',
    metadata={"hnsw:space":"cosine"}
) #从客户端读取/若无则新建'chemical_cache'的集合，并指定向量检索相似度为余弦距离

def add_to_memory(query:str, response_json:dict):
    """将查询词和 API 返回的 JSON 存入向量库，使用规范名称或 ChemSpider ID 作为键"""
    # 使用 ChemSpider 返回的通用名或 ID 作为存储键，避免同义词重复存储
    compound_id = response_json.get("id")
    common_name = response_json.get("commonName", query).strip()

    # 优先使用 ChemSpider ID 作为唯一键，如果没有则回退到通用名
    if compound_id:
        store_key = f"csid_{compound_id}"
    else:
        # 简单标准化：去除多余空格、转为小写
        store_key = common_name.lower().strip()

    doc_str = json.dumps(response_json, ensure_ascii=False)
    embedding = embedder.encode(common_name).tolist()  # 使用通用名生成向量，方便语义检索

    # 先检查是否已存在该键，存在则更新（合并信息），否则新增
    existing = collection.get(ids=[store_key])
    if existing and existing["ids"]:
        print(f"记忆库已存在记录 {store_key}，将更新。")
    else:
        print(f"记忆库新增记录 {store_key}。")

    collection.upsert(
        documents=[doc_str],
        embeddings=[embedding],
        metadatas=[{"query": query, "commonName": common_name}],
        ids=[store_key]
    )

def search_memory(query: str, top_k: int = 1) -> Optional[str]: #top_k为1，返回最相似的缓存JSON字符串
    """从向量库中检索最相似的化学数据"""
    if collection.count() == 0: #判断是否有缓存
        return None
    embedding = embedder.encode(query).tolist() #用户“再次输入”的查询
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    ) # 返回字典，包括documents、embeddings、metadatas、ids四个键，每个键对应二维列表值
    if results["documents"] and results["documents"][0]: #检查非空
        return results["documents"][0][0]
    return None

def clear_memory():
    """清空向量记忆库中的所有数据"""
    global collection
    # 获取所有记录的 ID 列表
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)


class ChemicalMemoryInput(BaseModel):
    query: str = Field(description="将要在历史缓存中搜索的化学物质名称或描述")

class ChemicalMemoryTool(BaseTool):
    name: str = "ChemicalMemory"
    description: str = (
        "在调用ChemSpider API或ChemSpiderTool工具进行查询前，务必先使用此工具在历史缓存中搜索"
        "输入：一个化学物质名称字符串。"
        "返回：若在历史缓存中命中，则返回完整化学物质信息JSON；否则返回'Not Found'"
    )
    args_schema: Type[BaseModel] = ChemicalMemoryInput

    def _run(self, query:str) -> str:
        result = search_memory(query)
        return result if result else "Not Found"

    def _arun(self, query:str):
        raise NotImplementedError