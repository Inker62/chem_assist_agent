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

def add_to_memory(query:str, response_josn:dict):
    """将查询词和API返回的JSON存入向量库"""
    doc_str = json.dumps(response_josn, ensure_ascii=False) #将python字典序列化为JSON字符串，保证非ASCII字符不被转义
    embedding = embedder.encode(query).tolist() #嵌入模型转化用户查询文本为向量，tolist方法将其转化为python列表
    collection.upsert(
        documents=[doc_str], #API返回的JSON字符串
        embeddings=[embedding], #对应查询向量
        metadatas=[{"query": query}], #元数据，对应至用户原始查询词
        ids=[query] #唯一标识符
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