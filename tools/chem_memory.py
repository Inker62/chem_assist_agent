"""化学知识库——基于 ChromaDB 的持久化分子记忆系统，支持语义检索、生命周期管理和 RAG

每次检索都存入知识库，Agent 可从知识库中获取历史上下文，实现"个人化学知识库"。

核心功能:
- 语义向量检索 (cosine similarity) + top-k 候选 + 相似度门槛
- 每条记录带时间戳，支持生命周期管理
- 前端可浏览、搜索、删除单条记录
"""
import os
import json
import time
import logging
from typing import Optional, Type
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "local_data")
CACHE_DIR = os.path.join(DATA_DIR, "model_cache")
PERSIST_DIR = os.path.join(DATA_DIR, "chemical_memory_db")
os.makedirs(CACHE_DIR, exist_ok=True)

embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=CACHE_DIR, local_files_only=True)
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(
    name='chemical_cache',
    metadata={"hnsw:space": "cosine"}
)

# 默认检索参数
DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.4
MAX_ENTRIES = 500  # 知识库容量上限，超出后 LRU 淘汰


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def add_to_memory(query: str, response_json: dict):
    """将查询词和 API 返回的 JSON 存入向量知识库，自动去重"""
    compound_id = response_json.get("id")
    common_name = response_json.get("commonName", query).strip()

    if compound_id:
        store_key = f"csid_{compound_id}"
    else:
        store_key = common_name.lower().strip()

    doc_str = json.dumps(response_json, ensure_ascii=False)
    embedding = embedder.encode(common_name).tolist()

    existing = collection.get(ids=[store_key])
    now = _now_iso()
    if existing and existing["ids"]:
        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
        created_at = old_meta.get("created_at", now)
        access_count = old_meta.get("access_count", 0)
        logger.info("知识库更新记录 %s", store_key)
    else:
        created_at = now
        access_count = 0
        logger.info("知识库新增记录 %s", store_key)

    collection.upsert(
        documents=[doc_str],
        embeddings=[embedding],
        metadatas=[{
            "query": query,
            "commonName": common_name,
            "created_at": created_at,
            "last_accessed": now,
            "access_count": access_count,
        }],
        ids=[store_key]
    )

    # LRU 淘汰：超出容量上限时删除最久未访问的条目
    _enforce_capacity()


def search_memory(query: str, top_k: int = DEFAULT_TOP_K,
                  threshold: float = DEFAULT_THRESHOLD) -> list:
    """语义检索知识库，返回 [(doc_json, score, metadata), ...]，按相似度降序

    只返回 similarity >= threshold 的结果。
    """
    if collection.count() == 0:
        return []

    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k * 2, collection.count())  # 多取一些再过滤
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    hits = []
    for i, doc in enumerate(results["documents"][0]):
        # ChromaDB cosine distance -> similarity = 1 - distance
        if results.get("distances") and i < len(results["distances"][0]):
            distance = results["distances"][0][i]
        else:
            distance = 0
        score = 1.0 - distance
        if score >= threshold:
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            hits.append((doc, round(score, 4), meta))

    # 按分数降序排列，取 top_k
    hits.sort(key=lambda x: x[1], reverse=True)
    hits = hits[:top_k]

    # 更新访问时间
    for doc, score, meta in hits:
        entry_id = meta.get("query", "")
        if entry_id:
            _touch(meta)

    return hits


def _touch(meta: dict):
    """更新条目的 last_accessed 和 access_count（通过 upsert 重新写入）"""
    store_key = None
    common_name = meta.get("commonName", "")
    existing = collection.get()
    for i, md in enumerate(existing.get("metadatas", [])):
        if md and md.get("commonName") == common_name:
            store_key = existing["ids"][i]
            break
    if store_key is None:
        return
    meta["last_accessed"] = _now_iso()
    meta["access_count"] = meta.get("access_count", 0) + 1
    collection.update(
        ids=[store_key],
        metadatas=[meta]
    )


def _enforce_capacity():
    """当条目数超过 MAX_ENTRIES 时，删除最久未访问的条目（LRU）"""
    all_data = collection.get()
    if len(all_data.get("ids", [])) <= MAX_ENTRIES:
        return

    entries = []
    for i, eid in enumerate(all_data["ids"]):
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        entries.append((eid, meta.get("last_accessed", "1970-01-01")))
    entries.sort(key=lambda x: x[1])  # 最旧的排前面

    to_remove = len(entries) - MAX_ENTRIES
    for eid, _ in entries[:to_remove]:
        collection.delete(ids=[eid])
        logger.info("LRU 淘汰: %s", eid)


def list_entries() -> list:
    """列出知识库中所有条目，供前端浏览。返回 [{id, query, commonName, created_at, last_accessed, access_count}, ...]"""
    all_data = collection.get()
    entries = []
    for i, eid in enumerate(all_data.get("ids", [])):
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        doc_str = all_data["documents"][i] if all_data["documents"] else "{}"
        try:
            doc = json.loads(doc_str)
        except (json.JSONDecodeError, TypeError):
            doc = {}
        entries.append({
            "id": eid,
            "query": meta.get("query", ""),
            "commonName": meta.get("commonName", ""),
            "smiles": doc.get("smiles", doc.get("SMILES", "")),
            "formula": doc.get("molecularFormula", doc.get("formula", "")),
            "molecular_weight": doc.get("molecularWeight", doc.get("molecular_weight", "")),
            "created_at": meta.get("created_at", ""),
            "last_accessed": meta.get("last_accessed", ""),
            "access_count": meta.get("access_count", 0),
        })
    entries.sort(key=lambda x: x["last_accessed"], reverse=True)
    return entries


def delete_entry(entry_id: str):
    """删除知识库中指定 ID 的条目"""
    collection.delete(ids=[entry_id])
    logger.info("知识库删除记录: %s", entry_id)


def get_stats() -> dict:
    """返回知识库统计信息"""
    all_data = collection.get()
    count = len(all_data.get("ids", []))
    metas = all_data.get("metadatas", []) if all_data.get("metadatas") else []

    if not metas:
        return {"total": 0, "oldest": None, "newest": None, "total_accesses": 0}

    times = [m.get("created_at", "") for m in metas if m]
    accesses = sum(m.get("access_count", 0) for m in metas if m)
    return {
        "total": count,
        "oldest": min(times) if times else None,
        "newest": max(times) if times else None,
        "total_accesses": accesses,
        "capacity_pct": round(count / MAX_ENTRIES * 100, 1),
    }


def clear_memory():
    """清空知识库中的所有数据"""
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        logger.info("知识库已清空，共删除 %d 条记录", len(existing_ids))


class ChemicalMemoryInput(BaseModel):
    query: str = Field(description="将要在化学知识库中搜索的物质名称或描述")


class ChemicalMemoryTool(BaseTool):
    """化学知识库检索工具——在调用外部 API 前先查本地知识库"""
    name: str = "ChemicalMemory"
    description: str = (
        "化学知识库——每次查询后自动积累的本地化学数据库。\n"
        "在调用 ChemSpider API 之前，务必先在此知识库中搜索。\n"
        "输入：化学物质名称、SMILES 或描述。\n"
        "返回：匹配的化合物信息 JSON 列表，含相似度评分。\n"
        "若返回空列表或评分低于 0.5，说明知识库中无可靠匹配，应继续调用 ChemSpider。"
    )
    args_schema: Type[BaseModel] = ChemicalMemoryInput

    def _run(self, query: str) -> str:
        hits = search_memory(query)
        if not hits:
            return json.dumps({"status": "miss", "message": "知识库中无匹配记录", "results": []},
                              ensure_ascii=False)
        results = []
        for doc_str, score, meta in hits:
            try:
                doc = json.loads(doc_str)
            except (json.JSONDecodeError, TypeError):
                doc = {"raw": doc_str}
            doc["_similarity"] = score
            doc["_source"] = meta.get("query", "")
            results.append(doc)
        return json.dumps({
            "status": "hit",
            "count": len(results),
            "query": query,
            "results": results
        }, ensure_ascii=False)

    def _arun(self, query: str):
        raise NotImplementedError
