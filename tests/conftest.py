"""
Pytest configuration for ChemAssist.

CRITICAL ORDERING: 环境变量和 sys.modules 注入必须在模块顶层执行
（在任何测试函数定义之前），原因：
1. 5 个生产文件中存在模块级 load_dotenv() 调用。在这里预先设置环境
   变量后，load_dotenv() 默认不会覆盖已有变量，实际上变为 no-op。
2. tools/chem_memory.py 在模块级导入时会创建 chromadb.PersistentClient
   和 SentenceTransformer 实例。必须在任何源码导入前将 Fake 注入
   sys.modules，防止真实文件系统/数据库/模型下载的副作用。
"""

import os
import sys
import hashlib
from unittest.mock import MagicMock

# ============================================================
# STEP 1: 在源码导入前设置环境变量
# ============================================================
os.environ['DEEPSEEK_API_KEY'] = 'test-deepseek-key'
os.environ['RSC_API_KEY'] = 'test-rsc-key'
os.environ['PUBMED_EMAIL'] = 'test@example.com'
os.environ['CHEMASSIST_TEST_DB'] = os.path.join(
    os.path.dirname(__file__), '..', 'local_data', 'test_checkpoints.db'
)

# ============================================================
# STEP 2: Fake ChromaDB 集合 和 SentenceTransformer
# ============================================================

class FakeEmbedding:
    """模拟 numpy 数组，提供 .tolist() 方法，供 embedder.encode().tolist() 模式使用"""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class FakeEmbedder:
    """确定性假嵌入模型，基于 MD5 哈希产生向量（48 维替代真实 384 维，加快测试）"""

    DIM = 48

    def encode(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        results = []
        for t in text:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            vec = [((h >> i) & 0xFF) / 255.0 for i in range(self.DIM)]
            results.append(vec)
        if len(results) == 1:
            return FakeEmbedding(results[0])
        return [FakeEmbedding(r) for r in results]


class FakeCollection:
    """纯内存字典实现的 ChromaDB 集合，支持 count/get/query/upsert/delete"""

    def __init__(self):
        self._docs = {}
        self._embeddings = {}
        self._metadatas = {}
        self._ids_order = []  # 保持插入顺序

    def count(self) -> int:
        return len(self._docs)

    def get(self, ids=None):
        if ids is not None:
            matched = [i for i in ids if i in self._docs]
            return {
                "ids": matched,
                "documents": [self._docs[i] for i in matched],
                "metadatas": [self._metadatas.get(i) for i in matched],
            }
        return {
            "ids": list(self._ids_order),
            "documents": [self._docs[i] for i in self._ids_order],
            "metadatas": [self._metadatas.get(i) for i in self._ids_order],
        }

    def query(self, query_embeddings, n_results=1):
        if not self._docs:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
        import math
        query_vec = query_embeddings[0]
        scores = []
        for key in self._ids_order:
            stored_vec = self._embeddings.get(key, [0] * len(query_vec))
            dot = sum(a * b for a, b in zip(query_vec, stored_vec))
            norm_q = math.sqrt(sum(a * a for a in query_vec))
            norm_s = math.sqrt(sum(b * b for b in stored_vec))
            if norm_q == 0 or norm_s == 0:
                similarity = 0.0
            else:
                similarity = dot / (norm_q * norm_s)
            scores.append((key, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:n_results]
        return {
            "documents": [[self._docs[k] for k, _ in top]],
            "metadatas": [[self._metadatas.get(k) for k, _ in top]],
            "ids": [[k for k, _ in top]],
            "distances": [[round(1.0 - s, 4) for _, s in top]],
        }

    def upsert(self, documents, embeddings, metadatas, ids):
        for i, id_ in enumerate(ids):
            if id_ not in self._docs:
                self._ids_order.append(id_)
            self._docs[id_] = documents[i]
            self._embeddings[id_] = embeddings[i]
            self._metadatas[id_] = metadatas[i]

    def update(self, ids, metadatas):
        for i, id_ in enumerate(ids):
            if id_ in self._docs:
                self._metadatas[id_] = metadatas[i] if i < len(metadatas) else metadatas[0]

    def delete(self, ids):
        for id_ in ids:
            self._docs.pop(id_, None)
            self._embeddings.pop(id_, None)
            self._metadatas.pop(id_, None)
            if id_ in self._ids_order:
                self._ids_order.remove(id_)


# 构建单例 Fake 实例并注入 sys.modules
_fake_collection = FakeCollection()

_mock_chroma = MagicMock()
_mock_chroma.PersistentClient = MagicMock()
_mock_chroma.PersistentClient.return_value.get_or_create_collection.return_value = (
    _fake_collection
)

_mock_st = MagicMock()
_mock_st.SentenceTransformer = MagicMock(return_value=FakeEmbedder())

sys.modules['chromadb'] = _mock_chroma
sys.modules['sentence_transformers'] = _mock_st

# ============================================================
# STEP 3: Pytest fixtures
# ============================================================
import pytest


@pytest.fixture
def fake_collection():
    """返回 FakeCollection 单例（被 chem_memory 模块使用）"""
    return _fake_collection


@pytest.fixture
def reset_memory(fake_collection):
    """每个测试前清空 ChromaDB 集合，确保测试隔离"""
    fake_collection._docs.clear()
    fake_collection._embeddings.clear()
    fake_collection._metadatas.clear()
    fake_collection._ids_order.clear()
    return fake_collection


@pytest.fixture
def sample_compound_json():
    """模拟 ChemSpider 返回的化合物详情 JSON"""
    return {
        "id": 2157,
        "commonName": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "formula": "C9H8O4",
        "molecularWeight": 180.157,
        "inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
        "inchiKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    }


@pytest.fixture
def sample_crossref_response():
    """模拟 CrossRef API 返回的文献列表"""
    return {
        "message": {
            "items": [
                {
                    "title": ["Synthesis of AIE Fluorescent Molecules"],
                    "author": [{"family": "Wang"}, {"family": "Li"}],
                    "container-title": ["Journal of Organic Chemistry"],
                    "issued": {"date-parts": [[2023]]},
                    "DOI": "10.1234/test.2023.001",
                    "is-referenced-by-count": 15,
                }
            ]
        }
    }
