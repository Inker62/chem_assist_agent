"""ChemicalMemory 工具和 ChromaDB 记忆库测试"""
import json
import pytest
from unittest.mock import MagicMock


class TestFakeCollection:
    """验证 conftest.py 中的 FakeCollection 行为正确"""

    def test_collection_starts_empty(self, reset_memory):
        assert reset_memory.count() == 0

    def test_get_returns_nothing_when_empty(self, reset_memory):
        result = reset_memory.get()
        assert result["ids"] == []

    def test_query_returns_empty_when_empty(self, reset_memory):
        result = reset_memory.query(query_embeddings=[[0.1]], n_results=1)
        assert result["documents"] == [[]]

    def test_delete_nonexistent_does_not_raise(self, reset_memory):
        reset_memory.delete(["nonexistent"])
        assert reset_memory.count() == 0


class TestAddToMemory:
    """add_to_memory 函数的各种场景"""

    def test_add_new_compound_with_csid(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory
        add_to_memory("aspirin", sample_compound_json)
        assert reset_memory.count() == 1

        stored = reset_memory.get()
        assert stored["ids"][0] == "csid_2157"

    def test_add_new_compound_without_csid(self, reset_memory):
        from tools.chem_memory import add_to_memory
        data = {"commonName": "UnknownCompound"}  # 没有 id 字段
        add_to_memory("UnknownCompound", data)
        assert reset_memory.count() == 1
        assert reset_memory.get()["ids"][0] == "unknowncompound"

    def test_add_duplicate_updates_existing(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory
        add_to_memory("aspirin", sample_compound_json)
        assert reset_memory.count() == 1

        updated_data = {**sample_compound_json, "molecularWeight": 200.0}
        add_to_memory("aspirin", updated_data)
        assert reset_memory.count() == 1  # 仍然是 1 条

        stored_doc = json.loads(reset_memory.get()["documents"][0])
        assert stored_doc["molecularWeight"] == 200.0


class TestSearchMemory:
    """search_memory 函数的各种场景"""

    def test_returns_empty_list_when_collection_empty(self, reset_memory):
        from tools.chem_memory import search_memory
        assert search_memory("aspirin") == []

    def test_returns_hit_list_when_found(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory, search_memory
        add_to_memory("aspirin", sample_compound_json)
        result = search_memory("aspirin")
        assert len(result) > 0
        doc_str, score, meta = result[0]
        parsed = json.loads(doc_str)
        assert parsed["id"] == 2157
        assert 0 <= score <= 1


class TestClearMemory:
    """clear_memory 函数"""

    def test_clear_removes_all_entries(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory, clear_memory
        add_to_memory("aspirin", sample_compound_json)
        add_to_memory("caffeine", {"id": 2519, "commonName": "Caffeine"})
        assert reset_memory.count() == 2

        clear_memory()
        assert reset_memory.count() == 0


class TestChemicalMemoryTool:
    """LangChain ChemicalMemoryTool 包装器"""

    def test_run_returns_json_when_found(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory, ChemicalMemoryTool
        add_to_memory("aspirin", sample_compound_json)
        tool = ChemicalMemoryTool()
        result = tool._run("aspirin")
        parsed = json.loads(result)
        assert parsed["status"] == "hit"
        assert parsed["count"] >= 1
        assert parsed["results"][0]["commonName"] == "Aspirin"

    def test_run_returns_miss_when_not_found(self, reset_memory):
        from tools.chem_memory import ChemicalMemoryTool
        tool = ChemicalMemoryTool()
        result = tool._run("nonexistent_compound")
        parsed = json.loads(result)
        assert parsed["status"] == "miss"
        assert parsed["results"] == []

    def test_tool_metadata_is_correct(self, reset_memory):
        from tools.chem_memory import ChemicalMemoryTool
        tool = ChemicalMemoryTool()
        assert tool.name == "ChemicalMemory"
        assert "知识库" in tool.description
        assert tool.args_schema is not None


class TestListEntries:
    """知识库浏览器相关"""

    def test_empty_returns_empty_list(self, reset_memory):
        from tools.chem_memory import list_entries
        assert list_entries() == []

    def test_returns_all_entries(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory, list_entries
        add_to_memory("aspirin", sample_compound_json)
        entries = list_entries()
        assert len(entries) == 1
        assert entries[0]["id"] == "csid_2157"
        assert "created_at" in entries[0]
        assert "last_accessed" in entries[0]


class TestDeleteEntry:
    def test_delete_single_entry(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory, delete_entry, list_entries
        add_to_memory("aspirin", sample_compound_json)
        assert len(list_entries()) == 1
        delete_entry("csid_2157")
        assert len(list_entries()) == 0

    def test_delete_nonexistent_does_not_raise(self, reset_memory):
        from tools.chem_memory import delete_entry
        delete_entry("no_such_id")  # 不应抛异常


class TestGetStats:
    def test_empty_stats(self, reset_memory):
        from tools.chem_memory import get_stats
        stats = get_stats()
        assert stats["total"] == 0

    def test_stats_with_entries(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory, get_stats
        add_to_memory("aspirin", sample_compound_json)
        stats = get_stats()
        assert stats["total"] == 1
        assert stats["newest"] is not None
        assert "capacity_pct" in stats


class TestSearchThreshold:
    """相似度阈值过滤"""

    def test_high_threshold_filters_results(self, reset_memory):
        """存入 'aspirin'，设置阈值 0.99 → 即使相关匹配也被过滤"""
        from tools.chem_memory import add_to_memory, search_memory
        add_to_memory("aspirin", {"commonName": "Aspirin", "id": 2157})
        hits = search_memory("aspirin", threshold=0.99)
        assert hits == []  # 极高门槛，伪嵌入向量达不到

    def test_relaxed_threshold_returns_matches(self, reset_memory):
        """存入 'aspirin'，用相同词 + 低阈值搜索 → 应命中"""
        from tools.chem_memory import add_to_memory, search_memory
        add_to_memory("aspirin", {"commonName": "Aspirin", "id": 2157})
        hits = search_memory("aspirin", threshold=0.1)
        assert len(hits) > 0
