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

    def test_returns_none_when_collection_empty(self, reset_memory):
        from tools.chem_memory import search_memory
        assert search_memory("aspirin") is None

    def test_returns_json_string_when_found(self, reset_memory, sample_compound_json):
        from tools.chem_memory import add_to_memory, search_memory
        add_to_memory("aspirin", sample_compound_json)
        result = search_memory("aspirin")
        assert result is not None
        parsed = json.loads(result)
        assert parsed["id"] == 2157


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
        assert "2157" in result
        parsed = json.loads(result)
        assert parsed["commonName"] == "Aspirin"

    def test_run_returns_not_found_when_missing(self, reset_memory):
        from tools.chem_memory import ChemicalMemoryTool
        tool = ChemicalMemoryTool()
        result = tool._run("nonexistent_compound")
        assert result == "Not Found"

    def test_tool_metadata_is_correct(self, reset_memory):
        from tools.chem_memory import ChemicalMemoryTool
        tool = ChemicalMemoryTool()
        assert tool.name == "ChemicalMemory"
        assert "ChemSpider" in tool.description
        assert tool.args_schema is not None
