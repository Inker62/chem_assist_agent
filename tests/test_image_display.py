"""extract_mol_images 函数测试——验证 ToolMessage 图片路径提取"""
import os
import json
import tempfile
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class TestExtractMolImages:
    @pytest.fixture
    def real_image_path(self, tmp_path):
        """在临时目录创建一个真实图片文件"""
        img = tmp_path / "test_molecule.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake png content")
        return str(img)

    def test_extracts_valid_image_from_tool_message(self, real_image_path):
        from tools.chem_calc import extract_mol_images
        data = json.dumps({"image_path": real_image_path, "smiles": "CCO"})
        messages = [ToolMessage(content=data, tool_call_id="t1")]
        result = extract_mol_images(messages)
        assert len(result) == 1
        assert result[0][0] == real_image_path
        assert result[0][1] == "CCO"

    def test_skips_nonexistent_image_path(self, tmp_path):
        from tools.chem_calc import extract_mol_images
        bad_path = str(tmp_path / "does_not_exist.png")
        data = json.dumps({"image_path": bad_path, "smiles": "CCO"})
        messages = [ToolMessage(content=data, tool_call_id="t1")]
        result = extract_mol_images(messages)
        assert len(result) == 0

    def test_ignores_non_chemcalc_messages(self, real_image_path):
        from tools.chem_calc import extract_mol_images
        messages = [
            HumanMessage(content="查询 aspirin"),
            AIMessage(content="查到结果了"),
            ToolMessage(content="Not Found", tool_call_id="memory"),
            # 这张是有 image_path 的
            ToolMessage(content=json.dumps({"image_path": real_image_path, "smiles": "CCO"}), tool_call_id="calc"),
        ]
        result = extract_mol_images(messages)
        assert len(result) == 1
        assert result[0][0] == real_image_path

    def test_handles_invalid_json_gracefully(self):
        from tools.chem_calc import extract_mol_images
        messages = [
            ToolMessage(content="this is not valid JSON at all", tool_call_id="bad"),
            ToolMessage(content=json.dumps({"some_field": "no image_path here"}), tool_call_id="ok"),
        ]
        result = extract_mol_images(messages)
        assert len(result) == 0
