"""RDKit 分子计算与结构标注工具测试——descriptors + annotation + StructureAnnotationTool"""
import json
import pytest


class TestComputeDescriptors:
    def test_aspirin_descriptors(self):
        from tools.chem_calc import compute_descriptors
        result = compute_descriptors("CC(=O)Oc1ccccc1C(=O)O")
        assert result["molecular_weight"] == 180.16
        assert result["formula"] == "C9H8O4"
        assert result["num_atoms"] == 13  # 重原子：9C + 4O
        assert result["num_rings"] == 1

    def test_invalid_smiles_returns_error(self):
        from tools.chem_calc import compute_descriptors
        result = compute_descriptors("INVALID")
        assert "error" in result


class TestAnnotateStructure:
    def test_aspirin_no_chiral_centers(self):
        from tools.chem_calc import annotate_structure
        result = annotate_structure("CC(=O)Oc1ccccc1C(=O)O")
        assert "chiral_centers" not in result
        names = [g["name"] for g in result.get("functional_groups", [])]
        assert "羧酸" in names       # 羧酸
        assert "酯基" in names       # 酯基
        assert "苯环" in names       # 苯环
        assert result["murcko_scaffold"] == "c1ccccc1"

    def test_lactic_acid_chiral_center(self):
        from tools.chem_calc import annotate_structure
        result = annotate_structure("C[C@H](O)C(=O)O")
        assert len(result["chiral_centers"]) == 1
        c = result["chiral_centers"][0]
        assert c["stereo"] == "S"
        assert c["symbol"] == "C"

    def test_ez_isomer_detection(self):
        from tools.chem_calc import annotate_structure
        result = annotate_structure("C/C=C/C")
        assert "ez_isomers" in result
        assert result["ez_isomers"][0]["type"].startswith("E")

    def test_acetaminophen_amide_and_phenol(self):
        from tools.chem_calc import annotate_structure
        result = annotate_structure("CC(=O)Nc1ccc(O)cc1")
        names = [g["name"] for g in result.get("functional_groups", [])]
        assert "酰胺" in names       # 酰胺
        assert "酚羟基" in names  # 酚羟基

    def test_invalid_smiles_returns_error(self):
        from tools.chem_calc import annotate_structure
        result = annotate_structure("GARBAGE")
        assert "error" in result

    def test_reactive_sites_detection(self):
        from tools.chem_calc import annotate_structure
        result = annotate_structure("CC(=O)O")
        sites = result.get("reactive_sites", [])
        assert len(sites) >= 1
        assert any("亲电" in s["type"] for s in sites)  # 亲电


class TestStructureAnnotationTool:
    def test_run_returns_json_with_annotation(self):
        from tools.chem_calc import StructureAnnotationTool
        tool = StructureAnnotationTool()
        output = tool._run("C[C@H](O)C(=O)O")
        data = json.loads(output)
        assert "chiral_centers" in data
        assert data["chiral_centers"][0]["stereo"] == "S"

    def test_run_invalid_smiles_returns_error_json(self):
        from tools.chem_calc import StructureAnnotationTool
        tool = StructureAnnotationTool()
        output = tool._run("GARBAGE")
        data = json.loads(output)
        assert "error" in data

    def test_tool_metadata_is_correct(self):
        from tools.chem_calc import StructureAnnotationTool
        tool = StructureAnnotationTool()
        assert tool.name == "StructureAnnotator"
        assert "手性" in tool.description
        assert "官能团" in tool.description
