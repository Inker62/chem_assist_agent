"""PubChem 检索工具测试——SMILES→名称解析，mock HTTP 避免真实 API 调用"""
import json
import pytest
from unittest.mock import patch, MagicMock


MOCK_ASPIRIN_RESPONSE = {
    "PropertyTable": {
        "Properties": [{
            "CID": 2244,
            "Title": "Aspirin",
            "IUPACName": "2-acetyloxybenzoic acid",
            "MolecularFormula": "C9H8O4",
            "MolecularWeight": "180.16",
            "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "CanonicalSMILES": "CC(=O)Oc1ccccc1C(=O)O"
        }]
    }
}

MOCK_SYNONYMS_RESPONSE = {
    "InformationList": {
        "Information": [{
            "CID": 2244,
            "Synonym": ["Aspirin", "Acetylsalicylic acid", "2-Acetoxybenzoic acid", "阿司匹林"]
        }]
    }
}


class TestPubChemBySmiles:
    def test_success_returns_compound_data(self):
        with patch("tools.pubchem_search.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = MOCK_ASPIRIN_RESPONSE
            mock_get.return_value = mock_resp

            from tools.pubchem_search import search_pubchem_by_smiles
            result = search_pubchem_by_smiles("CC(=O)Oc1ccccc1C(=O)O")
            assert result["Title"] == "Aspirin"
            assert result["MolecularFormula"] == "C9H8O4"
            assert result["CID"] == 2244

    def test_404_returns_none(self):
        with patch("tools.pubchem_search.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_get.return_value = mock_resp

            from tools.pubchem_search import search_pubchem_by_smiles
            result = search_pubchem_by_smiles("C1=INVALID")
            assert result is None

    def test_network_error_returns_none(self):
        import requests as rq
        with patch("tools.pubchem_search.requests.get") as mock_get:
            mock_get.side_effect = rq.RequestException("Connection timeout")

            from tools.pubchem_search import search_pubchem_by_smiles
            result = search_pubchem_by_smiles("CCO")
            assert result is None

    def test_empty_properties_returns_none(self):
        with patch("tools.pubchem_search.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"PropertyTable": {"Properties": []}}
            mock_get.return_value = mock_resp

            from tools.pubchem_search import search_pubchem_by_smiles
            result = search_pubchem_by_smiles("CCO")
            assert result is None


class TestPubChemSynonyms:
    def test_returns_synonym_list(self):
        with patch("tools.pubchem_search.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = MOCK_SYNONYMS_RESPONSE
            mock_get.return_value = mock_resp

            from tools.pubchem_search import search_pubchem_synonyms
            result = search_pubchem_synonyms("CC(=O)Oc1ccccc1C(=O)O")
            assert "Aspirin" in result
            assert "阿司匹林" in result

    def test_404_returns_empty_list(self):
        with patch("tools.pubchem_search.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_get.return_value = mock_resp

            from tools.pubchem_search import search_pubchem_synonyms
            result = search_pubchem_synonyms("INVALID")
            assert result == []


class TestPubChemTool:
    def test_run_returns_json_with_name(self, reset_memory):
        with patch("tools.pubchem_search.search_pubchem_by_smiles") as mock_s:
            mock_s.return_value = MOCK_ASPIRIN_RESPONSE["PropertyTable"]["Properties"][0]
            with patch("tools.pubchem_search.search_pubchem_synonyms") as mock_y:
                mock_y.return_value = ["Aspirin", "阿司匹林"]

                from tools.pubchem_search import PubChemTool
                tool = PubChemTool()
                result = tool._run("CC(=O)Oc1ccccc1C(=O)O")

                data = json.loads(result)
                assert data["Title"] == "Aspirin"
                assert "阿司匹林" in data["Synonyms"]

    def test_run_not_found_returns_error_json(self, reset_memory):
        with patch("tools.pubchem_search.search_pubchem_by_smiles") as mock_s:
            mock_s.return_value = None

            from tools.pubchem_search import PubChemTool
            tool = PubChemTool()
            result = tool._run("INVALID_SMILES")

            data = json.loads(result)
            assert data["status"] == "not_found"

    def test_tool_metadata_is_correct(self):
        from tools.pubchem_search import PubChemTool
        tool = PubChemTool()
        assert tool.name == "PubChemSearch"
        assert "SMILES" in tool.description
        assert "名称" in tool.description
