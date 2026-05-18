"""ChemSpider 工具测试"""
import json
import pytest
import requests
from unittest.mock import MagicMock


class TestSearchCompoundByName:
    """search_compound_by_name 函数——4 步 RSC API 调用流程"""

    def test_success_returns_compound_data(self, mocker, sample_compound_json):
        """模拟完整的 4 步 HTTP 调用成功流程"""
        mocker.patch('time.sleep')

        # Step 1: POST /filter/name → queryId
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"queryId": "test-qid"}
        mock_post_resp.raise_for_status = MagicMock()
        mocker.patch('tools.chemspider_search.requests.post', return_value=mock_post_resp)

        # Step 2+3: GET /status (Complete) + GET /results
        status_side_effects = []
        # 这两次调用都返回 Complete 状态
        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "Complete"}
        mock_status_resp.raise_for_status = MagicMock()

        mock_results_resp = MagicMock()
        mock_results_resp.json.return_value = {"results": [2157]}
        mock_results_resp.raise_for_status = MagicMock()

        mock_detail_resp = MagicMock()
        mock_detail_resp.json.return_value = sample_compound_json
        mock_detail_resp.raise_for_status = MagicMock()

        mocker.patch('tools.chemspider_search.requests.get',
                       side_effect=[mock_status_resp, mock_results_resp, mock_detail_resp])

        from tools.chemspider_search import search_compound_by_name
        result = search_compound_by_name("aspirin")
        assert result is not None
        assert result["id"] == 2157
        assert result["commonName"] == "Aspirin"

    def test_empty_results_returns_none(self, mocker):
        """API 返回空结果列表时返回 None"""
        mocker.patch('time.sleep')

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"queryId": "test-qid"}
        mock_post_resp.raise_for_status = MagicMock()
        mocker.patch('tools.chemspider_search.requests.post', return_value=mock_post_resp)

        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "Complete"}
        mock_status_resp.raise_for_status = MagicMock()

        mock_results_resp = MagicMock()
        mock_results_resp.json.return_value = {"results": []}  # 空结果
        mock_results_resp.raise_for_status = MagicMock()

        mocker.patch('tools.chemspider_search.requests.get',
                       side_effect=[mock_status_resp, mock_results_resp])

        from tools.chemspider_search import search_compound_by_name
        result = search_compound_by_name("nonexistent")
        assert result is None

    def test_query_id_missing_returns_none(self, mocker):
        """POST 响应中没有 queryId 时返回 None"""
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {}  # 无 queryId
        mock_post_resp.raise_for_status = MagicMock()
        mocker.patch('tools.chemspider_search.requests.post', return_value=mock_post_resp)

        from tools.chemspider_search import search_compound_by_name
        result = search_compound_by_name("aspirin")
        assert result is None

    def test_status_timeout_returns_none(self, mocker):
        """状态一直不是 Complete（超时 15 次重试后）返回 None"""
        mocker.patch('time.sleep')

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"queryId": "test-qid"}
        mock_post_resp.raise_for_status = MagicMock()
        mocker.patch('tools.chemspider_search.requests.post', return_value=mock_post_resp)

        # 始终返回非 Complete 状态
        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "Running"}
        mock_status_resp.raise_for_status = MagicMock()
        mocker.patch('tools.chemspider_search.requests.get', return_value=mock_status_resp)

        from tools.chemspider_search import search_compound_by_name
        result = search_compound_by_name("aspirin")
        assert result is None

    def test_http_error_propagates(self, mocker):
        """HTTP 错误抛出异常"""
        mock_post_resp = MagicMock()
        mock_post_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("502 Bad Gateway")
        mocker.patch('tools.chemspider_search.requests.post', return_value=mock_post_resp)

        from tools.chemspider_search import search_compound_by_name
        with pytest.raises(requests.exceptions.HTTPError):
            search_compound_by_name("aspirin")


class TestChemSpiderTool:
    """LangChain ChemSpiderTool 包装器"""

    def test_run_success(self, mocker, sample_compound_json):
        """工具 _run 成功返回 JSON 字符串，并调用 add_to_memory"""
        mocker.patch('tools.chemspider_search.search_compound_by_name', return_value=sample_compound_json)
        mock_add = mocker.patch('tools.chemspider_search.add_to_memory')

        from tools.chemspider_search import ChemSpiderTool
        tool = ChemSpiderTool()
        result = tool._run("aspirin")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["id"] == 2157

        mock_add.assert_called_once_with("aspirin", sample_compound_json)

    def test_run_not_found(self, mocker):
        """未找到化合物时返回友好提示"""
        mocker.patch('tools.chemspider_search.search_compound_by_name', return_value=None)

        from tools.chemspider_search import ChemSpiderTool
        tool = ChemSpiderTool()
        result = tool._run("nonexistent")

        assert "未找到" in result
        assert "nonexistent" in result

    def test_run_http_error(self, mocker):
        """HTTP 异常时返回错误信息"""
        mocker.patch(
            'tools.chemspider_search.search_compound_by_name',
            side_effect=requests.exceptions.HTTPError("500 Server Error")
        )

        from tools.chemspider_search import ChemSpiderTool
        tool = ChemSpiderTool()
        result = tool._run("aspirin")

        assert "API 请求失败" in result
        assert "500" in result

    def test_run_unexpected_error(self, mocker):
        """意外异常时返回错误信息"""
        mocker.patch(
            'tools.chemspider_search.search_compound_by_name',
            side_effect=RuntimeError("Something unexpected")
        )

        from tools.chemspider_search import ChemSpiderTool
        tool = ChemSpiderTool()
        result = tool._run("aspirin")

        assert "未预期的错误" in result

    def test_tool_metadata_is_correct(self):
        from tools.chemspider_search import ChemSpiderTool
        tool = ChemSpiderTool()
        assert tool.name == "ChemSpiderSearch"
        assert "SMILES" in tool.description
        assert tool.args_schema is not None
