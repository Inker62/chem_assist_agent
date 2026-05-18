"""CrossRef 文献检索工具测试"""
import json
import pytest
import requests
from unittest.mock import MagicMock


class TestCrossrefSearchTool:
    """LangChain CrossrefSearchTool 包装器"""

    def test_run_returns_literature_list(self, mocker, sample_crossref_response):
        """成功返回文献 JSON 列表"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_crossref_response
        mock_resp.raise_for_status = MagicMock()
        mock_get = mocker.patch('tools.literature_search.requests.get', return_value=mock_resp)

        from tools.literature_search import CrossrefSearchTool
        tool = CrossrefSearchTool()
        result = tool._run("AIE synthesis")

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Synthesis of AIE Fluorescent Molecules"
        assert parsed[0]["doi"] == "10.1234/test.2023.001"

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["params"]["query"] == "AIE synthesis"
        assert call_kwargs["params"]["rows"] == 5

    def test_run_empty_results_returns_empty_list(self, mocker):
        """无结果时返回空 JSON 数组"""
        empty_response = {"message": {"items": []}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = empty_response
        mock_resp.raise_for_status = MagicMock()
        mocker.patch('tools.literature_search.requests.get', return_value=mock_resp)

        from tools.literature_search import CrossrefSearchTool
        tool = CrossrefSearchTool()
        result = tool._run("xyzzy_nonexistent")

        parsed = json.loads(result)
        assert parsed == []

    def test_run_respects_max_results_parameter(self, mocker):
        """max_results 参数正确传递给 API"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"items": []}}
        mock_resp.raise_for_status = MagicMock()
        mock_get = mocker.patch('tools.literature_search.requests.get', return_value=mock_resp)

        from tools.literature_search import CrossrefSearchTool
        tool = CrossrefSearchTool()
        tool._run("test", max_results=10)

        assert mock_get.call_args.kwargs["params"]["rows"] == 10

    def test_run_network_error_returns_error_string(self, mocker):
        """网络错误时返回错误信息而非抛出异常"""
        mocker.patch(
            'tools.literature_search.requests.get',
            side_effect=requests.exceptions.ConnectionError("Network unreachable")
        )

        from tools.literature_search import CrossrefSearchTool
        tool = CrossrefSearchTool()
        result = tool._run("test")

        assert "检索失败" in result

    def test_run_http_error_returns_error_string(self, mocker):
        """HTTP 错误时返回错误信息"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("503 Service Unavailable")
        mocker.patch('tools.literature_search.requests.get', return_value=mock_resp)

        from tools.literature_search import CrossrefSearchTool
        tool = CrossrefSearchTool()
        result = tool._run("test")

        assert "检索失败" in result

    def test_run_missing_fields_uses_defaults(self, mocker):
        """缺失字段时使用 N/A 作为默认值"""
        minimal_response = {
            "message": {
                "items": [
                    {
                        "title": ["Bare Minimum Paper"],
                        "author": [],
                        "container-title": [],
                        "issued": {},
                        "DOI": None,
                        "is-referenced-by-count": 0,
                    }
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = minimal_response
        mock_resp.raise_for_status = MagicMock()
        mocker.patch('tools.literature_search.requests.get', return_value=mock_resp)

        from tools.literature_search import CrossrefSearchTool
        tool = CrossrefSearchTool()
        result = tool._run("test")

        parsed = json.loads(result)
        assert parsed[0]["title"] == "Bare Minimum Paper"
        assert parsed[0]["authors"] == []  # 空作者列表保持不变
        assert parsed[0]["journal"] == "N/A"
        assert parsed[0]["year"] == 0
        assert parsed[0]["doi"] == "N/A"

    def test_tool_metadata_is_correct(self):
        from tools.literature_search import CrossrefSearchTool
        tool = CrossrefSearchTool()
        assert tool.name == "CrossrefSearch"
        assert "文献" in tool.description
        assert tool.args_schema is not None
