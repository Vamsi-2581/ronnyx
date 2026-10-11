from unittest.mock import MagicMock, patch

from ronnyx.cli import chat_loop


def run_tools_command(capsys, payload):
    response = MagicMock()
    response.json.return_value = payload
    with (
        patch("ronnyx.cli.requests.get", return_value=response) as get,
        patch("builtins.input", side_effect=["/tools", "exit"]),
    ):
        chat_loop("http://localhost:8000/api/chat", "1")
    get.assert_called_once_with("http://localhost:8000/api/tools", timeout=30)
    return capsys.readouterr().out


class TestToolsCommand:
    def test_lists_tool_names(self, capsys):
        out = run_tools_command(capsys, {"tools": ["search_repositories", "list_issues"]})
        assert " - search_repositories" in out
        assert " - list_issues" in out
        assert " - tools" not in out

    def test_no_tools_loaded(self, capsys):
        out = run_tools_command(capsys, {"tools": []})
        assert "No tools loaded." in out
