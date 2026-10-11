import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ronnyx import index
from ronnyx.config import RonnyxConfig
from ronnyx.knowledge import load_knowledge_tools


class TestKnowledgeConfig:
    def test_disabled_by_default(self, tmp_path):
        cfg = tmp_path / "ronnyx.yaml"
        cfg.write_text("llm:\n  api_key: k\n")
        assert RonnyxConfig(path=str(cfg)).knowledge is None

    def test_defaults(self, tmp_path):
        cfg = tmp_path / "ronnyx.yaml"
        cfg.write_text("llm:\n  api_key: k\nknowledge:\n  provider: graphrag\n")
        assert RonnyxConfig(path=str(cfg)).knowledge == {
            "provider": "graphrag",
            "root": "knowledge",
        }

    def test_custom_root(self, tmp_path):
        cfg = tmp_path / "ronnyx.yaml"
        cfg.write_text("llm:\n  api_key: k\nknowledge:\n  root: ./docs\n")
        assert RonnyxConfig(path=str(cfg)).knowledge["root"] == "./docs"

    def test_unknown_provider_raises(self, tmp_path):
        cfg = tmp_path / "ronnyx.yaml"
        cfg.write_text("llm:\n  api_key: k\nknowledge:\n  provider: other\n")
        with pytest.raises(ValueError, match="Unsupported knowledge provider"):
            RonnyxConfig(path=str(cfg))


class TestLoadKnowledgeTools:
    def test_returns_loaded_tools(self):
        with patch("ronnyx.knowledge.load_tools", AsyncMock(return_value=["tool"])):
            assert asyncio.run(load_knowledge_tools({"root": "kb"}, "k")) == ["tool"]

    def test_missing_graphrag_returns_empty(self):
        with patch("ronnyx.knowledge.load_tools", AsyncMock(side_effect=ImportError)):
            assert asyncio.run(load_knowledge_tools({"root": "kb"}, "k")) == []

    def test_missing_index_returns_empty(self):
        with patch(
            "ronnyx.knowledge.load_tools",
            AsyncMock(side_effect=FileNotFoundError("no index")),
        ):
            assert asyncio.run(load_knowledge_tools({"root": "kb"}, "k")) == []


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ronnyx.yaml").write_text(
        "llm:\n  api_key: k\nknowledge:\n  root: kb\n"
    )
    return tmp_path


class TestIndexCommand:
    def test_requires_knowledge_section(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ronnyx.yaml").write_text("llm:\n  api_key: k\n")
        assert index.main([]) == 1
        assert "no 'knowledge' section" in capsys.readouterr().out

    def test_init_creates_project(self, project):
        with patch("ronnyx.knowledge.init_project") as init:
            assert index.main(["--init"]) == 0
        init.assert_called_once()

    def test_requires_init_before_indexing(self, project, capsys):
        assert index.main(["--yes"]) == 1
        assert "ronnyx-index --init" in capsys.readouterr().out

    def test_aborts_without_confirmation(self, project):
        (project / "kb").mkdir()
        (project / "kb" / "settings.yaml").write_text("")
        with (
            patch("builtins.input", return_value="n"),
            patch("ronnyx.knowledge.build_index", AsyncMock()) as build,
        ):
            assert index.main([]) == 1
        build.assert_not_called()

    def test_builds_index(self, project):
        (project / "kb").mkdir()
        (project / "kb" / "settings.yaml").write_text("")
        with patch(
            "ronnyx.knowledge.build_index", AsyncMock(return_value=True)
        ) as build:
            assert index.main(["--yes"]) == 0
        build.assert_awaited_once()
