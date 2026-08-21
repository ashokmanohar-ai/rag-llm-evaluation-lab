from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rag_eval.cli.main import app

runner = CliRunner()


def test_cli_query_returns_cited_json() -> None:
    result = runner.invoke(app, ["query", "How long is a temporary password valid?"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "30 minutes" in payload["answer"]
    assert payload["citations"] == ["IT-PASSWORD-RESET-C001"]


def test_cli_index_build_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "index.json"
    result = runner.invoke(app, ["index", "build", "--output", str(output)])
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["chunk_count"] >= 20
