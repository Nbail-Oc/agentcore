"""Guards: agentcore must stay hospital-free and framework-free."""

import ast
from pathlib import Path

FORBIDDEN = {
    "agent_service", "mock_hospital", "langchain", "langgraph", "crewai", "llama_index",
    "pydantic_ai", "autogen", "agents", "claude_agent_sdk",
}
SOURCES = sorted((Path(__file__).parent.parent / "agentcore").glob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_agentcore_has_source_files():
    assert SOURCES


def test_agentcore_imports_no_hospital_code_or_agent_framework():
    for path in SOURCES:
        assert not _imported_roots(path) & FORBIDDEN, path.name


def test_agentcore_does_not_use_the_sdk_tool_runner():
    for path in SOURCES:
        assert "tool_runner" not in path.read_text(encoding="utf-8"), path.name
