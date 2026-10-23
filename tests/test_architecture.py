"""分层边界：methods 和 models 不依赖 data、config 与 training 层。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "prose"


def imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "layer, forbidden",
    [
        ("methods", ("prose.data", "prose.config", "prose.training", "prose.models", "prose.semantic_cards")),
        ("models", ("prose.data", "prose.config", "prose.training")),
    ],
)
def test_layer_dependencies(layer: str, forbidden: tuple[str, ...]) -> None:
    violations = [
        f"{path.name} imports {name}"
        for path in sorted((SRC / layer).glob("*.py"))
        for name in imports(path)
        if any(name == item or name.startswith(f"{item}.") for item in forbidden)
    ]
    assert not violations
