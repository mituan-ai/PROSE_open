"""离线语义模型登记表：``configs/llm.yaml`` 与 ``configs/embedding.yaml``。

每个文件指定一个 ``default`` 模型，并在 ``models`` 下列出候选模型；
模型条目继承 ``defaults``，也可覆盖任意字段。
候选模型未设置 API Key 或 base URL 时，报告为未配置，批量生成时跳过；
因此可以先登记模型，再配置密钥。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from prose.io import expand_env, load_yaml


def model_slug(name: str) -> str:
    """适用于文件系统的模型名称，例如 ``vendor/model-1`` -> ``vendor-model-1``。"""

    return re.sub(r"[^A-Za-z0-9.]+", "-", name).strip("-")


def source_name(llm: str, embedding: str) -> str:
    """由一个 LLM 与一个编码器共同构建的语义坐标目录名。"""

    return f"{model_slug(llm)}__{model_slug(embedding)}"


@dataclass(frozen=True)
class Registry:
    path: Path
    default: str
    entries: dict[str, dict[str, Any]]  # 已合并 defaults；尚未展开 ${VAR}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def resolve(self, name: str) -> str:
        name = self.default if name == "default" else str(name)
        if name not in self.entries:
            raise KeyError(f"{name!r} is not a model in {self.path.name}; available: {list(self.entries)}")
        return name

    def select(self, name: str) -> tuple[str, ...]:
        """``all`` 返回全部模型，默认模型排在首位；其他值返回一个解析后的模型名称。"""

        if name == "all":
            return (self.default, *(item for item in self.entries if item != self.default))
        return (self.resolve(name),)

    def entry(self, name: str) -> dict[str, Any]:
        """返回模型设置，并将 ${VAR} 替换为环境变量值。"""

        name = self.resolve(name)
        return {**expand_env(self.entries[name]), "model": name}

    def configured(self, name: str) -> bool:
        entry = self.entry(name)
        return bool(str(entry.get("base_url", "")).strip()) and bool(
            os.environ.get(str(entry["api_key_env"]))
        )


def load_registry(path: Path) -> Registry:
    stat = path.stat()
    return _load_registry(str(path), stat.st_size, stat.st_mtime_ns)


@cache
def _load_registry(path: str, _size: int, _mtime: int) -> Registry:
    payload = load_yaml(Path(path))
    defaults = payload.get("defaults") or {}
    models = payload.get("models") or {}
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{path} must list at least one model under 'models'")
    entries = {str(name): {**defaults, **(settings or {})} for name, settings in models.items()}
    default = str(payload.get("default", ""))
    if default not in entries:
        raise ValueError(f"{path}: default {default!r} is not one of the listed models")
    for name, entry in entries.items():
        if "api_key_env" not in entry:
            raise ValueError(f"{path}: model {name!r} has no api_key_env")
    return Registry(Path(path), default, entries)
