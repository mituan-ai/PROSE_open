"""预测器 = 输入适配器（Base / PROSE）+ 主干 + 线性输出头。

先构建主干与输出头，再在分叉的随机状态中初始化适配器。
因此在相同种子下，各方法的主干和输出头初始权重一致，
并为后续训练保留相同的全局随机流（如 dropout）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from prose.methods.prose import ProseAdapter, ProseOutput
from prose.models.backbones import build_backbone
from prose.semantic_artifact import load_artifact

METHODS = ("base", "prose")


class Forecaster(nn.Module):
    def __init__(self, *, adapter: nn.Module, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(self.adapter(x)))

    def forward_components(self, x: Tensor) -> tuple[Tensor, ProseOutput | None]:
        if isinstance(self.adapter, ProseAdapter):
            components = self.adapter(x, return_components=True)
            return self.head(self.backbone(components.z)), components
        return self.forward(x), None


def build_adapter(method: dict[str, Any], *, feature_names: tuple[str, ...], window_size: int) -> nn.Module:
    name = method["name"]
    if name == "base":
        return nn.Identity()
    if name == "prose":
        if "artifact" in method:
            semantics = load_artifact(Path(method["artifact"]), feature_names)
            history, condition = semantics.history, semantics.condition
        else:  # coupling="none" 时不读取语义坐标
            history = condition = np.zeros((len(feature_names), 1))
        return ProseAdapter(
            history_semantics=history,
            condition_semantics=condition,
            window_size=window_size,
            memory_dim=int(method["memory_dim"]),
            coupling=str(method.get("coupling", "semantic")),
            local_term=bool(method.get("local_term", True)),
            scan=str(method.get("scan", "parallel")),
        )
    raise ValueError(f"Unsupported method {name!r}; expected one of {METHODS}")


def build_forecaster(
    *,
    method: dict[str, Any],
    backbone: dict[str, Any],
    feature_names: tuple[str, ...],
    window_size: int,
    output_dim: int,
) -> Forecaster:
    backbone_config = dict(backbone)
    kind = str(backbone_config.pop("type"))
    backbone_module = build_backbone(
        kind, input_dim=len(feature_names), window_size=window_size, config=backbone_config
    )
    head = nn.Linear(int(backbone_module.output_dim), output_dim)
    with torch.random.fork_rng(devices=[]):
        adapter = build_adapter(method, feature_names=feature_names, window_size=window_size)
    return Forecaster(adapter=adapter, backbone=backbone_module, head=head)
