from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from prose.models.forecaster import build_forecaster
from prose.semantic_artifact import build_random

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = yaml.safe_load((ROOT / "configs" / "backbones.yaml").read_text(encoding="utf-8"))
NAMES = ("a", "b", "c", "d")


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("artifacts")
    build_random(NAMES, dim=8, seed=0).save(root / "prose")
    return {"prose": root / "prose"}


@pytest.mark.parametrize("backbone", sorted(BACKBONES))
@pytest.mark.parametrize("method", ["base", "prose"])
def test_every_backbone_reads_every_adapter_output(backbone: str, method: str, artifacts) -> None:
    config = {"name": method}
    if method != "base":
        config["artifact"] = str(artifacts[method])
    if method == "prose":
        config.update({"semantics": "random", "memory_dim": 4, "coupling": "semantic", "local_term": True})
    model = build_forecaster(
        method=config, backbone=BACKBONES[backbone], feature_names=NAMES, window_size=16, output_dim=3
    )
    out, components = model.forward_components(torch.randn(2, 16, len(NAMES)))
    assert out.shape == (2, 3)
    assert (components is not None) == (method == "prose")
    if method == "prose":
        assert components.z.shape == (2, 16, len(NAMES))


@pytest.mark.parametrize("window", [30, 60, 120])
def test_patchtst_reads_the_latest_steps_and_keeps_channels_independent(window: int) -> None:
    torch.manual_seed(0)
    backbone = build_forecaster(
        method={"name": "base"},
        backbone=BACKBONES["patchtst"],
        feature_names=NAMES,
        window_size=window,
        output_dim=1,
    ).backbone.eval()
    x = torch.randn(2, window, len(NAMES), requires_grad=True)
    out = backbone(x)
    assert torch.autograd.grad(out.sum(), x, retain_graph=True)[0][:, -1].abs().min() > 0
    blocks = out.reshape(2, len(NAMES), -1)
    grad_other = torch.autograd.grad(blocks[:, 0].sum(), x)[0][..., 1:]
    assert grad_other.abs().max() == 0  # 变量 0 独立编码，不混入其他变量


def test_patchtst_patch_positions_matter() -> None:
    torch.manual_seed(0)
    backbone = build_forecaster(
        method={"name": "base"},
        backbone=BACKBONES["patchtst"],
        feature_names=NAMES,
        window_size=32,
        output_dim=1,
    ).backbone.eval()
    x = torch.randn(1, 32, len(NAMES))
    swapped = torch.cat((x[:, 16:], x[:, :16]), dim=1)  # 相同分块采用不同顺序
    assert not torch.allclose(backbone(x).sum(dim=-1), backbone(swapped).sum(dim=-1))


@pytest.mark.parametrize("backbone", ["gru", "patchtst", "informer"])
def test_methods_share_backbone_initialization_and_random_stream(backbone: str, artifacts) -> None:
    methods = {
        "base": {"name": "base"},
        "prose": {
            "name": "prose",
            "artifact": str(artifacts["prose"]),
            "semantics": "random",
            "memory_dim": 4,
            "coupling": "semantic",
            "local_term": True,
        },
    }
    states, streams = {}, {}
    for name, config in methods.items():
        torch.manual_seed(2024)
        model = build_forecaster(
            method=config, backbone=BACKBONES[backbone], feature_names=NAMES, window_size=16, output_dim=3
        )
        states[name] = {
            key: value for key, value in model.state_dict().items() if not key.startswith("adapter.")
        }
        streams[name] = torch.rand(3)
    assert states["prose"].keys() == states["base"].keys()
    assert all(torch.equal(states["prose"][key], states["base"][key]) for key in states["base"])
    assert torch.equal(streams["prose"], streams["base"])


def test_itransformer_tokens_have_no_variable_embedding() -> None:
    """官方 iTransformer 没有位置或变量编码：交换输入变量只交换对应的输出 token。"""

    torch.manual_seed(0)
    backbone = build_forecaster(
        method={"name": "base"},
        backbone=BACKBONES["itransformer"],
        feature_names=NAMES,
        window_size=16,
        output_dim=1,
    ).backbone.eval()
    x = torch.randn(2, 16, len(NAMES))
    order = torch.tensor([2, 0, 3, 1])
    tokens = backbone(x).reshape(2, len(NAMES), -1)
    permuted = backbone(x[..., order]).reshape(2, len(NAMES), -1)
    assert torch.allclose(permuted, tokens[:, order], atol=1e-5)
