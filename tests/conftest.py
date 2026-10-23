from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

FEATURES = ("feed", "pressure", "phase")
WINDOW = 8


def write_synthetic_dataset(root: Path, *, samples: int = 72, seed: int = 0) -> Path:
    """包含 train / same_family / far 分组的小型窗口包。"""

    rng = np.random.default_rng(seed)
    dataset = root / "toy"
    dataset.mkdir(parents=True)
    buckets = [("train", "train", "train", "train", "same_family", "far")[i % 6] for i in range(samples)]
    windows = rng.normal(size=(samples, WINDOW, len(FEATURES))).astype(np.float32)
    targets = windows[:, :, 0].mean(axis=1, keepdims=True) + 0.5 * windows[:, -1, 1:2]
    np.savez_compressed(
        dataset / "windows.npz",
        windows=windows,
        targets=targets.astype(np.float32),
        feature_names=np.asarray(FEATURES),
        target_names=np.asarray(["y"]),
        meta_domain_bucket=np.asarray(buckets),
        meta_scenario_name=np.asarray([f"{b}_{i % 12 // 6}" for i, b in enumerate(buckets)]),
    )
    return dataset


# 只在测试中使用的结构开关组合（代码中的 coupling / local_term）
TEST_METHODS = {
    "prose_random_semantics": {"inherit": "prose", "semantics": "random"},
    "prose_no_field": {"inherit": "prose", "coupling": "none"},
    "prose_time_index": {"inherit": "prose", "coupling": "none", "local_term": False},
}


@pytest.fixture
def toy_config_dir(tmp_path: Path) -> Path:
    """只包含一个合成数据集的配置目录。"""

    from prose.semantic_artifact import build_random

    dataset = write_synthetic_dataset(tmp_path)
    build_random(FEATURES, dim=6, seed=0).save(tmp_path / "artifacts" / "random")
    config_dir = tmp_path / "configs"
    (config_dir / "datasets").mkdir(parents=True)
    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("backbones.yaml", "llm.yaml", "embedding.yaml"):
        (config_dir / name).write_text((root / name).read_text(encoding="utf-8"), encoding="utf-8")
    methods = yaml.safe_load((root / "methods.yaml").read_text(encoding="utf-8"))
    (config_dir / "methods.yaml").write_text(yaml.safe_dump({**methods, **TEST_METHODS}), encoding="utf-8")
    training = yaml.safe_load((root / "training.yaml").read_text(encoding="utf-8"))
    training["training"].update({"device": "cpu", "max_epochs": 2, "batch_size": 16})
    training["efficiency"].update(
        {"timing_warmup_loops": 0, "timing_repeat_loops": 1, "latency_samples": 3, "train_timing_steps": 2}
    )
    (config_dir / "training.yaml").write_text(yaml.safe_dump(training), encoding="utf-8")
    (config_dir / "datasets" / "toy.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "path": str(dataset),
                "file": "windows.npz",
                "split": {
                    "train": {"column": "domain_bucket", "values": ["train"]},
                    "validation": {"column": "scenario_name", "values": ["train_1"]},
                    "test": {"column": "domain_bucket", "values": ["same_family", "far"]},
                },
                "eval_groups": {"domain": "domain_bucket", "scenario": "scenario_name"},
                "artifacts": {
                    "prose_llm": str(tmp_path / "artifacts" / "llm"),  # 不构建 LLM 语义产物
                    "prose_random": str(tmp_path / "artifacts" / "random"),
                },
            }
        ),
        encoding="utf-8",
    )
    return config_dir
