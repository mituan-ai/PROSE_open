from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import FEATURES, WINDOW, write_synthetic_dataset

from prose.config import dataset_config
from prose.data import Normalization, build_split, load_dataset


def toy_config(dataset: Path) -> dict:
    return {
        "name": "toy",
        "path": str(dataset),
        "file": "windows.npz",
        "split": {
            "train": {"column": "domain_bucket", "values": ["train"]},
            "validation": {"column": "scenario_name", "values": ["train_1"]},
            "test": {"column": "domain_bucket", "values": ["same_family", "far"]},
        },
        "eval_groups": {"domain": "domain_bucket"},
    }


def test_window_bundle_layout_and_split(tmp_path: Path) -> None:
    config = toy_config(write_synthetic_dataset(tmp_path))
    dataset = load_dataset(config)
    assert dataset.windows.shape == (72, WINDOW, len(FEATURES))
    assert np.allclose(
        dataset.targets[:, 0],
        dataset.windows[:, :, 0].mean(axis=1) + 0.5 * dataset.windows[:, -1, 1],
        atol=1e-5,
    )
    split = build_split(dataset, config)
    assert split.train.size + split.validation.size == 48 and split.test.size == 24
    assert set(split.groups["domain"]) == {"same_family", "far"}
    assert not set(split.train) & set(split.validation)
    assert dataset.meta[0] == {"domain_bucket": "train", "scenario_name": "train_0"}


def test_constant_training_variables_get_a_finite_scale() -> None:
    x = np.zeros((10, 4, 3), dtype=np.float32)
    x[..., 0] = np.random.default_rng(0).normal(size=(10, 4))
    x[..., 1] = 100.0  # 全部训练批次中的搅拌转速
    norm = Normalization.fit(x, np.zeros((10, 1), dtype=np.float32))
    changed = x.copy()
    changed[..., 1] = 80.0
    assert np.allclose(norm.windows(x)[..., 1:], 0.0)
    assert np.allclose(norm.windows(changed)[..., 1], -0.2)
    assert np.isclose(norm.input_std[2], 1.0)


def test_normalization_uses_given_statistics() -> None:
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(50, 4, 2)).astype(np.float32)
    y = np.random.default_rng(1).normal(size=(50, 1)).astype(np.float32)
    norm = Normalization.fit(x, y)
    assert np.allclose(norm.windows(x).reshape(-1, 2).mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(norm.inverse_targets(norm.targets(y)), y, atol=1e-5)


@pytest.mark.dataset
@pytest.mark.parametrize(
    "name, train, test",
    [
        ("thickener_dewatering", 1626, 3523),
        ("indpensim", 541, 652),
        ("ladle_preheating", 17736, 2104),
    ],
)
def test_real_dataset_protocols(name: str, train: int, test: int) -> None:
    config = dataset_config(name)
    split = build_split(load_dataset(config), config)
    assert split.train.size + split.validation.size == train
    assert split.test.size == test
