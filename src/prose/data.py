"""窗口级数据集、声明式划分、评测组与归一化。

每个数据集目录提供一个窗口包 ``windows.npz``：

    windows        (N, L, d) float32，最后一步为当前时刻
    targets        (N, p) float32
    feature_names  (d,) 变量名，顺序与 windows 最后一维一致
    target_names   (p,) 标签名
    meta_<列名>    (N,) 逐样本元数据（划分与评测组按这些列声明）

窗口包的 SHA-256 纳入运行身份。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prose.io import resolve_path, sha256_file


@dataclass(frozen=True)
class WindowDataset:
    name: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    windows: np.ndarray  # (N, L, d)
    targets: np.ndarray  # (N, p)
    meta: tuple[dict[str, Any], ...]

    @property
    def window_size(self) -> int:
        return int(self.windows.shape[1])

    def column(self, name: str) -> np.ndarray:
        if name not in self.meta[0]:
            raise KeyError(f"dataset {self.name} has no metadata column {name!r}")
        return np.asarray([row[name] for row in self.meta], dtype=object)


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    groups: dict[str, dict[str, np.ndarray]]  # 分组类型 -> 分组名称 -> 样本索引

    def summary(self) -> dict[str, Any]:
        return {
            "train": int(self.train.size),
            "validation": int(self.validation.size),
            "test": int(self.test.size),
            "groups": {
                kind: {name: int(indices.size) for name, indices in groups.items()}
                for kind, groups in self.groups.items()
            },
        }


@dataclass(frozen=True)
class Normalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, windows: np.ndarray, targets: np.ndarray) -> Normalization:
        x = np.asarray(windows, dtype=np.float64).reshape(-1, windows.shape[-1])
        y = np.asarray(targets, dtype=np.float64)
        mean, std = x.mean(axis=0), x.std(axis=0)
        # 训练时恒定的变量用 |mean| 缩放，均值为零时用 1；
        # 因此测试时的变化表示为相对偏差，避免数值溢出。
        constant = std <= 1e-8 * np.maximum(np.abs(mean), 1.0)
        std = np.where(constant, np.where(np.abs(mean) > 0, np.abs(mean), 1.0), std)
        return cls(
            input_mean=mean.astype(np.float32),
            input_std=std.astype(np.float32),
            target_mean=y.mean(axis=0).astype(np.float32),
            target_std=np.maximum(y.std(axis=0), 1e-12).astype(np.float32),
        )

    def windows(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.input_mean) / self.input_std).astype(np.float32)

    def targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.target_std + self.target_mean

    def as_dict(self, feature_names: tuple[str, ...], target_names: tuple[str, ...]) -> dict[str, Any]:
        return {
            "input_mean": dict(zip(feature_names, self.input_mean.tolist(), strict=True)),
            "input_std": dict(zip(feature_names, self.input_std.tolist(), strict=True)),
            "target_mean": dict(zip(target_names, self.target_mean.tolist(), strict=True)),
            "target_std": dict(zip(target_names, self.target_std.tolist(), strict=True)),
        }


def bundle_path(config: dict[str, Any]) -> Path:
    return resolve_path(config["path"]) / config.get("file", "windows.npz")


_FINGERPRINTS: dict[tuple, str] = {}


def dataset_fingerprint(config: dict[str, Any]) -> dict[str, str | None]:
    """窗口包的 SHA-256；按文件的 (size, mtime) 在进程内缓存。"""

    path = bundle_path(config)
    if not path.exists():
        return {path.name: None}
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _FINGERPRINTS:
        _FINGERPRINTS[key] = sha256_file(path)
    return {path.name: _FINGERPRINTS[key]}


def read_feature_names(config: dict[str, Any]) -> tuple[str, ...]:
    with np.load(bundle_path(config)) as arrays:
        return tuple(str(name) for name in arrays["feature_names"])


def load_dataset(config: dict[str, Any]) -> WindowDataset:
    with np.load(bundle_path(config)) as arrays:
        features = tuple(str(name) for name in arrays["feature_names"])
        targets = tuple(str(name) for name in arrays["target_names"])
        x = np.asarray(arrays["windows"], dtype=np.float32)
        y = np.asarray(arrays["targets"], dtype=np.float32).reshape(len(x), -1)
        columns = {key[len("meta_") :]: arrays[key].tolist() for key in arrays.files if key.startswith("meta_")}
    meta = [dict(zip(columns, values, strict=True)) for values in zip(*columns.values(), strict=True)]
    if x.ndim != 3 or x.shape[2] != len(features) or y.shape[1] != len(targets) or len(meta) != len(x):
        raise ValueError(f"dataset {config['name']} arrays do not match their names and metadata")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"dataset {config['name']} contains non-finite values")
    return WindowDataset(str(config["name"]), features, targets, x, y, tuple(meta))


def _select(dataset: WindowDataset, rule: dict[str, Any], pool: np.ndarray | None = None) -> np.ndarray:
    values = dataset.column(str(rule["column"]))
    mask = np.isin(values, np.asarray(rule["values"], dtype=object))
    indices = np.flatnonzero(mask)
    if pool is not None:
        indices = np.intersect1d(indices, pool)
    if indices.size == 0:
        raise ValueError(f"no samples match {rule} in dataset {dataset.name}")
    return indices.astype(np.int64)


def build_split(dataset: WindowDataset, config: dict[str, Any]) -> Split:
    """按声明的元数据规则划分；验证集从训练数据池中留出。"""

    split = config["split"]
    pool = _select(dataset, split["train"])
    validation = _select(dataset, split["validation"], pool)
    train = np.setdiff1d(pool, validation)
    test = _select(dataset, split["test"])
    if np.intersect1d(test, pool).size:
        raise ValueError(f"test samples overlap the training pool in dataset {dataset.name}")
    groups: dict[str, dict[str, np.ndarray]] = {}
    for kind, column in config.get("eval_groups", {}).items():
        values = dataset.column(str(column))[test]
        groups[kind] = {str(value): test[values == value] for value in sorted(set(values), key=str)}
    return Split(train=train, validation=validation, test=test, groups=groups)
