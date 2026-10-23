"""训练并评估单次运行；另行独占设备测量效率。

运行目录内容：
    config.resolved.yaml   解析后的配置
    split.json             各划分与评测组的样本数
    normalization.json     训练集统计量
    history.csv            每轮训练损失与验证指标
    best.pt                验证最优检查点，仅含模型状态
    predictions.csv        带评测组字段的测试预测
    metrics.csv            总体及各组的 MAE / RMSE / R2
    summary.json           训练来源记录与完成标记
    efficiency.json        独占设备测量的参数量、耗时与内存

训练时可能与其他运行共享设备，因此 ``summary.json`` 中的耗时只作参考。
``measure`` 重新加载 ``best.pt``，测量推理与训练步耗时；
全部训练完成后，实验调度器在每个设备上用一个进程执行效率测量。
"""

from __future__ import annotations

import fcntl
import json
import random
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from prose.config import RunSpec, efficiency_settings_sha256, implementation_sha256
from prose.data import Normalization, Split, WindowDataset, build_split, load_dataset
from prose.io import git_state, read_json, write_csv, write_json
from prose.models.forecaster import Forecaster, build_forecaster
from prose.training.evaluation import (
    count_parameters,
    grouped_metrics,
    measure_efficiency,
    peak_rss_mb,
    regression_metrics,
)

EVALUATION_SEED = 0  # 评估使用固定随机状态，Informer 在 eval 模式下仍会采样键


def run_status(spec: RunSpec) -> str:
    """运行状态：completed（完成）、outdated（配置相同、代码不同）、stale（配置不同）、
    incomplete（未完成）、not_started（未开始）、blocked（语义产物尚未构建）。"""

    summary = spec.run_dir / "summary.json"
    if summary.exists():
        record = read_json(summary)
        if record.get("config_sha256") != spec.config_sha256:
            return "stale"
        return "completed" if record.get("implementation_sha256") == implementation_sha256() else "outdated"
    method = spec.config["method"]
    if "artifact" in method and method.get("artifact_sha256") is None:
        return "blocked"
    return "incomplete" if spec.run_dir.exists() else "not_started"


def efficiency_status(spec: RunSpec) -> str:
    """效率状态为 measured 或 pending；每次测量对应一份训练结果和一组计时设置。"""

    path = spec.run_dir / "efficiency.json"
    if not path.exists() or not (spec.run_dir / "summary.json").exists():
        return "pending"
    record = read_json(path)
    current = (
        read_json(spec.run_dir / "summary.json").get("completed_at_utc"),
        efficiency_settings_sha256(spec.config),
        implementation_sha256(),
    )
    measured = (
        record.get("run_completed_at_utc"),
        record.get("settings_sha256"),
        record.get("implementation_sha256"),
    )
    return "measured" if measured == current else "pending"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    """每个运行目录只允许一个进程使用；进程退出时释放锁。"""

    lock_path = run_dir.parent / f".{run_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{run_dir} is in use by another process") from None
        yield


@contextmanager
def _evaluation_rng(device: torch.device) -> Iterator[None]:
    """评估结果由模型权重确定，不改变训练随机流。"""

    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(EVALUATION_SEED)
        yield


class Batches:
    """将设备上常驻的张量按 mini-batch 迭代。

    每轮使用独立的 CPU 随机数生成器打乱顺序，复现同一生成器下
    ``DataLoader(shuffle=True)`` 的顺序：迭代器先抽取工作进程的基础种子，
    采样器再调用两次 ``torch.randperm``，第二次排列被截为一个空的剩余切片。
    """

    def __init__(self, tensors: tuple[torch.Tensor, ...], batch_size: int, *, seed: int | None) -> None:
        self.tensors = tensors
        self.batch_size = int(batch_size)
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def __len__(self) -> int:
        return -(-len(self.tensors[0]) // self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        size = len(self.tensors[0])
        order = None
        if self.generator is not None:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            order = torch.randperm(size, generator=self.generator).to(self.tensors[0].device)
            torch.randperm(size, generator=self.generator)
        for start in range(0, size, self.batch_size):
            if order is None:
                yield tuple(tensor[start : start + self.batch_size] for tensor in self.tensors)
            else:
                index = order[start : start + self.batch_size]
                yield tuple(tensor[index] for tensor in self.tensors)


def _batches(
    dataset: WindowDataset,
    indices: np.ndarray,
    normalization: Normalization,
    batch_size: int,
    device: torch.device,
    *,
    seed: int | None = None,
) -> Batches:
    tensors = (
        torch.as_tensor(normalization.windows(dataset.windows[indices])).to(device),
        torch.as_tensor(normalization.targets(dataset.targets[indices])).to(device),
        torch.as_tensor(indices, dtype=torch.long).to(device),
    )
    return Batches(tensors, batch_size, seed=seed)


def _device(training: dict[str, Any]) -> torch.device:
    return torch.device(training.get("device", "cuda") if torch.cuda.is_available() else "cpu")


def _prepare(spec: RunSpec) -> tuple[WindowDataset, Split, Normalization, Forecaster, torch.device]:
    config = spec.config
    device = _device(config["training"])
    set_seed(spec.seed)
    dataset = load_dataset(config["dataset"])
    split = build_split(dataset, config["dataset"])
    normalization = Normalization.fit(dataset.windows[split.train], dataset.targets[split.train])
    model = build_forecaster(
        method=config["method"],
        backbone=config["backbone"],
        feature_names=dataset.feature_names,
        window_size=dataset.window_size,
        output_dim=len(dataset.target_names),
    ).to(device)
    if device.type == "cuda" and getattr(model.backbone, "fast_path", True) is False:
        raise RuntimeError(
            "Mamba CUDA kernels are unavailable, so HuggingFace would fall back to its sequential scan; "
            "run once with network access to cache the pinned kernels (prose.models.mamba_kernels)"
        )
    return dataset, split, normalization, model, device


def run(spec: RunSpec, *, force: bool = False, log=print) -> Path:
    with _run_lock(spec.run_dir):
        status = run_status(spec)
        if status == "completed" and not force:
            log(f"[skip] {spec.name} already completed")
            return spec.run_dir
        if status == "blocked":
            raise RuntimeError(f"semantic artifact {spec.config['method']['artifact']} is not built yet")
        if status in ("stale", "outdated") and not force:
            what = "another config" if status == "stale" else "other code"
            raise RuntimeError(f"{spec.run_dir} holds a run with {what}; pass --force to replace it")
        if spec.run_dir.exists():
            shutil.rmtree(spec.run_dir)
        spec.run_dir.mkdir(parents=True)
        return _train_run(spec, log)


def _train_run(spec: RunSpec, log) -> Path:
    config = spec.config
    (spec.run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    training = config["training"]
    rss_start = peak_rss_mb()  # 导入完成、读取数据与构建模型之前的常驻内存峰值
    dataset, split, normalization, model, device = _prepare(spec)
    write_json(spec.run_dir / "split.json", split.summary())
    write_json(
        spec.run_dir / "normalization.json",
        normalization.as_dict(dataset.feature_names, dataset.target_names),
    )
    batch_size = int(training["batch_size"])
    batches = {
        "train": _batches(dataset, split.train, normalization, batch_size, device, seed=spec.seed),
        "validation": _batches(dataset, split.validation, normalization, batch_size, device),
        "test": _batches(dataset, split.test, normalization, batch_size, device),
    }

    start = time.perf_counter()
    history, best_epoch = train(
        model, batches, normalization, training, device, spec.run_dir / "best.pt", log=log
    )
    train_seconds = time.perf_counter() - start
    write_csv(spec.run_dir / "history.csv", history)

    model.load_state_dict(torch.load(spec.run_dir / "best.pt", map_location=device))
    sample_index, target, prediction = predict(model, batches["test"], normalization, device)
    group_columns = list(dict.fromkeys(config["dataset"].get("eval_groups", {}).values()))
    _write_predictions(
        spec.run_dir / "predictions.csv", dataset, group_columns, sample_index, target, prediction
    )
    write_csv(spec.run_dir / "metrics.csv", grouped_metrics(split, sample_index, target, prediction))

    total, trainable = count_parameters(model)
    adapter_total, _ = count_parameters(model.adapter)
    summary: dict[str, Any] = {
        "run": spec.name,
        "dataset": spec.dataset,
        "backbone": spec.backbone,
        "method": spec.method,
        "variant": spec.variant,
        "seed": spec.seed,
        "overrides": dict(spec.overrides),
        "config_sha256": spec.config_sha256,
        "pairing_sha256": spec.pairing_sha256,
        "implementation_sha256": implementation_sha256(),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_RMSE": float(history[best_epoch - 1]["val_RMSE"]),
        "train_wall_seconds": train_seconds,  # 设备可能被共享，此耗时只作参考
        "parameters": total,
        "trainable_parameters": trainable,
        "adapter_parameters": adapter_total,
        "peak_rss_mb": peak_rss_mb(),
        "peak_rss_increase_mb": peak_rss_mb() - rss_start,
        "artifact": _artifact_provenance(config["method"]),
        "mamba_fast_path": getattr(model.backbone, "fast_path", None),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "git": git_state(),
        "command": sys.argv,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(spec.run_dir / "summary.json", summary)
    log(f"[done] {spec.name} best_epoch={best_epoch} val_RMSE={summary['best_val_RMSE']:.5g}")
    return spec.run_dir


def train(
    model: Forecaster,
    batches: dict[str, Batches],
    normalization: Normalization,
    training: dict[str, Any],
    device: torch.device,
    checkpoint: Path,
    *,
    log=print,
) -> tuple[list[dict[str, Any]], int]:
    """返回训练历史与已保存检查点对应的轮次。

    只有验证 RMSE 相对当前最优值的下降超过 ``early_stopping_rel_delta``，才视为改善。
    不同数据集的标签尺度相差多个数量级，绝对阈值会使部分数据集过早停止。
    """

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
        min_lr=float(training["min_lr"]),
    )
    criterion = torch.nn.MSELoss()
    clip = float(training["gradient_clip_norm"])
    margin = float(training["early_stopping_rel_delta"])
    best, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for x, y, _ in batches["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            loss_sum += loss.detach()
        _, target, prediction = predict(model, batches["validation"], normalization, device)
        metrics = regression_metrics(target, prediction)
        scheduler.step(metrics["RMSE"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss_sum) / len(batches["train"]),
                **{f"val_{key}": value for key, value in metrics.items()},
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if metrics["RMSE"] < best * (1.0 - margin):
            best, best_epoch, stale = metrics["RMSE"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        # 调度器读取该行显示进度
        log(
            f"epoch {epoch}/{training['max_epochs']} train_loss={history[-1]['train_loss']:.4g} "
            f"val_RMSE={metrics['RMSE']:.4g} best={best:.4g}@{best_epoch} "
            f"stale={stale}/{training['early_stopping_patience']} lr={history[-1]['learning_rate']:.2g}"
        )
        if stale >= int(training["early_stopping_patience"]):
            log(f"early stopping at epoch {epoch}")
            break
    if not checkpoint.exists():
        raise RuntimeError("training produced no finite validation result")
    return history, best_epoch


@torch.no_grad()
def predict(
    model: Forecaster, batches: Batches, normalization: Normalization, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices, targets, predictions = [], [], []
    with _evaluation_rng(device):
        for x, y, index in batches:
            predictions.append(model(x))
            targets.append(y)
            indices.append(index)
    return (
        torch.cat(indices).cpu().numpy(),
        normalization.inverse_targets(torch.cat(targets).cpu().numpy()),
        normalization.inverse_targets(torch.cat(predictions).cpu().numpy()),
    )


def measure(spec: RunSpec, *, log=print) -> Path:
    """从已完成运行的检查点测量效率；调用方负责保持设备独占。"""

    with _run_lock(spec.run_dir):
        if run_status(spec) != "completed":
            raise RuntimeError(f"{spec.name} is not a completed run with the current config and code")
        dataset, split, normalization, model, device = _prepare(spec)
        model.load_state_dict(torch.load(spec.run_dir / "best.pt", map_location=device))
        batch_size = int(spec.config["training"]["batch_size"])
        cpu = torch.device("cpu")
        settings = spec.config["efficiency"]
        with _evaluation_rng(device):
            result = measure_efficiency(
                model,
                train_batches=[
                    (x, y)
                    for x, y, _ in _batches(dataset, split.train, normalization, batch_size, cpu, seed=0)
                ],
                test_batches=[x for x, _, _ in _batches(dataset, split.test, normalization, batch_size, cpu)],
                device=device,
                training=spec.config["training"],
                settings=settings,
            )
        summary = read_json(spec.run_dir / "summary.json")
        # 独占设备的训练时间估计：每轮一次训练集更新与一次验证集前向，乘以实际训练轮数
        epoch_ms = (
            split.train.size * result["train_ms_per_sample"]
            + split.validation.size * result["inference_ms_per_sample"]
        )
        write_json(
            spec.run_dir / "efficiency.json",
            {
                **result,
                "train_time_h": summary["epochs_run"] * epoch_ms / 3.6e6,
                "adapter_parameters": count_parameters(model.adapter)[0],
                "settings": settings,
                "settings_sha256": efficiency_settings_sha256(spec.config),
                "implementation_sha256": implementation_sha256(),
                "run_completed_at_utc": summary["completed_at_utc"],
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "torch": torch.__version__,
                "measured_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        log(f"[measured] {spec.name} inference={result['inference_ms_per_sample']:.4f} ms/sample")
        return spec.run_dir


def _write_predictions(
    path: Path,
    dataset: WindowDataset,
    columns: list[str],
    sample_index: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    rows = []
    for row, index in enumerate(sample_index):
        item: dict[str, Any] = {"sample_index": int(index)}
        item.update({column: dataset.meta[int(index)][column] for column in columns})
        for position, name in enumerate(dataset.target_names):
            item[f"target.{name}"] = float(target[row, position])
            item[f"prediction.{name}"] = float(prediction[row, position])
        rows.append(item)
    write_csv(path, rows)


def _artifact_provenance(method: dict[str, Any]) -> dict[str, Any] | None:
    if "artifact" not in method:
        return None
    artifact = Path(method["artifact"])
    if (artifact / "metadata.json").exists():
        metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        return {
            "path": str(artifact),
            "source": metadata.get("source"),
            "arrays_sha256": method.get("artifact_sha256"),
        }
    return {"path": str(artifact)}
