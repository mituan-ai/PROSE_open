"""指标、分组评估与效率测量。"""

from __future__ import annotations

import copy
import math
import platform
import resource
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from prose.data import Split


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    true = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    error = pred - true
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(math.sqrt(np.mean(error**2))),
        "R2": float(1.0 - np.sum(error**2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def grouped_metrics(
    split: Split, sample_index: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> list[dict[str, object]]:
    """为整个测试集生成一行，再为每个声明的评测组各生成一行。"""

    row_of = {int(index): row for row, index in enumerate(sample_index)}
    rows: list[dict[str, object]] = [
        {
            "group_type": "overall",
            "group": "test",
            "n_samples": len(sample_index),
            **regression_metrics(target, prediction),
        }
    ]
    for kind, groups in split.groups.items():
        for name, indices in groups.items():
            selected = np.asarray([row_of[int(index)] for index in indices], dtype=np.int64)
            rows.append(
                {
                    "group_type": kind,
                    "group": name,
                    "n_samples": int(selected.size),
                    **regression_metrics(target[selected], prediction[selected]),
                }
            )
    return rows


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def measure_efficiency(
    model: nn.Module,
    *,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    test_batches: list[torch.Tensor],
    device: torch.device,
    training: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """测量批量推理耗时（ms/sample）、单样本延迟（ms，均值、中位数与 P95）、训练步耗时
    （ms/sample），以及推理与训练的峰值已分配 GPU 显存（MiB）。

    每项显存峰值包含模型、一个 batch，以及训练时的优化器状态，不包含常驻数据集。
    不修改传入的 ``model``。
    """

    total, trainable = count_parameters(model)
    result: dict[str, Any] = {"parameters": total, "trainable_parameters": trainable}
    model = model.to(device).eval()
    with torch.no_grad():
        _reset_peak(device)
        model(test_batches[0].to(device))
        result["inference_peak_gpu_memory_mb"] = _peak_mb(device)

        resident = [x.to(device) for x in test_batches]
        for _ in range(int(settings["timing_warmup_loops"])):
            for x in resident:
                model(x)
        _synchronize(device)
        start, samples = time.perf_counter(), 0
        for _ in range(max(int(settings["timing_repeat_loops"]), 1)):
            for x in resident:
                model(x)
                samples += int(x.shape[0])
        _synchronize(device)
        result["inference_ms_per_sample"] = (time.perf_counter() - start) / samples * 1000.0

        singles = torch.cat(resident)[: int(settings["latency_samples"])].unsqueeze(1)
        for x in singles[: min(10, len(singles))]:
            model(x)
        timings = []
        for x in singles:
            _synchronize(device)
            start = time.perf_counter()
            model(x)
            _synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
        result["latency_ms_mean"] = float(np.mean(timings))
        result["latency_ms_median"] = float(np.median(timings))
        result["latency_ms_p95"] = float(np.percentile(timings, 95))
        del resident, singles

    trainer = copy.deepcopy(model).train()
    optimizer = torch.optim.AdamW(
        trainer.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.MSELoss()
    clip = float(training["gradient_clip_norm"])

    def step(x: torch.Tensor, y: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        criterion(trainer(x), y).backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), clip)
        optimizer.step()

    warmup, timed = int(settings["train_warmup_steps"]), max(int(settings["train_timing_steps"]), 1)
    needed = [train_batches[index % len(train_batches)] for index in range(warmup + timed)]
    _reset_peak(device)
    x, y = needed[0]
    step(x.to(device), y.to(device))
    step(x.to(device), y.to(device))  # 第二步也包含已创建的优化器状态
    result["train_peak_gpu_memory_mb"] = _peak_mb(device)
    resident = [(x.to(device), y.to(device)) for x, y in needed]
    for x, y in resident[:warmup]:
        step(x, y)
    _synchronize(device)
    start, samples = time.perf_counter(), 0
    for x, y in resident[warmup:]:
        step(x, y)
        samples += int(x.shape[0])
    _synchronize(device)
    result["train_ms_per_sample"] = (time.perf_counter() - start) / samples * 1000.0
    return result


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024)
