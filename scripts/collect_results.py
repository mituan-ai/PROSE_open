"""将一个实验清单的结果收集为原始长表，每个运行、每个评测组各占一行。

    uv run python scripts/collect_results.py configs/studies/smoke.yaml

输出：outputs/results/<study>/raw_results.csv。逐行复制各运行的 metrics.csv、summary.json
与 efficiency.json，不取平均，也不筛选。全部声明的运行必须使用当前配置和代码完成，
且已测量效率。脚本还检查按 (dataset, backbone, seed, sweep) 配对的运行
是否使用相同的数据内容、主干与训练配置。

``--allow-missing`` 改为写入 raw_results.partial.csv，制表脚本不会读取该文件。
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from prose.config import load_study
from prose.io import ROOT, read_csv, read_json, resolve_path, write_csv
from prose.training.runner import efficiency_status, run_status

SUMMARY_FIELDS = (
    "best_epoch",
    "epochs_run",
    "train_wall_seconds",
    "peak_rss_mb",
    "peak_rss_increase_mb",
    "device_name",
)
EFFICIENCY_FIELDS = (
    "parameters",
    "adapter_parameters",
    "inference_ms_per_sample",
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_p95",
    "train_ms_per_sample",
    "train_time_h",
    "inference_peak_gpu_memory_mb",
    "train_peak_gpu_memory_mb",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("study")
    parser.add_argument(
        "--allow-missing", action="store_true", help="write raw_results.partial.csv for finished runs only"
    )
    args = parser.parse_args()

    study = load_study(resolve_path(args.study))
    missing = [
        spec.name
        for spec in study.runs
        if run_status(spec) != "completed" or efficiency_status(spec) != "measured"
    ]
    if missing and not args.allow_missing:
        raise SystemExit(
            f"{len(missing)} of {len(study.runs)} runs are not completed with the current config and code "
            f"or not measured, e.g. {missing[:3]}"
        )

    rows = []
    pairing: dict[tuple, set[str]] = defaultdict(set)
    for spec in study.runs:
        if spec.name in missing:
            continue
        summary = read_json(spec.run_dir / "summary.json")
        efficiency = read_json(spec.run_dir / "efficiency.json")
        memberships = study.sweeps.get(spec.name) or [("", "")]
        for sweep_key, sweep_value in memberships:
            pairing[(spec.dataset, spec.backbone, spec.seed, sweep_key, str(sweep_value))].add(
                summary["pairing_sha256"]
            )
            base = {
                "study": study.name,
                "dataset": spec.dataset,
                "backbone": spec.backbone,
                "method": spec.method,
                "variant": spec.variant,
                "seed": spec.seed,
                "sweep_key": sweep_key,
                "sweep_value": sweep_value,
                **{field: summary.get(field) for field in SUMMARY_FIELDS},
                **{field: efficiency.get(field) for field in EFFICIENCY_FIELDS},
                "run_dir": spec.run_dir.relative_to(ROOT)
                if spec.run_dir.is_relative_to(ROOT)
                else spec.run_dir,
            }
            for metric in read_csv(spec.run_dir / "metrics.csv"):
                rows.append({**base, **metric})
    unpaired = {key: value for key, value in pairing.items() if len(value) > 1 and not key[3]}
    if unpaired:
        raise SystemExit(f"paired runs differ outside the method config: {list(unpaired)[:3]}")
    name = "raw_results.partial.csv" if missing else "raw_results.csv"
    output = ROOT / "outputs" / "results" / study.name / name
    write_csv(output, rows)
    print(f"{output}: {len(rows)} rows from {len(study.runs) - len(missing)} runs")


if __name__ == "__main__":
    main()
