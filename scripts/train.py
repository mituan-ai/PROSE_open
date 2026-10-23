"""训练并评估单次运行，或测量已完成运行的效率。

    uv run --extra train python scripts/train.py --dataset thickener_dewatering --backbone gru \
        --method prose --seed 2024 --device cpu --set training.max_epochs=1
    # 从 best.pt 测量效率；期间保持设备独占
    uv run --extra train python scripts/train.py --dataset thickener_dewatering --backbone gru \
        --method prose --seed 2024 --efficiency
"""

from __future__ import annotations

import argparse

import yaml

from prose.config import RUN_ROOT, resolve_run
from prose.io import resolve_path
from prose.training.runner import measure, run


def parse_overrides(items: list[str]) -> dict[str, object]:
    overrides = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--device", help="cuda, cuda:1, or cpu; not part of the run identity")
    parser.add_argument("--output-root", default=str(RUN_ROOT))
    parser.add_argument("--force", action="store_true", help="replace a completed, stale, or outdated run")
    parser.add_argument(
        "--efficiency", action="store_true", help="measure a completed run instead of training"
    )
    args = parser.parse_args()

    spec = resolve_run(
        args.dataset,
        args.backbone,
        args.method,
        args.seed,
        parse_overrides(args.set),
        output_root=resolve_path(args.output_root),
    )
    if args.device:
        spec.config["training"]["device"] = args.device
    print(measure(spec) if args.efficiency else run(spec, force=args.force))


if __name__ == "__main__":
    main()
