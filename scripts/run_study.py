"""按实验清单排队执行各次运行，自动摊到全部 GPU，并显示实时进度。

    uv run python scripts/run_study.py                                   # 默认读取 configs/run.yaml
    uv run python scripts/run_study.py configs/studies/smoke.yaml --status
    uv run python scripts/run_study.py --match indpensim.gru
    uv run python scripts/run_study.py --devices cuda:0 cuda:2

设备默认取全部可见 GPU（可用 CUDA_VISIBLE_DEVICES 限制），没有 GPU 时使用 CPU。
每个设备默认同时只运行一个 run：PROSE 的跨变量项受显存带宽限制，同卡并发会降低总吞吐。
待运行的 run 按估计计算量从大到小排队，避免最后只剩一个长 run 拖尾。

分为三个阶段：
1. 预热：在工作进程启动前，为各数据集构建一次缓存，并获取 Mamba 内核；
2. 训练：每次运行使用独立进程；
3. 效率测量：全部训练完成后，从 best.pt 加载模型计时，每个设备只运行一个进程。
   期间不要在这些设备上运行其他 GPU 任务；``--no-efficiency`` 可跳过此阶段。

已使用相同解析配置和代码完成的运行会跳过。尚未构建语义产物的运行标记为 blocked
并跳过；构建产物后可重新执行相同命令。中断（Ctrl-C）后未完成的运行下次从头训练。
日志保存到 outputs/logs/<study>/<run>.log 与 <run>.efficiency.log。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from prose.config import RunSpec, load_study
from prose.io import ROOT, resolve_path
from prose.training.runner import efficiency_status, run_status

DEFAULT_STUDY = ROOT / "configs" / "run.yaml"
PROGRESS = re.compile(
    r"epoch (\d+)/(\d+) train_loss=(\S+) val_RMSE=(\S+) best=(\S+)@(\d+) stale=(\d+)/(\d+)"
)
REFRESH_SECONDS = 1.0
PLAIN_REPORT_SECONDS = 60.0  # 输出不是终端时（如 nohup），每隔该时间打印一次面板


def command(spec: RunSpec, device: str, force: bool, *, efficiency: bool = False) -> list[str]:
    args = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--dataset",
        spec.dataset,
        "--backbone",
        spec.backbone,
        "--method",
        spec.method,
        "--seed",
        str(spec.seed),
        "--device",
        device,
        "--output-root",
        spec.config["output_root"],
    ]
    for key, value in spec.overrides:
        args += ["--set", f"{key}={value}"]
    for key, value in spec.config.get("efficiency", {}).items():
        args += ["--set", f"efficiency.{key}={value}"]
    if efficiency:
        return args + ["--efficiency"]
    return args + (["--force"] if force else [])


def detect_devices() -> list[str]:
    import torch

    count = torch.cuda.device_count()
    return [f"cuda:{index}" for index in range(count)] if count else ["cpu"]


def warm_up(specs: list[RunSpec], devices: list[str]) -> dict[str, float]:
    """构建数据缓存并返回各数据集每个 epoch 的相对计算量（训练样本数 x 窗口长度 x 变量数）。"""

    from prose.data import build_split, load_dataset

    sizes = {}
    for name, config in {spec.dataset: spec.config["dataset"] for spec in specs}.items():
        dataset = load_dataset(config)
        train = build_split(dataset, config).train.size
        sizes[name] = float(train * dataset.window_size * len(dataset.feature_names))
        print(f"warm-up: data cache of {name} ready ({train} training samples)", flush=True)
    if any(spec.backbone == "mamba" for spec in specs) and any(
        device.startswith("cuda") for device in devices
    ):
        from prose.models.mamba_kernels import enable_hf_mamba_kernels

        if not enable_hf_mamba_kernels():
            raise SystemExit("Mamba CUDA kernels are unavailable; fetch them once with network access")
        print("warm-up: Mamba kernels ready", flush=True)
    return sizes


@dataclass
class Job:
    spec: RunSpec
    weight: float  # 估计计算量，只用于排队顺序与剩余时间
    log_path: Path
    device: str = ""
    process: subprocess.Popen | None = None
    started: float = 0.0
    progress: re.Match | None = field(default=None, repr=False)

    def fraction(self) -> float:
        if self.progress is None:
            return 0.0
        return int(self.progress[1]) / max(int(self.progress[2]), 1)

    def poll_progress(self) -> None:
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(max(handle.seek(0, os.SEEK_END) - 4096, 0))
                tail = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return
        matches = list(PROGRESS.finditer(tail))
        if matches:
            self.progress = matches[-1]


def clock(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def bar(fraction: float, width: int) -> str:
    filled = round(min(max(fraction, 0.0), 1.0) * width)
    return "█" * filled + "░" * (width - filled)


class Dashboard:
    """终端中原地刷新的进度面板；输出不是终端时改为定期打印汇总行。"""

    def __init__(self, title: str, total: int, total_weight: float) -> None:
        self.title, self.total, self.total_weight = title, total, total_weight
        self.start = time.monotonic()
        self.done = self.failed = 0
        self.done_weight = 0.0
        self.drawn = 0
        self.tty = sys.stdout.isatty()
        self.last_plain = self.start

    def event(self, text: str) -> None:
        self._clear()
        print(text, flush=True)

    def summary(self, running: list[Job]) -> str:
        elapsed = time.monotonic() - self.start
        work = self.done_weight + sum(job.weight * job.fraction() for job in running)
        eta = elapsed * (self.total_weight - work) / work if work > 0 else None
        return (
            f"{self.title}  [{bar(self.done / max(self.total, 1), 24)}] {self.done}/{self.total}"
            f"  running {len(running)}  failed {self.failed}  elapsed {clock(elapsed)}"
            f"  eta {'~' + clock(eta) if eta is not None else '--'}"
        )

    def draw(self, running: list[Job], idle: list[str]) -> None:
        if not self.tty:
            if time.monotonic() - self.last_plain >= PLAIN_REPORT_SECONDS:
                self.last_plain = time.monotonic()
                print("\n".join(self.lines(running, [])), flush=True)
            return
        width = shutil.get_terminal_size((160, 24)).columns
        lines = self.lines(running, idle)
        self._clear()
        print("\n".join(line[: width - 1] for line in lines), flush=True)
        self.drawn = len(lines)

    def lines(self, running: list[Job], idle: list[str]) -> list[str]:
        lines = [self.summary(running)]
        for job in sorted(running, key=lambda job: job.device):
            line = f"  {job.device:<7} {job.spec.name}"
            match = job.progress
            if match is not None:
                epoch, epochs, loss, val, best, best_epoch, stale, patience = match.groups()
                line += (
                    f"  ep {epoch:>3}/{epochs} [{bar(job.fraction(), 10)}]  loss {loss}  val {val}"
                    f"  best {best}@{best_epoch}  stale {stale}/{patience}"
                )
            lines.append(f"{line}  {clock(time.monotonic() - job.started)}")
        return lines + [f"  {device:<7} idle" for device in idle]

    def _clear(self) -> None:
        if self.tty and self.drawn:
            sys.stdout.write(f"\x1b[{self.drawn}F\x1b[J")
            self.drawn = 0


def execute_all(title: str, jobs: list[Job], slots: list[str], build) -> list[str]:
    """在 ``slots`` 中的设备上依次执行 ``build(spec, device)``，返回失败的运行名称。"""

    dashboard = Dashboard(title, len(jobs), sum(job.weight for job in jobs))
    queue, running, free, failures = list(jobs), [], list(slots), []
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        while queue or running:
            while queue and free:
                job = queue.pop(0)
                job.device, job.started = free.pop(0), time.monotonic()
                try:
                    with job.log_path.open("w", encoding="utf-8") as log:
                        job.process = subprocess.Popen(
                            build(job.spec, job.device),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            cwd=ROOT,
                            env=environment,
                        )
                except OSError as error:  # 任务无法启动，例如日志文件无法打开
                    dashboard.event(f"FAILED {job.spec.name}: {type(error).__name__}: {error}")
                    failures.append(job.spec.name)
                    dashboard.failed += 1
                    free.append(job.device)
                    continue
                running.append(job)
            for job in list(running):
                code = job.process.poll()
                if code is None:
                    job.poll_progress()
                    continue
                running.remove(job)
                free.append(job.device)
                job.poll_progress()
                dashboard.done += 1
                dashboard.done_weight += job.weight
                took = clock(time.monotonic() - job.started)
                if code == 0:
                    best = f"  best {job.progress[5]}@{job.progress[6]}" if job.progress else ""
                    dashboard.event(f"done    {job.spec.name}  {took}{best}")
                else:
                    failures.append(job.spec.name)
                    dashboard.failed += 1
                    dashboard.event(f"FAILED  {job.spec.name}  exit {code}  see {job.log_path}")
            dashboard.draw(running, sorted(free) if not queue else [])
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        for job in running:
            job.process.terminate()
        for job in running:
            job.process.wait()
        dashboard.event(f"interrupted; {len(running)} unfinished runs restart from scratch next time")
        raise SystemExit(130) from None
    dashboard.event(dashboard.summary([]))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("study", nargs="?", default=str(DEFAULT_STUDY))
    parser.add_argument("--status", action="store_true", help="show progress without running")
    parser.add_argument("--dry-run", action="store_true", help="list pending runs without running")
    parser.add_argument("--match", nargs="+", default=[], help="keep runs whose name contains every token")
    parser.add_argument("--devices", nargs="+", help="default: every visible GPU, else cpu")
    parser.add_argument("--per-device", type=int, default=1, help="concurrent training runs per device")
    parser.add_argument("--force", action="store_true", help="rerun stale and outdated runs")
    parser.add_argument("--no-efficiency", action="store_true", help="skip the efficiency phase")
    args = parser.parse_args()

    study = load_study(resolve_path(args.study))
    runs = [spec for spec in study.runs if all(token in spec.name for token in args.match)]
    statuses = {spec.name: run_status(spec) for spec in runs}
    print(f"study {study.name}: {len(runs)} runs  {dict(Counter(statuses.values()))}")
    pending = [spec for spec in runs if statuses[spec.name] not in ("completed", "blocked")]
    blocked = [spec for spec in runs if statuses[spec.name] == "blocked"]
    if args.status or args.dry_run:
        measured = Counter(efficiency_status(spec) for spec in runs if statuses[spec.name] == "completed")
        print(f"efficiency of completed runs: {dict(measured)}")
        for spec in pending + blocked:
            print(f"  {statuses[spec.name]:<12} {spec.name}")
        return
    replaced = [spec.name for spec in pending if statuses[spec.name] in ("stale", "outdated")]
    if replaced and not args.force:
        raise SystemExit(
            f"{len(replaced)} runs hold another config or were produced by other code; "
            f"inspect them or pass --force: {replaced[:3]}"
        )
    if blocked:
        print(f"skipping {len(blocked)} blocked runs whose semantic artifact is not built yet", flush=True)

    devices = args.devices or detect_devices()
    print(f"devices: {' '.join(devices)}  ({args.per_device} training run(s) per device)", flush=True)
    log_dir = ROOT / "outputs" / "logs" / study.name
    log_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    if pending:
        sizes = warm_up(pending, devices)
        jobs = [
            Job(
                spec,
                sizes[spec.dataset] * (3.0 if spec.config["method"]["name"] == "prose" else 1.0),
                log_dir / f"{spec.name}.log",
            )
            for spec in pending
        ]
        jobs.sort(key=lambda job: -job.weight)
        failures += execute_all(
            f"{study.name} · train",
            jobs,
            [device for _ in range(args.per_device) for device in devices],
            lambda spec, device: command(spec, device, args.force),
        )

    if not args.no_efficiency:
        to_measure = [
            spec for spec in runs if run_status(spec) == "completed" and efficiency_status(spec) == "pending"
        ]
        if to_measure:
            print(f"efficiency phase: {len(to_measure)} runs, one process per device", flush=True)
            failures += execute_all(
                f"{study.name} · efficiency",
                [Job(spec, 1.0, log_dir / f"{spec.name}.efficiency.log") for spec in to_measure],
                list(dict.fromkeys(devices)),
                lambda spec, device: command(spec, device, False, efficiency=True),
            )
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; see {log_dir}: {failures[:5]}")


if __name__ == "__main__":
    main()
