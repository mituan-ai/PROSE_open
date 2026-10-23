from __future__ import annotations

from pathlib import Path

import pytest

from prose.config import resolve_run
from prose.io import read_csv, read_json
from prose.training.runner import run, run_status


@pytest.mark.parametrize("method", ["base", "prose_random_semantics", "prose_time_index"])
def test_run_writes_outputs_and_resumes(method: str, toy_config_dir: Path, tmp_path: Path) -> None:
    spec = resolve_run("toy", "gru", method, 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    assert run_status(spec) == "not_started"
    run_dir = run(spec, log=lambda _: None)
    for name in (
        "config.resolved.yaml",
        "history.csv",
        "best.pt",
        "predictions.csv",
        "metrics.csv",
        "summary.json",
    ):
        assert (run_dir / name).exists(), name
    summary = read_json(run_dir / "summary.json")
    assert summary["config_sha256"] == spec.config_sha256 and summary["epochs_run"] == 2
    groups = {(row["group_type"], row["group"]) for row in read_csv(run_dir / "metrics.csv")}
    assert ("overall", "test") in groups and ("domain", "far") in groups
    assert run_status(spec) == "completed"

    messages: list[str] = []
    run(spec, log=messages.append)
    assert messages and messages[0].startswith("[skip]")

    spec.config["training"]["learning_rate"] = 0.01
    assert run_status(spec) == "stale"
    with pytest.raises(RuntimeError, match="another config"):
        run(spec, log=lambda _: None)


def test_reported_best_epoch_is_the_saved_checkpoint(toy_config_dir: Path, tmp_path: Path) -> None:
    import torch

    from prose.data import Normalization, build_split, load_dataset
    from prose.models.forecaster import build_forecaster
    from prose.training.evaluation import regression_metrics
    from prose.training.runner import _batches, predict

    spec = resolve_run(
        "toy",
        "gru",
        "prose_random_semantics",
        2024,
        {"training.max_epochs": 6},
        config_dir=toy_config_dir,
        output_root=tmp_path / "runs",
    )
    run_dir = run(spec, log=lambda _: None)
    summary = read_json(run_dir / "summary.json")
    data = load_dataset(spec.config["dataset"])
    split = build_split(data, spec.config["dataset"])
    norm = Normalization.fit(data.windows[split.train], data.targets[split.train])
    model = build_forecaster(
        method=spec.config["method"],
        backbone=spec.config["backbone"],
        feature_names=data.feature_names,
        window_size=data.window_size,
        output_dim=1,
    )
    model.load_state_dict(torch.load(run_dir / "best.pt"))
    _, target, prediction = predict(
        model, _batches(data, split.validation, norm, 16, torch.device("cpu")), norm, torch.device("cpu")
    )
    assert abs(regression_metrics(target, prediction)["RMSE"] - summary["best_val_RMSE"]) < 1e-9


def test_efficiency_is_measured_separately_and_tracks_the_run(toy_config_dir: Path, tmp_path: Path) -> None:
    from prose.training.runner import efficiency_status, measure

    spec = resolve_run(
        "toy", "gru", "prose_random_semantics", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs"
    )
    with pytest.raises(RuntimeError, match="not a completed run"):
        measure(spec, log=lambda _: None)
    run(spec, log=lambda _: None)
    assert efficiency_status(spec) == "pending"
    measure(spec, log=lambda _: None)
    record = read_json(spec.run_dir / "efficiency.json")
    for key in ("inference_ms_per_sample", "latency_ms_median", "train_ms_per_sample", "parameters"):
        assert record[key] > 0
    assert efficiency_status(spec) == "measured"
    changed = resolve_run(
        "toy",
        "gru",
        "prose_random_semantics",
        2024,
        {"efficiency.latency_samples": 5},
        config_dir=toy_config_dir,
        output_root=tmp_path / "runs",
    )
    assert changed.run_dir == spec.run_dir and changed.config_sha256 == spec.config_sha256
    assert efficiency_status(changed) == "pending"


def test_a_run_directory_is_locked(toy_config_dir: Path, tmp_path: Path) -> None:
    from prose.training.runner import _run_lock

    spec = resolve_run("toy", "gru", "base", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    with _run_lock(spec.run_dir), pytest.raises(RuntimeError, match="in use"):
        run(spec, log=lambda _: None)


def test_runs_from_other_code_are_outdated(toy_config_dir: Path, tmp_path: Path, monkeypatch) -> None:
    import prose.training.runner as runner

    spec = resolve_run("toy", "gru", "base", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    run(spec, log=lambda _: None)
    monkeypatch.setattr(runner, "implementation_sha256", lambda: "other")
    assert run_status(spec) == "outdated"
    with pytest.raises(RuntimeError, match="other code"):
        run(spec, log=lambda _: None)


def test_device_batches_follow_the_dataloader_order() -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from prose.training.runner import Batches

    data = torch.arange(23.0)
    loader = DataLoader(
        TensorDataset(data), batch_size=5, shuffle=True, generator=torch.Generator().manual_seed(7)
    )
    batches = Batches((data,), 5, seed=7)
    for _ in range(3):
        assert [b[0].tolist() for b in loader] == [b[0].tolist() for b in batches]


def test_evaluation_random_state_is_fixed_and_isolated() -> None:
    import torch

    from prose.training.runner import _evaluation_rng

    torch.manual_seed(1)
    expected_next = torch.rand(1)
    torch.manual_seed(1)
    with _evaluation_rng(torch.device("cpu")):
        first = torch.rand(2)
    with _evaluation_rng(torch.device("cpu")):
        second = torch.rand(2)
    assert torch.equal(first, second) and torch.equal(torch.rand(1), expected_next)


def test_runs_without_their_semantic_artifact_are_blocked(toy_config_dir: Path, tmp_path: Path) -> None:
    spec = resolve_run("toy", "gru", "prose", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    assert run_status(spec) == "blocked"
    with pytest.raises(RuntimeError, match="not built yet"):
        run(spec, log=lambda _: None)
    no_field = resolve_run(
        "toy", "gru", "prose_no_field", 2024, config_dir=toy_config_dir, output_root=tmp_path
    )
    assert run_status(no_field) == "not_started"


def test_epoch_log_matches_the_scheduler_progress_pattern(
    toy_config_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_study.py"
    module_spec = importlib.util.spec_from_file_location("run_study", path)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, "run_study", module)  # dataclass 需要按模块名查找
    module_spec.loader.exec_module(module)
    spec = resolve_run("toy", "gru", "base", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    messages: list[str] = []
    run(spec, log=messages.append)
    matches = [module.PROGRESS.search(message) for message in messages]
    assert [match[1] for match in matches if match] == ["1", "2"]
