from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prose.config import load_study, resolve_run

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "configs" / "studies"


def test_resolved_run_layers_and_artifacts() -> None:
    spec = resolve_run("ladle_preheating", "gru", "prose", 2024)
    assert spec.config["training"]["batch_size"] == 128
    assert spec.config["training"]["seed"] == 2024
    assert spec.config["method"]["artifact"].endswith("ladle_preheating/llm/claude-opus-5-5__embed-v5.0-pro")
    assert spec.run_dir.parts[-4:] == ("ladle_preheating", "gru", "prose", "seed_2024")
    random = resolve_run("ladle_preheating", "gru", "prose", 2024, {"method.semantics": "random"})
    assert random.config["method"]["artifact"].endswith("ladle_preheating/random_seed0_256d")


def test_default_valued_override_keeps_the_run_identity() -> None:
    default = resolve_run("indpensim", "gru", "prose", 2024)
    same = resolve_run("indpensim", "gru", "prose", 2024, {"method.memory_dim": 16})
    changed = resolve_run("indpensim", "gru", "prose", 2024, {"method.memory_dim": 8})
    assert same.run_dir == default.run_dir and same.config_sha256 == default.config_sha256
    assert changed.run_dir.name == "seed_2024" and changed.variant == "prose__method.memory_dim=8"
    assert changed.pairing_sha256 == default.pairing_sha256


def test_runtime_fields_do_not_change_the_config_hash() -> None:
    spec = resolve_run("indpensim", "gru", "base", 2024)
    before = spec.config_sha256
    spec.config["training"]["device"] = "cpu"
    assert spec.config_sha256 == before


def test_invalid_overrides_are_rejected() -> None:
    with pytest.raises(KeyError):
        resolve_run("indpensim", "gru", "base", 2024, {"method.memory_dim": 8})
    with pytest.raises(ValueError):
        resolve_run("indpensim", "gru", "base", 2024, {"dataset.file": "x.npz"})


def test_study_expansion() -> None:
    smoke = load_study(STUDIES / "smoke.yaml")
    assert [spec.method for spec in smoke.runs] == ["base", "prose"]
    assert all("smoke_runs" in spec.run_dir.parts for spec in smoke.runs)


def test_data_and_artifact_contents_enter_the_run_identity(toy_config_dir: Path, tmp_path: Path) -> None:
    import os

    from prose.config import load_yaml

    before = resolve_run("toy", "gru", "prose_random_semantics", 2024, config_dir=toy_config_dir)
    assert before.config["method"]["artifact_sha256"]
    bundle = Path(load_yaml(toy_config_dir / "datasets" / "toy.yaml")["path"]) / "windows.npz"
    with np.load(bundle) as arrays:
        data = dict(arrays)
    data["windows"] = data["windows"] + 1.0
    np.savez_compressed(bundle, **data)
    os.utime(bundle, ns=(1, 1))
    after = resolve_run("toy", "gru", "prose_random_semantics", 2024, config_dir=toy_config_dir)
    assert after.config_sha256 != before.config_sha256
    assert after.pairing_sha256 != before.pairing_sha256


def test_implementation_fingerprint_ignores_docstrings_and_comments(tmp_path: Path) -> None:
    from prose.config import implementation_sha256

    source = tmp_path / "src"
    source.mkdir()
    (source / "a.py").write_text('"""doc"""\n\ndef f():\n    return 1\n', encoding="utf-8")
    first = implementation_sha256(source)
    (source / "a.py").write_text('"""other doc"""\n# note\n\ndef f():\n    """x"""\n    return 1\n')
    implementation_sha256.cache_clear()
    assert implementation_sha256(source) == first
    (source / "a.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    implementation_sha256.cache_clear()
    assert implementation_sha256(source) != first


def test_semantic_sources_resolve_to_model_names_and_reuse_the_default_run() -> None:
    from prose.semantic_models import load_registry

    llms = load_registry(ROOT / "configs" / "llm.yaml")
    default = resolve_run("indpensim", "gru", "prose", 2024)
    named = resolve_run("indpensim", "gru", "prose", 2024, {"method.llm": llms.default})
    alias = resolve_run("indpensim", "gru", "prose", 2024, {"method.embedding": "default"})
    assert named.run_dir == default.run_dir == alias.run_dir
    with pytest.raises(KeyError):
        resolve_run("indpensim", "gru", "prose", 2024, {"method.llm": "no-such-model"})
    random = resolve_run("indpensim", "gru", "prose", 2024, {"method.semantics": "random"}).config["method"]
    assert "llm" not in random and "embedding" not in random


def test_run_config_expands_semantic_sources_only_for_llm_methods(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: run\ndatasets: [indpensim]\nbackbones: [gru]\n"
        "methods: [base, prose]\nseeds: [2024]\n"
        "llms: [claude-opus-5-5]\nembeddings: [embed-v5.0-pro]\n",
        encoding="utf-8",
    )
    study = load_study(path)
    assert len(study.runs) == 1 + 1
    default = resolve_run("indpensim", "gru", "prose", 2024)
    assert default.name in {spec.name for spec in study.runs}
    assert load_study(ROOT / "configs" / "run.yaml").runs
