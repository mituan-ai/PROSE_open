from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prose.config import dataset_config
from prose.data import read_feature_names
from prose.semantic_artifact import build_random, center_normalize, load_artifact
from prose.semantic_cards import (
    PIPELINE_VERSION,
    LLMConfig,
    LLMReply,
    build_prompt,
    dataset_materials,
    generate_cards,
    load_cards,
    load_materials,
    save_cards,
    validate_cards,
)

DATASETS = ("ladle_preheating", "thickener_dewatering", "indpensim")


def test_center_normalize_removes_the_common_component() -> None:
    values = np.random.default_rng(0).normal(size=(5, 7)) + 3.0
    coordinates = center_normalize(values)
    assert np.allclose(np.linalg.norm(coordinates, axis=1), 1.0)
    assert np.allclose(center_normalize(values + 10.0), coordinates)


def test_artifact_roundtrip_and_order_check(tmp_path: Path) -> None:
    build_random(("a", "b", "c"), dim=4, seed=1).save(tmp_path / "artifact")
    artifact = load_artifact(tmp_path / "artifact", ("a", "b", "c"))
    assert artifact.history.shape == (3, 4)
    with pytest.raises(ValueError, match="feature order"):
        load_artifact(tmp_path / "artifact", ("b", "a", "c"))
    with pytest.raises(FileExistsError):
        build_random(("a", "b", "c"), dim=4, seed=1).save(tmp_path / "artifact")


def test_degenerate_semantics_are_rejected() -> None:
    artifact = build_random(("a", "b", "c", "d"), dim=5, seed=1)
    raw = artifact.history_raw.copy()
    raw[3] = raw[0] + raw[1] - raw[2]  # 四点共面，中心化后的秩为 2 < 3
    broken = type(artifact)(
        artifact.variable_names, center_normalize(raw), artifact.condition, raw, artifact.condition_raw
    )
    with pytest.raises(ValueError, match="degenerate"):
        broken.validate()


def _card(name: str, suffix: str = "") -> dict:
    return {
        "variable_name": name,
        "history_view": f"History of {name} records accumulated load{suffix}.",
        "condition_view": f"A change of {name} shifts the operating state{suffix}.",
        "rationale": {"basis": "process_knowledge", "evidence": "general process knowledge"},
    }


def test_card_validation() -> None:
    cards = validate_cards({"cards": [_card("a"), _card("b")]}, ("a", "b"))
    assert [card["variable_name"] for card in cards] == ["a", "b"]
    with pytest.raises(ValueError, match="order"):
        validate_cards({"cards": [_card("b"), _card("a")]}, ("a", "b"))
    bad = _card("b")
    bad["history_view"] = "压力历史"
    with pytest.raises(ValueError, match="English"):
        validate_cards({"cards": [_card("a"), bad]}, ("a", "b"))
    bad = _card("b")
    bad["condition_view"] = "Unlike the previous column, it changes the phase."
    with pytest.raises(ValueError, match="column order"):
        validate_cards({"cards": [_card("a"), bad]}, ("a", "b"))


def test_card_quotes_must_come_from_the_materials() -> None:
    cited = _card("b")
    cited["rationale"] = {"basis": "task_document", "evidence": "“Pressure  rises during filling。”"}
    sources = "Background: pressure rises during filling. Other text."
    assert validate_cards({"cards": [_card("a"), cited]}, ("a", "b"), sources=sources)
    cited["rationale"]["evidence"] = "Pressure falls during filling."
    with pytest.raises(ValueError, match="not in the materials"):
        validate_cards({"cards": [_card("a"), cited]}, ("a", "b"), sources=sources)


def _materials(tmp_path: Path, document: str = "# Process\n\nPressure rises during filling.\n"):
    import json

    spec = {
        "dataset": "toy",
        "target": "y",
        "horizon": "now",
        "online_boundary": "inputs only",
        "process_background": "A toy process.",
        "split_description": "SPLIT",
        "variables": [{"name": "a"}, {"name": "b"}],
    }
    (tmp_path / "task_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "PROCESS.md").write_text(document, encoding="utf-8")
    return load_materials(tmp_path, protocol_terms=("process_13",))


@pytest.mark.parametrize(
    "leak", ["测试集包含高负荷工况。", "Batch far_fault_091 drifts.", "过程 process_13 较长。"]
)
def test_process_document_rejects_split_information(tmp_path: Path, leak: str) -> None:
    with pytest.raises(ValueError, match="split or evaluation"):
        _materials(tmp_path, f"# Process\n\n{leak}\n")


def _reply(text: str, finish: str | None = "stop") -> LLMReply:
    return LLMReply(text, "m-0101", finish, {"prompt_tokens": 10, "completion_tokens": 5}, 0.5)


def _cards_json(suffix: str) -> str:
    import json

    return json.dumps({"cards": [_card("a", suffix), _card("b", suffix)]})


def test_pipeline_drafts_three_times_reviews_once_and_records_failures(tmp_path: Path) -> None:
    materials = _materials(tmp_path)
    config = LLMConfig("m", "http://x", "KEY")
    replies = iter(
        [
            _reply(_cards_json(" one")),
            _reply("", "length"),  # 截断：记录并重试
            _reply(_cards_json(" two")),
            _reply("not json"),
            _reply("still not json"),  # 第三次归纳两次尝试均失败，不进入复核
            _reply(_cards_json(" final")),
        ]
    )
    prompts = []

    def caller(prompt, _config):
        prompts.append(prompt)
        return next(replies)

    cards, report, files = generate_cards(materials, config, caller=caller)
    assert [card["history_view"] for card in cards] == [
        "History of a records accumulated load final.",
        "History of b records accumulated load final.",
    ]
    assert report["pipeline_version"] == PIPELINE_VERSION and report["status"] == "accepted"
    assert [run["status"] for run in report["extraction_runs"]] == ["accepted", "accepted", "failed"]
    assert report["extraction_runs"][1]["attempts"][0]["error"].endswith("finish_reason=length")
    assert report["review"]["draft_runs"] == [1, 2] and report["calls"] == 6
    assert report["usage"]["prompt_tokens"] == 60
    assert "load one" in prompts[-1] and "load two" in prompts[-1]
    assert set(files) == {"prompt.txt", "review_prompt.txt"}


def test_saved_cards_report_stale_materials(tmp_path: Path) -> None:
    materials = _materials(tmp_path)
    config = LLMConfig("m", "http://x", "KEY")
    cards, report, _ = generate_cards(materials, config, caller=lambda *_: _reply(_cards_json("")))
    assert report["pipeline_version"] == PIPELINE_VERSION and report["status"] == "accepted"
    path = tmp_path / "out" / "semantic_cards.json"
    save_cards(path, materials=materials, cards=cards, llm_config=config, report=report)
    payload = load_cards(path, materials)
    assert not payload["edited_after_generation"] and payload["llm"]["response_model"] == "m-0101"
    assert "api_key_env" not in payload["llm"] and "base_url" not in payload["llm"]
    (tmp_path / "PROCESS.md").write_text("# Process\n\nPressure falls during discharge.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        load_cards(path, load_materials(tmp_path))


@pytest.mark.parametrize("dataset", DATASETS)
def test_task_specs_and_materials_match_dataset_variables(dataset: str) -> None:
    config = dataset_config(dataset)
    names = read_feature_names(config)
    materials = dataset_materials(config)  # PROCESS.md 通过划分与场景信息检查
    task = materials.task
    assert task.feature_names == names
    prompt = build_prompt(materials)
    assert all(f"- name: {name}\n" in prompt for name in names)
    assert materials.document.strip() in prompt
    assert "split_description" not in task.as_dict()
