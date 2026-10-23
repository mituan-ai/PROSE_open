"""生成语义归纳提示词；指定 --call-api 时运行 LLM 语义生成流程。

    # 仅生成提示词供审阅：outputs/semantics/<dataset>/prompt_review/prompt.txt
    uv run python scripts/build_semantic_cards.py --dataset all
    # 默认 LLM（configs/llm.yaml）的语义卡：outputs/semantics/<dataset>/<llm>/
    uv run --extra llm python scripts/build_semantic_cards.py --dataset all --call-api
    # 全部登记的 LLM；未配置密钥的模型会跳过
    uv run --extra llm python scripts/build_semantic_cards.py --dataset all --llm all --call-api

材料为数据集目录中的 task_spec.json 与 PROCESS.md。流程为三次独立归纳、一次复核定稿和程序校验
（见 prose.semantic_cards）。已有语义卡与当前材料、提示词和流程版本一致时保留；不一致时报告
stale 并失败，需用 --force 重建。失败的生成记录（含原始回答）写入 outputs/logs/semantics/。
单个模型失败不会中断其余任务；汇总列出每个 (dataset, LLM) 的结果，有任一失败时退出码为 1。
"""

from __future__ import annotations

import argparse

from prose.config import CONFIG_DIR, dataset_config
from prose.data import read_feature_names
from prose.io import ROOT, load_api_env_files, resolve_path, write_json
from prose.semantic_cards import (
    LLMConfig,
    build_prompt,
    call_llm,
    cards_path,
    dataset_materials,
    generate_cards,
    load_cards,
    save_cards,
)
from prose.semantic_models import load_registry, model_slug

DATASETS = ("ladle_preheating", "thickener_dewatering", "indpensim")


def build_one(dataset: str, llm: str, registry, *, force: bool, caller=call_llm) -> str:
    materials = dataset_materials(dataset_config(dataset))
    if materials.task.feature_names != read_feature_names(dataset_config(dataset)):
        raise ValueError(f"task_spec.json variables of {dataset} do not match the dataset")
    path = cards_path(dataset, llm)
    if path.exists() and not force:
        load_cards(path, materials)
        return "exists"
    if not registry.configured(llm):
        return "not configured"
    config = LLMConfig.from_mapping(registry.entry(llm))
    cards, report, prompts = generate_cards(materials, config, caller=caller)
    if cards is None:
        log = ROOT / "outputs" / "logs" / "semantics" / f"{dataset}.{model_slug(llm)}.cards.json"
        write_json(log, {"dataset": dataset, "llm": llm, "generation": report})
        raise RuntimeError(
            f"generation failed after {report['calls']} calls; record in {log.relative_to(ROOT)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, text in prompts.items():
        (path.parent / name).write_text(text + "\n", encoding="utf-8")
    save_cards(path, materials=materials, cards=cards, llm_config=config, report=report)
    usage = report["usage"]
    return (
        f"built {len(cards)} cards ({report['calls']} calls, {report['seconds']:.0f}s, "
        f"{usage['prompt_tokens']} in / {usage['completion_tokens']} out tokens)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", nargs="+", required=True, help="dataset names or 'all'")
    parser.add_argument("--llm", default="default", help="a model of configs/llm.yaml, 'default', or 'all'")
    parser.add_argument("--llm-config", default=str(CONFIG_DIR / "llm.yaml"))
    parser.add_argument(
        "--call-api", action="store_true", help="call the LLM; otherwise write the prompt only"
    )
    parser.add_argument("--force", action="store_true", help="regenerate existing cards")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == ["all"] else tuple(args.dataset)
    if not args.call_api:
        for dataset in datasets:
            materials = dataset_materials(dataset_config(dataset))
            output = ROOT / "outputs" / "semantics" / dataset / "prompt_review"
            output.mkdir(parents=True, exist_ok=True)
            (output / "prompt.txt").write_text(build_prompt(materials) + "\n", encoding="utf-8")
            print(output / "prompt.txt")
        return

    load_api_env_files()
    registry = load_registry(resolve_path(args.llm_config))
    outcomes = []
    for dataset in datasets:
        for llm in registry.select(args.llm):
            try:
                outcome = build_one(dataset, llm, registry, force=args.force)
            except Exception as error:  # 继续处理其余模型，错误在汇总中报告
                outcome = f"FAILED {type(error).__name__}: {str(error)[:300]}"
            print(f"{dataset:<22} {llm:<24} {outcome}", flush=True)
            outcomes.append(outcome)
    print(
        f"summary: {sum(o.startswith('built') for o in outcomes)} built, {outcomes.count('exists')} kept, "
        f"{outcomes.count('not configured')} not configured, "
        f"{sum(o.startswith('FAILED') for o in outcomes)} failed"
    )
    if any(outcome.startswith("FAILED") for outcome in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
