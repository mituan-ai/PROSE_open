"""构建冻结的 PROSE 语义坐标。

    # 用默认编码器编码默认 LLM 的语义卡
    uv run --extra llm python scripts/build_semantic_artifact.py --dataset all --source llm
    # 登记表中的全部 LLM 配默认编码器，或全部编码器配默认 LLM
    uv run --extra llm python scripts/build_semantic_artifact.py --dataset all --source llm --llm all
    uv run --extra llm python scripts/build_semantic_artifact.py --dataset all --source llm --embedding all
    # 按编码器原生支持的其他维度重新编码，写入 <llm>__<embedding>__k<dim>/
    uv run --extra llm python scripts/build_semantic_artifact.py --dataset all --source llm --dim 512
    # 冻结随机坐标（对照），无需 API
    uv run python scripts/build_semantic_artifact.py --dataset all --source random

LLM 语义坐标保存到 <artifacts.prose_llm>/<llm>__<embedding>/，读取 outputs/semantics/<dataset>/<llm>/
中的语义卡。
除非指定 --force，否则保留现有有效产物；语义卡、材料或编码器配置变化后，旧产物报告 stale 并失败。
缺少语义卡或未配置编码器的模型对会跳过。
单个模型对失败不会中断其余任务；有任一失败时退出码为 1。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from prose.config import CONFIG_DIR, dataset_config, llm_artifact_dir
from prose.data import read_feature_names
from prose.io import ROOT, load_api_env_files, resolve_path, sha256_json
from prose.semantic_artifact import (
    SOURCES,
    EmbeddingConfig,
    build_from_cards,
    build_random,
    load_artifact,
)
from prose.semantic_cards import cards_path, dataset_materials, load_cards
from prose.semantic_models import load_registry

DATASETS = ("ladle_preheating", "thickener_dewatering", "indpensim")


def prepare(output: Path, names: tuple[str, ...], force: bool, *, dim: int) -> bool:
    """需要构建产物时返回 True；已有有效产物则保留。"""

    if output.exists() and any(output.iterdir()):
        if not force:
            artifact = load_artifact(output, names)
            if artifact.dim != dim:
                raise ValueError(
                    f"existing artifact has {artifact.dim} dimensions, expected {dim}; use --force"
                )
            return False
        shutil.rmtree(output)
    return True


def build_llm(dataset, names, llm, embedding, registry, *, force: bool, dim: int | None = None) -> str:
    """``dim`` 不同于登记维度时，按该维度向接口请求编码，产物目录追加 ``__k<dim>``。"""

    config = dataset_config(dataset)
    root = Path(config["artifacts"]["prose_llm"])
    output = resolve_path(llm_artifact_dir(root, llm, embedding))
    cards = cards_path(dataset, llm)
    if not cards.exists():
        return "no cards"
    if not registry.configured(embedding):
        return "not configured"
    entry = registry.entry(embedding)
    if dim is not None and dim != int(entry["dimensions"]):
        entry = {**entry, "dimensions": dim}
        output = output.with_name(f"{output.name}__k{dim}")
    encoder = EmbeddingConfig.from_mapping(entry)
    payload = load_cards(cards, dataset_materials(config))  # 语义卡与当前材料不一致时在此报告 stale
    if not force and output.exists() and any(output.iterdir()):
        metadata = load_artifact(output, names).metadata
        if (
            metadata.get("cards_sha256") != sha256_json(payload["cards"])
            or metadata.get("embedding") != encoder.public()
        ):
            raise ValueError(
                f"stale artifact {output.relative_to(ROOT)}: cards or encoder changed; use --force"
            )
    if not prepare(output, names, force, dim=encoder.dimensions):
        return "exists"
    artifact = build_from_cards(payload, cards_path=cards, embedding=encoder)
    artifact.validate(names)
    artifact.save(output)
    return f"built {output.relative_to(ROOT)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", nargs="+", required=True, help="dataset names or 'all'")
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument(
        "--llm", default="default", help="source llm: a model of configs/llm.yaml, 'default', or 'all'"
    )
    parser.add_argument(
        "--embedding", default="default", help="source llm: an embedding model, 'default', or 'all'"
    )
    parser.add_argument("--embedding-config", default=str(CONFIG_DIR / "embedding.yaml"))
    parser.add_argument("--llm-config", default=str(CONFIG_DIR / "llm.yaml"))
    parser.add_argument(
        "--dim",
        type=int,
        help="semantic dimension k (default: embedding registry dimensions); "
        "source llm writes non-default dimensions to <llm>__<embedding>__k<dim>/",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (source random)")
    parser.add_argument("--force", action="store_true", help="rebuild existing artifacts")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == ["all"] else tuple(args.dataset)
    outcomes = []
    encoders = load_registry(resolve_path(args.embedding_config))
    random_dim = args.dim if args.dim is not None else int(encoders.entries[encoders.default]["dimensions"])
    if args.source == "llm":
        load_api_env_files()
        llms = load_registry(resolve_path(args.llm_config))
        pairs = [
            (llm, embedding) for llm in llms.select(args.llm) for embedding in encoders.select(args.embedding)
        ]
    for dataset in datasets:
        config = dataset_config(dataset)
        names = read_feature_names(config)
        jobs = pairs if args.source == "llm" else [(None, None)]
        for llm, embedding in jobs:
            label = f"{dataset:<22} {args.source if llm is None else f'{llm} + {embedding}':<48}"
            try:
                if args.source == "llm":
                    outcome = build_llm(
                        dataset,
                        names,
                        llm,
                        embedding,
                        encoders,
                        force=args.force,
                        dim=args.dim,
                    )
                else:
                    output = resolve_path(config["artifacts"]["prose_random"])
                    if prepare(output, names, args.force, dim=random_dim):
                        artifact = build_random(names, dim=random_dim, seed=args.seed)
                        artifact.validate(names)
                        artifact.save(output)
                        outcome = f"built {output.relative_to(ROOT)}"
                    else:
                        outcome = "exists"
            except Exception as error:  # 继续处理其余模型对，错误在汇总中报告
                outcome = f"FAILED {type(error).__name__}: {str(error)[:200]}"
            print(f"{label} {outcome}", flush=True)
            outcomes.append(outcome)
    print(
        f"summary: {sum(o.startswith('built') for o in outcomes)} built, {outcomes.count('exists')} kept, "
        f"{outcomes.count('no cards') + outcomes.count('not configured')} skipped, "
        f"{sum(o.startswith('FAILED') for o in outcomes)} failed"
    )
    if any(outcome.startswith("FAILED") for outcome in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
