"""分层实验配置、实验清单展开与运行身份。

一次运行由 (dataset, backbone, method, seed, overrides) 完整定义。
解析后的配置合并以下四层：

    configs/datasets/<dataset>.yaml   数据、划分、评测组与语义产物
    configs/backbones.yaml            主干超参数
    configs/methods.yaml              base / prose
    configs/training.yaml             优化器、学习率调度、早停与效率计时

运行目录只取决于运行定义，与实验清单无关，因此不同实验清单可以复用已完成的运行。
覆盖值与默认值相同时，不改变运行目录。

解析后的配置还包含数据文件的内容指纹（``dataset.fingerprint``）和语义产物的内容指纹
（``method.artifact_sha256``）；在相同路径下替换任一内容，都会使旧运行变为 stale。
训练代码单独计算指纹（``implementation_sha256``）；使用其他代码完成的运行标为
outdated，不会被直接复用。
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

from prose.data import dataset_fingerprint
from prose.io import ROOT, load_yaml, resolve_path, sha256_json, sha256_text
from prose.semantic_models import load_registry, source_name

CONFIG_DIR = ROOT / "configs"
RUN_ROOT = ROOT / "outputs" / "runs"
OVERRIDE_SECTIONS = ("method", "backbone", "training", "efficiency")
# efficiency 只设置已完成运行的计时方式，不属于运行身份；
# 覆盖该部分不会改变运行目录。
MEASUREMENT_SECTIONS = ("efficiency",)
SEMANTIC_MODELS = {"llm": "llm.yaml", "embedding": "embedding.yaml"}  # 方法字段 -> 模型登记文件
RUNTIME_KEYS = ("device",)
SOURCE_DIR = ROOT / "src" / "prose"
OFFLINE_ONLY = ("semantic_cards.py", "task_spec.py")  # LLM 离线预处理，不参与训练运行


@cache
def implementation_sha256(source_dir: Path = SOURCE_DIR) -> str:
    """去除文档字符串后计算运行代码的 AST 哈希；注释与排版不影响哈希。

    结果按进程缓存；运行启动时在 ``run_status`` 中计算，运行期间的代码修改不会写入其记录。
    """

    digest = []
    for path in sorted(source_dir.rglob("*.py")):
        if path.name in OFFLINE_ONLY:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
        digest.append(f"{path.relative_to(source_dir)}:{ast.dump(tree)}")
    return sha256_text("\n".join(digest))


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    backbone: str
    method: str
    seed: int
    overrides: tuple[tuple[str, Any], ...]
    config: dict[str, Any]

    @property
    def variant(self) -> str:
        # 名称中的 "/"（如 vendor/model-1）会被解释为子目录
        return "__".join(
            [self.method, *(f"{key}={value}".replace("/", "-") for key, value in self.overrides)]
        )

    @property
    def name(self) -> str:
        return f"{self.dataset}.{self.backbone}.{self.variant}.seed_{self.seed}"

    @property
    def run_dir(self) -> Path:
        root = resolve_path(self.config["output_root"])
        return root / self.dataset / self.backbone / self.variant / f"seed_{self.seed}"

    @property
    def config_sha256(self) -> str:
        return config_sha256(self.config)

    @property
    def pairing_sha256(self) -> str:
        """方法以外的全部共享设置；Base 与其他方法的配对运行必须保持一致。"""

        shared = {key: self.config[key] for key in ("dataset", "backbone", "training")}
        return config_sha256(shared)


def config_sha256(config: dict[str, Any]) -> str:
    payload = copy.deepcopy(config)
    for key in RUNTIME_KEYS:
        payload.get("training", {}).pop(key, None)
    payload.pop("output_root", None)
    for section in MEASUREMENT_SECTIONS:
        payload.pop(section, None)
    return sha256_json(payload)


def efficiency_settings_sha256(config: dict[str, Any]) -> str:
    return sha256_json(config.get("efficiency", {}))


def _resolve_inherit(table: dict[str, Any], key: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
    if key not in table:
        raise KeyError(f"unknown entry {key!r}; available: {sorted(table)}")
    if key in seen:
        raise ValueError(f"inherit cycle: {' -> '.join((*seen, key))}")
    entry = dict(table[key])
    parent = entry.pop("inherit", None)
    return {**_resolve_inherit(table, parent, (*seen, key)), **entry} if parent else entry


def dataset_config(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    return load_yaml(config_dir / "datasets" / f"{name}.yaml")


def resolve_run(
    dataset: str,
    backbone: str,
    method: str,
    seed: int,
    overrides: dict[str, Any] | None = None,
    *,
    config_dir: Path = CONFIG_DIR,
    output_root: Path = RUN_ROOT,
) -> RunSpec:
    data = dataset_config(dataset, config_dir)
    training = load_yaml(config_dir / "training.yaml")
    default = {
        "dataset": {key: value for key, value in data.items() if key not in ("training", "artifacts")},
        "backbone": copy.deepcopy(_resolve_inherit(load_yaml(config_dir / "backbones.yaml"), backbone)),
        "method": _semantic_fields(
            copy.deepcopy(_resolve_inherit(load_yaml(config_dir / "methods.yaml"), method)), config_dir
        ),
        "training": {**training.get("training", {}), **data.get("training", {}), "seed": int(seed)},
        "efficiency": training.get("efficiency", {}),
        "output_root": str(output_root),
    }
    config = copy.deepcopy(default)
    effective: list[tuple[str, Any]] = []
    for key, value in sorted((overrides or {}).items()):
        section, _, field = key.partition(".")
        if section not in OVERRIDE_SECTIONS or not field:
            raise ValueError(f"override {key!r} must look like <{'|'.join(OVERRIDE_SECTIONS)}>.<field>")
        if field not in config[section]:
            raise KeyError(f"override {key!r} does not exist in the {method} config")
        if section == "method" and field in SEMANTIC_MODELS:
            value = load_registry(config_dir / SEMANTIC_MODELS[field]).resolve(value)
        config[section][field] = value
        if default[section][field] != value and section not in MEASUREMENT_SECTIONS:
            effective.append((key, value))
    config["method"] = _attach_artifact(
        _semantic_fields(config["method"], config_dir), data.get("artifacts", {}), dataset
    )
    config["dataset"]["fingerprint"] = dataset_fingerprint(config["dataset"])
    return RunSpec(dataset, backbone, method, int(seed), tuple(effective), config)


def _semantic_fields(method: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """明确 PROSE 方法的语义来源。

    没有跨变量项（``coupling: none``）时不读取语义坐标，也不使用语义产物。
    LLM 语义明确记录 LLM 和编码器名称；``default`` 解析为登记表中的默认模型，
    因而按具体模型名扫描时可复用默认运行。其他来源不携带模型字段。
    """

    if method.get("name") != "prose":
        return method
    method = dict(method)
    if method.get("coupling", "semantic") == "none":
        method["semantics"] = "none"
    if method.get("semantics") == "llm":
        for field, registry in SEMANTIC_MODELS.items():
            method[field] = load_registry(config_dir / registry).resolve(method.get(field, "default"))
    else:
        for field in SEMANTIC_MODELS:
            method.pop(field, None)
    return method


def llm_artifact_dir(root: str | Path, llm: str, embedding: str) -> Path:
    return Path(root) / source_name(llm, embedding)


def _attach_artifact(method: dict[str, Any], artifacts: dict[str, Any], dataset: str) -> dict[str, Any]:
    if method["name"] == "prose" and method["semantics"] != "none":
        key = f"prose_{method['semantics']}"
    else:
        return method
    if key not in artifacts:
        raise KeyError(f"dataset {dataset} declares no artifact {key!r}")
    location = Path(artifacts[key])
    if key == "prose_llm":  # 每个 (LLM, embedding) 模型对各有一个子目录的根路径
        location = llm_artifact_dir(location, method["llm"], method["embedding"])
    return {**method, "artifact": str(location), "artifact_sha256": artifact_sha256(resolve_path(location))}


def artifact_sha256(path: Path) -> str | None:
    """PROSE 产物数组的内容哈希。"""

    if (path / "metadata.json").exists():
        return json.loads((path / "metadata.json").read_text(encoding="utf-8")).get("arrays_sha256")
    return None  # 尚未构建；运行加载该产物时会失败


@dataclass(frozen=True)
class Study:
    name: str
    runs: tuple[RunSpec, ...]
    raw: dict[str, Any]
    sweeps: dict[str, list[tuple[str, Any]]]  # 运行名称 -> 所属的扫描 (key, value) 条目


def load_study(path: Path, *, config_dir: Path = CONFIG_DIR, output_root: Path | None = None) -> Study:
    """展开 datasets x backbones x methods x seeds。

    ``llms`` 与 ``embeddings``（可省略，默认 ``[default]``）只作用于使用 LLM 语义的方法，
    这些方法再展开 llms x embeddings；其他方法各只有一次运行。
    ``overrides`` 应用于全部运行；``sweep`` 每次只改变一个字段，
    不包含该字段的方法会跳过相应扫描。
    """

    raw = load_yaml(path)
    if output_root is None:
        output_root = resolve_path(raw.get("output_root", RUN_ROOT))
    fixed = raw.get("overrides") or {}
    sweep = raw.get("sweep") or {}
    sources = [
        {"method.llm": llm, "method.embedding": embedding}
        for llm, embedding in product(raw.get("llms") or ["default"], raw.get("embeddings") or ["default"])
    ]
    runs: dict[str, RunSpec] = {}
    sweeps: dict[str, list[tuple[str, Any]]] = {}
    for dataset, backbone, method, seed in product(
        raw["datasets"], raw["backbones"], raw["methods"], raw["seeds"]
    ):
        plain = resolve_run(
            dataset, backbone, method, seed, fixed, config_dir=config_dir, output_root=output_root
        )
        uses_llm = "llm" in plain.config["method"]
        for overrides in [{**fixed, **source} for source in sources] if uses_llm else [fixed]:
            _expand(runs, sweeps, dataset, backbone, method, seed, overrides, sweep, config_dir, output_root)
    return Study(
        name=str(raw["name"]),
        runs=tuple(runs.values()),
        raw=raw,
        sweeps=sweeps,
    )


def _expand(
    runs: dict[str, RunSpec],
    sweeps: dict[str, list[tuple[str, Any]]],
    dataset: str,
    backbone: str,
    method: str,
    seed: int,
    fixed: dict[str, Any],
    sweep: dict[str, list[Any]],
    config_dir: Path,
    output_root: Path,
) -> None:
    """加入一次运行及其各扫描点；同名运行只保留一份并记录所属扫描。"""

    base = resolve_run(dataset, backbone, method, seed, fixed, config_dir=config_dir, output_root=output_root)
    candidates: list[tuple[RunSpec, tuple[str, Any] | None]] = [(base, None)]
    for key, values in sweep.items():
        section, _, field = key.partition(".")
        if field not in base.config.get(section, {}):
            continue
        for value in values:
            spec = resolve_run(
                dataset,
                backbone,
                method,
                seed,
                {**fixed, key: value},
                config_dir=config_dir,
                output_root=output_root,
            )
            candidates.append((spec, (key, value)))
    for spec, swept in candidates:
        runs.setdefault(spec.name, spec)
        memberships = sweeps.setdefault(spec.name, [])
        if swept is not None and swept not in memberships:
            memberships.append(swept)
