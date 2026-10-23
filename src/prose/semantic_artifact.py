"""冻结的过程时间语义 S = {(e^H_i, e^C_i)}。

产物目录包含 ``semantics.npz`` 与 ``metadata.json``。
``history`` 和 ``condition`` 保存 PROSE 使用的任务内中心化、L2 归一化坐标；
``history_raw`` 和 ``condition_raw`` 保留编码器原始输出。

来源：
- ``llm``：将 LLM 语义卡的两个视角分别用文本编码器编码；
- ``random``：冻结的随机坐标，作为不含语义内容的对照。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from prose.io import ROOT, sha256_json, write_json

ARTIFACT_TYPE = "prose_process_temporal_semantics"
SOURCES = ("llm", "random")


def center_normalize(embeddings: np.ndarray) -> np.ndarray:
    """去除任务内公共分量，再对每行作 L2 归一化。"""

    values = np.asarray(embeddings, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("a variable has no semantic difference from the task mean")
    return centered / norms


@dataclass(frozen=True)
class SemanticArtifact:
    variable_names: tuple[str, ...]
    history: np.ndarray
    condition: np.ndarray
    history_raw: np.ndarray
    condition_raw: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.history.shape[1])

    def validate(self, feature_names: tuple[str, ...] | None = None) -> None:
        d = len(self.variable_names)
        if d < 2:
            raise ValueError("PROSE needs at least two variables")
        for name in ("history", "condition", "history_raw", "condition_raw"):
            array = getattr(self, name)
            if array.ndim != 2 or array.shape[0] != d or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite ({d}, k) array")
        if self.history.shape != self.condition.shape:
            raise ValueError("history and condition coordinates must have the same shape")
        for name in ("history", "condition"):
            norms = np.linalg.norm(getattr(self, name), axis=1)
            if not np.allclose(norms, 1.0, atol=1e-6):
                raise ValueError(f"{name} coordinates must be unit vectors")
            centered = getattr(self, f"{name}_raw") - getattr(self, f"{name}_raw").mean(axis=0)
            expected = min(d - 1, centered.shape[1])
            if np.linalg.matrix_rank(centered, tol=1e-8) < expected:
                raise ValueError(f"{name} coordinates are degenerate: rank below {expected}")
        if feature_names is not None and tuple(feature_names) != self.variable_names:
            raise ValueError(
                f"feature order {list(feature_names)} does not match semantics {list(self.variable_names)}"
            )

    def save(self, output_dir: Path) -> None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"artifact directory is not empty: {output_dir}")
        self.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "history": self.history,
            "condition": self.condition,
            "history_raw": self.history_raw,
            "condition_raw": self.condition_raw,
        }
        np.savez(output_dir / "semantics.npz", **arrays)
        write_json(
            output_dir / "metadata.json",
            {
                **self.metadata,
                "artifact_type": ARTIFACT_TYPE,
                "variable_names": list(self.variable_names),
                "semantic_dim": self.dim,
                "arrays_sha256": sha256_json({key: np.round(value, 10) for key, value in arrays.items()}),
            },
        )


def load_artifact(artifact_dir: Path, feature_names: tuple[str, ...] | None = None) -> SemanticArtifact:
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"not a PROSE semantic artifact: {artifact_dir}")
    with np.load(artifact_dir / "semantics.npz") as arrays:
        artifact = SemanticArtifact(
            variable_names=tuple(metadata["variable_names"]),
            history=arrays["history"].astype(np.float64),
            condition=arrays["condition"].astype(np.float64),
            history_raw=arrays["history_raw"].astype(np.float64),
            condition_raw=arrays["condition_raw"].astype(np.float64),
            metadata=metadata,
        )
    artifact.validate(feature_names)
    return artifact


def _artifact(
    names: tuple[str, ...], history_raw: np.ndarray, condition_raw: np.ndarray, metadata: dict[str, Any]
) -> SemanticArtifact:
    return SemanticArtifact(
        variable_names=names,
        history=center_normalize(history_raw),
        condition=center_normalize(condition_raw),
        history_raw=np.asarray(history_raw, dtype=np.float64),
        condition_raw=np.asarray(condition_raw, dtype=np.float64),
        metadata={"created_at_utc": datetime.now(UTC).isoformat(), "centering": "task_mean", **metadata},
    )


EMBEDDING_PROVIDERS = {  # 接口类型 -> 设置输出维度的请求字段
    "openai_compatible": ("dimensions", "output_dimension"),
    "google_generative_language": ("outputDimensionality",),
    "cohere_native": ("output_dimension",),
}


@dataclass(frozen=True)
class EmbeddingConfig:
    """configs/embedding.yaml 中的一个模型条目（``Registry.entry``）。

    ``dimensions`` 为语义维度 k；``request_dimensions`` 为 API 请求维度，默认等于 k。
    请求维度较大时只保留前 k 维，该操作仅适用于经过 Matryoshka 训练的模型；当前登记的编码器
    均原生输出 k 维。编码结果随后减去任务内各变量的均值，再对每个变量作 L2 归一化。
    """

    model: str
    base_url: str
    api_key_env: str
    dimensions: int
    provider: str = "openai_compatible"
    dimension_parameter: str = "dimensions"
    request_dimensions: int | None = None
    batch_size: int = 10
    timeout_seconds: float = 120.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> EmbeddingConfig:
        if not str(payload.get("model", "")).strip() or not str(payload.get("base_url", "")).strip():
            raise ValueError("embedding model and base_url must be set; fill configs/env/api.local.env")
        provider = str(payload.get("provider", "openai_compatible"))
        parameter = str(payload.get("dimension_parameter", "dimensions"))
        if provider not in EMBEDDING_PROVIDERS:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        if parameter not in EMBEDDING_PROVIDERS[provider]:
            raise ValueError(
                f"{provider} sets the output dimension with one of {EMBEDDING_PROVIDERS[provider]}"
            )
        dimensions = int(payload["dimensions"])
        request = payload.get("request_dimensions")
        request = None if request in (None, "", "none") else int(request)
        if request is not None and request < dimensions:
            raise ValueError("request_dimensions must not be smaller than dimensions")
        return cls(
            model=str(payload["model"]).strip(),
            base_url=str(payload["base_url"]).strip(),
            api_key_env=str(payload["api_key_env"]),
            dimensions=dimensions,
            provider=provider,
            dimension_parameter=parameter,
            request_dimensions=request,
            batch_size=int(payload.get("batch_size", 10)),
            timeout_seconds=float(payload.get("timeout_seconds", 120.0)),
        )

    @property
    def requested(self) -> int:
        return self.request_dimensions or self.dimensions

    def public(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension_parameter": self.dimension_parameter,
            "dimensions": self.dimensions,
            "request_dimensions": self.requested,
        }


def embed_texts(texts: list[str], config: EmbeddingConfig) -> tuple[np.ndarray, int | None]:
    """返回 (len(texts), k) 的编码与接口报告的输入 token 数（接口不报告时为 None）。"""

    vectors, tokens, _ = request_embeddings(texts, config)
    embeddings = np.asarray(vectors, dtype=np.float64)
    if embeddings.shape != (len(texts), config.requested):
        raise ValueError(
            f"embedding API returned shape {embeddings.shape}, not {(len(texts), config.requested)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding API returned non-finite values")
    return embeddings[:, : config.dimensions], tokens


def request_embeddings(
    texts: list[str], config: EmbeddingConfig
) -> tuple[list[list[float]], int | None, str | None]:
    """按接口类型发送编码请求，返回未经检查的向量、输入 token 数与接口报告的模型名。

    正式编码（``embed_texts``）与 ``scripts/check_api.py`` 共用此函数，二者的请求体因此一致。
    """

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
    if config.provider == "openai_compatible":
        return _embed_openai_compatible(texts, config, api_key)
    if config.provider == "cohere_native":
        return _embed_cohere(texts, config, api_key)
    rows = [_embed_google(text, config, api_key) for text in texts]
    model = rows[0][2] if rows else None
    return [row for row, _, _ in rows], _total([count for _, count, _ in rows]), model


def _total(counts: list[int | None]) -> int | None:
    return None if not counts or any(count is None for count in counts) else int(sum(counts))


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"embedding API returned HTTP {error.code}: {error.read()[:300]!r}") from error


def _embed_openai_compatible(
    texts: list[str], config: EmbeddingConfig, api_key: str
) -> tuple[list[list[float]], int | None, str | None]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds, max_retries=0)
    vectors: list[list[float]] = []
    tokens: list[int | None] = []
    model = None
    try:
        for start in range(0, len(texts), config.batch_size):
            batch = texts[start : start + config.batch_size]
            request: dict[str, Any] = {"model": config.model, "input": batch}
            if config.dimension_parameter == "dimensions":
                request["dimensions"] = config.requested
            else:
                request["extra_body"] = {"output_dimension": config.requested}
            response = client.embeddings.create(**request)
            usage = getattr(response, "usage", None)
            tokens.append(getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", None))
            model = getattr(response, "model", None) or model
            data = sorted(response.data, key=lambda item: int(item.index))
            if [int(item.index) for item in data] != list(range(len(batch))):
                raise ValueError("embedding API returned missing or reordered rows")
            vectors.extend(item.embedding for item in data)
    finally:
        client.close()
    return vectors, _total(tokens), model


def _embed_cohere(
    texts: list[str], config: EmbeddingConfig, api_key: str
) -> tuple[list[list[float]], int | None, str | None]:
    """使用原生 v2 接口（OpenAI 兼容端点不支持 dimensions）；该接口不返回模型名。"""

    vectors: list[list[float]] = []
    tokens: list[int | None] = []
    for start in range(0, len(texts), config.batch_size):
        batch = texts[start : start + config.batch_size]
        payload = _post_json(
            f"{config.base_url.rstrip('/')}/embed",
            {
                "model": config.model,
                "texts": batch,
                "input_type": "clustering",
                "output_dimension": config.requested,
                "embedding_types": ["float"],
                "truncate": "NONE",
            },
            {"Authorization": f"Bearer {api_key}"},
            config.timeout_seconds,
        )
        rows = (payload.get("embeddings") or {}).get("float")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise ValueError("Cohere embedding API returned missing embedding rows")
        vectors.extend(rows)
        tokens.append(((payload.get("meta") or {}).get("billed_units") or {}).get("input_tokens"))
    return vectors, _total(tokens), None


def _embed_google(
    text: str, config: EmbeddingConfig, api_key: str
) -> tuple[list[float], int | None, str | None]:
    model = config.model if config.model.startswith("models/") else f"models/{config.model}"
    payload = _post_json(
        f"{config.base_url.rstrip('/')}/{model}:embedContent",
        {"content": {"parts": [{"text": text}]}, "outputDimensionality": config.requested},
        {"x-goog-api-key": api_key},
        config.timeout_seconds,
    )
    values = (payload.get("embedding") or {}).get("values")
    if not isinstance(values, list):
        raise ValueError("Google embedding API returned no embedding values")
    return values, (payload.get("usageMetadata") or {}).get("promptTokenCount"), payload.get("modelVersion")


def build_from_cards(
    cards_payload: dict[str, Any], *, cards_path: Path, embedding: EmbeddingConfig
) -> SemanticArtifact:
    cards = cards_payload["cards"]
    names = tuple(card["variable_name"] for card in cards)
    texts = [card["history_view"] for card in cards] + [card["condition_view"] for card in cards]
    start = time.perf_counter()
    vectors, tokens = embed_texts(texts, embedding)
    seconds = time.perf_counter() - start
    return _artifact(
        names,
        vectors[: len(names)],
        vectors[len(names) :],
        {
            "source": "llm",
            "cards_path": str(
                cards_path.relative_to(ROOT) if cards_path.is_relative_to(ROOT) else cards_path
            ),
            "cards_sha256": sha256_json(cards),
            "cards_edited_after_generation": cards_payload.get("edited_after_generation", False),
            "prompt_version": cards_payload.get("prompt_version"),
            "llm": cards_payload.get("llm"),
            "embedding": embedding.public(),
            "embedding_generation": {
                "texts": len(texts),
                "seconds": round(seconds, 3),
                "input_tokens": tokens,
            },
        },
    )


def build_random(names: tuple[str, ...], *, dim: int, seed: int) -> SemanticArtifact:
    rng = np.random.default_rng(seed)
    return _artifact(
        names,
        rng.standard_normal((len(names), dim)),
        rng.standard_normal((len(names), dim)),
        {"source": "random", "seed": seed},
    )
