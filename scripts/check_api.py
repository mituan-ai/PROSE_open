"""API 连通性检查：对 configs/llm.yaml 与 configs/embedding.yaml 登记的模型各发送一次最小请求。

    # 全部登记的 LLM 与编码器
    uv run --extra llm python scripts/check_api.py
    # 只测指定模型；LLM 输出上限可调
    uv run --extra llm python scripts/check_api.py claude-opus-5-5 embed-v5.0-pro --max-tokens 1024

LLM 请求与正式生成相同（系统提示、JSON 输出、登记表中的思考强度、token 参数名与流式设置），
只把提示词换成要求回答 {"ok":true}。编码器只编码一段短文本，检查返回维度等于登记的维度（默认 256）
且全部为有限数值。
每个模型只请求一次，不重试，不写任何文件。

状态：OK；TRUNCATED 表示输出截断（finish_reason 为 length 或 content_filter），接口可用但输出上限
不足以完成回答，不计为失败；MISMATCH 表示接口有响应但内容不符合预期；ERROR 表示请求失败；
SKIPPED 表示未配置地址或密钥，未发送请求。存在 MISMATCH 或 ERROR 时退出码为 1。
输出中不显示地址与密钥，错误信息中的密钥被遮盖。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from prose.config import CONFIG_DIR
from prose.io import load_api_env_files, resolve_path
from prose.semantic_artifact import EmbeddingConfig, request_embeddings
from prose.semantic_cards import TRUNCATED, LLMConfig, call_llm, parse_json_response
from prose.semantic_models import Registry, load_registry

PROMPT = 'Reply with this JSON only: {"ok":true}'
EXPECTED = {"ok": True}
PROBE_TEXT = "Underflow density rises after the flocculant dose is increased."
FAILURES = ("MISMATCH", "ERROR")
_TOKEN_LIKE = re.compile(r"(sk-|Bearer\s+|key=)[A-Za-z0-9_\-.]{6,}", re.IGNORECASE)


@dataclass(frozen=True)
class Probe:
    status: str
    returned_model: str = "-"
    tokens: str = "-"
    detail: str = ""


def _usage(*pairs: tuple[str, Any]) -> str:
    return " ".join(f"{label}={value}" for label, value in pairs if value is not None) or "-"


def probe_llm(config: LLMConfig) -> Probe:
    """发送与正式生成相同的请求（``prose.semantic_cards.call_llm``），只换成极短的提示词。"""

    reply = call_llm(PROMPT, config)
    tokens = _usage(
        ("in", reply.usage.get("prompt_tokens")),
        ("out", reply.usage.get("completion_tokens")),
        ("reasoning", reply.usage.get("reasoning_tokens")),
    )
    if reply.finish_reason in TRUNCATED:
        detail = f"finish_reason={reply.finish_reason} at {config.max_tokens} tokens"
        return Probe("TRUNCATED", reply.model, tokens, detail)
    try:
        ok = parse_json_response(reply.text) == EXPECTED
    except RuntimeError, ValueError:
        ok = False
    if ok:
        return Probe("OK", reply.model, tokens)
    return Probe(
        "MISMATCH", reply.model, tokens, f"finish_reason={reply.finish_reason} reply={reply.text[:80]!r}"
    )


def probe_embedding(config: EmbeddingConfig) -> Probe:
    """发送与正式编码相同的请求（``prose.semantic_artifact.request_embeddings``），只编码一段短文本。"""

    vectors, count, returned = request_embeddings([PROBE_TEXT], config)
    returned, tokens = str(returned or "-"), _usage(("in", count))
    vector = vectors[0] if vectors else None
    if not isinstance(vector, list):
        return Probe("MISMATCH", returned, tokens, "no embedding vector in the response")
    if len(vector) != config.requested:
        return Probe(
            "MISMATCH", returned, tokens, f"returned {len(vector)} dimensions, expected {config.requested}"
        )
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vector)
    if not numeric or not all(math.isfinite(v) for v in vector):
        return Probe("MISMATCH", returned, tokens, "non-numeric or non-finite values")
    if not any(vector):
        return Probe("MISMATCH", returned, tokens, "all-zero vector")
    return Probe("OK", returned, tokens, f"{len(vector)} finite values")


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "***")
    return _TOKEN_LIKE.sub(lambda match: f"{match.group(1)}***", " ".join(text.split()))


def run_one(kind: str, name: str, registry: Registry, max_tokens: int, secrets: list[str]):
    if not registry.configured(name):
        return Probe("SKIPPED", detail=f"{registry.entries[name]['api_key_env']} or base URL not set"), None
    start = time.perf_counter()
    try:
        if kind == "llm":
            config = replace(LLMConfig.from_mapping(registry.entry(name)), max_tokens=max_tokens)
            probe = probe_llm(config)
        else:
            probe = probe_embedding(EmbeddingConfig.from_mapping(registry.entry(name)))
    except Exception as error:  # 单个模型失败不影响其余检查，错误在汇总中报告
        probe = Probe("ERROR", detail=redact(f"{type(error).__name__}: {error}", secrets)[:300])
    seconds = time.perf_counter() - start
    return replace(probe, detail=redact(probe.detail, secrets)), seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="*", help="model names from either registry (default: all)")
    parser.add_argument("--max-tokens", type=int, default=512, help="LLM output token limit (default 512)")
    parser.add_argument("--llm-config", default=str(CONFIG_DIR / "llm.yaml"))
    parser.add_argument("--embedding-config", default=str(CONFIG_DIR / "embedding.yaml"))
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    # openai 客户端按需导入子模块；多线程同时首次导入会触发导入锁死锁，故先在主线程导入
    import openai.resources.chat  # noqa: F401
    import openai.resources.embeddings  # noqa: F401

    load_api_env_files()
    registries = {
        "llm": load_registry(resolve_path(args.llm_config)),
        "embedding": load_registry(resolve_path(args.embedding_config)),
    }
    available = {name: kind for kind, registry in registries.items() for name in registry.names}
    unknown = [name for name in args.models if name not in available]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; available: {list(available)}")
    jobs = [(available[name], name) for name in (args.models or available)]
    secrets = sorted(
        {
            os.environ[entry["api_key_env"]]
            for registry in registries.values()
            for entry in registry.entries.values()
            if os.environ.get(entry["api_key_env"])
        },
        key=len,
        reverse=True,
    )

    print(f"checking {len(jobs)} model(s), one request each, LLM max tokens {args.max_tokens}", flush=True)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [
            pool.submit(run_one, kind, name, registries[kind], args.max_tokens, secrets)
            for kind, name in jobs
        ]
        results = [future.result() for future in futures]

    print(f"{'kind':<10}{'model':<24}{'status':<11}{'returned model':<34}{'time':>8}  {'tokens':<36}detail")
    for (kind, name), (probe, seconds) in zip(jobs, results, strict=True):
        elapsed = "-" if seconds is None else f"{seconds:.1f}s"
        print(
            f"{kind:<10}{name:<24}{probe.status:<11}{probe.returned_model:<34}{elapsed:>8}  "
            f"{probe.tokens:<36}{probe.detail}"
        )
    statuses = [probe.status for probe, _ in results]
    print("summary: " + ", ".join(f"{statuses.count(s)} {s}" for s in dict.fromkeys(statuses)))
    if any(status in FAILURES for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
