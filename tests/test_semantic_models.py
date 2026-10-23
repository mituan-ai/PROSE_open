"""模型登记、LLM 响应处理与编码接口的无网络测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prose.semantic_artifact import EmbeddingConfig, embed_texts
from prose.semantic_cards import LLMConfig, _collect_stream, parse_json_response
from prose.semantic_models import load_registry, model_slug, source_name

ROOT = Path(__file__).resolve().parents[1]


def test_registries_list_a_default_and_candidates(monkeypatch) -> None:
    llms = load_registry(ROOT / "configs" / "llm.yaml")
    encoders = load_registry(ROOT / "configs" / "embedding.yaml")
    assert llms.resolve("default") == llms.default and llms.select("all")[0] == llms.default
    monkeypatch.delenv("PROSE_EMBEDDING_API_KEY", raising=False)
    assert not encoders.configured(encoders.default)
    monkeypatch.setenv("PROSE_EMBEDDING_API_KEY", "x")
    monkeypatch.setenv("PROSE_EMBEDDING_BASE_URL", "https://example.invalid/v2")
    assert encoders.configured(encoders.default)
    for name in llms.names:
        assert LLMConfig.from_mapping({**llms.entry(name), "base_url": "http://x"}).stream
    for name in encoders.names:
        config = EmbeddingConfig.from_mapping({**encoders.entry(name), "base_url": "http://x"})
        assert config.dimensions == config.requested == 256
    assert model_slug("vendor/model-1") == "vendor-model-1"
    assert source_name("vendor/model-1", "embed-v5.0-pro") == "vendor-model-1__embed-v5.0-pro"


def test_json_responses_and_streams() -> None:
    assert parse_json_response('{"cards": []}') == {"cards": []}
    assert parse_json_response('```json\n{"cards": []}\n```') == {"cards": []}
    with pytest.raises(json.JSONDecodeError):
        parse_json_response('Here it is: ```json\n{"cards": []}\n```')
    chunk = lambda text, finish=None: SimpleNamespace(  # noqa: E731
        model="m", choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)]
    )
    assert _collect_stream([chunk('{"a"'), chunk(": 1}", "stop")], "x")[:3] == ('{"a": 1}', "m", "stop")


class _FakeOpenAI:
    requests: list[dict] = []

    def __init__(self, **_):
        self.embeddings = self

    def create(self, **request):
        _FakeOpenAI.requests.append(request)
        size = request.get("dimensions") or request["extra_body"]["output_dimension"]
        rows = [
            SimpleNamespace(index=i, embedding=list(np.arange(size) + i))
            for i in range(len(request["input"]))
        ]
        return SimpleNamespace(data=rows[::-1])  # 故意打乱返回顺序

    def close(self):
        pass


def test_openai_compatible_embeddings_request_and_truncate(monkeypatch) -> None:
    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("KEY", "x")
    config = EmbeddingConfig.from_mapping(
        {
            "model": "m",
            "base_url": "http://x",
            "api_key_env": "KEY",
            "dimensions": 4,
            "dimension_parameter": "output_dimension",
            "request_dimensions": 8,
            "batch_size": 2,
        }
    )
    vectors, tokens = embed_texts(["a", "b", "c"], config)
    assert vectors.shape == (3, 4) and vectors[1, 0] == 1.0  # 恢复输入行顺序，只保留前 k 维
    assert _FakeOpenAI.requests[-1]["extra_body"] == {"output_dimension": 8}
    with pytest.raises(ValueError, match="smaller"):
        EmbeddingConfig.from_mapping(
            {"model": "m", "base_url": "u", "api_key_env": "KEY", "dimensions": 4, "request_dimensions": 2}
        )
    with pytest.raises(ValueError, match="output dimension"):
        EmbeddingConfig.from_mapping(
            {
                "model": "m",
                "base_url": "u",
                "api_key_env": "KEY",
                "dimensions": 4,
                "provider": "google_generative_language",
            }
        )


def test_google_embeddings(monkeypatch) -> None:
    import urllib.request

    seen = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.body

    def urlopen(request, timeout):
        seen.append((request.full_url, json.loads(request.data)))
        return Response(json.dumps({"embedding": {"values": [0.5] * 4}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("KEY", "x")
    config = EmbeddingConfig.from_mapping(
        {
            "model": "gemini-embedding-2",
            "base_url": "https://g/v1beta",
            "api_key_env": "KEY",
            "dimensions": 4,
            "provider": "google_generative_language",
            "dimension_parameter": "outputDimensionality",
        }
    )
    assert embed_texts(["a", "b"], config)[0].shape == (2, 4)
    assert seen[0][0] == "https://g/v1beta/models/gemini-embedding-2:embedContent"
    assert seen[0][1]["outputDimensionality"] == 4


def test_cohere_native_requests_dimensions_and_preserves_batch_rows(monkeypatch) -> None:
    import urllib.request

    seen = []

    class Response:
        def __init__(self, rows):
            self.rows = rows

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {
                    "embeddings": {"float": self.rows},
                    "meta": {"billed_units": {"input_tokens": len(self.rows)}},
                }
            ).encode()

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        seen.append((request.full_url, payload))
        return Response([[float(text)] * payload["output_dimension"] for text in payload["texts"]])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("KEY", "x")
    config = EmbeddingConfig.from_mapping(
        {
            "model": "embed-v5.0-pro",
            "base_url": "https://c/v2",
            "api_key_env": "KEY",
            "provider": "cohere_native",
            "dimension_parameter": "output_dimension",
            "dimensions": 256,
            "batch_size": 2,
        }
    )
    values, tokens = embed_texts(["1", "2", "3"], config)
    assert values.shape == (3, 256) and list(values[:, 0]) == [1, 2, 3] and tokens == 3
    assert len(seen) == 2 and seen[0][0] == "https://c/v2/embed"
    assert seen[0][1]["output_dimension"] == 256
    assert seen[0][1]["input_type"] == "clustering"
    assert "dimensions" not in seen[0][1]
