"""scripts/check_api.py 的无网络测试：请求沿用登记表设置，截断单独标记，错误信息不含密钥。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_api", ROOT / "scripts" / "check_api.py")
check_api = importlib.util.module_from_spec(_spec)
sys.modules["check_api"] = check_api  # dataclass 需要按模块名查找
_spec.loader.exec_module(check_api)


class _FakeOpenAI:
    requests: list[dict] = []
    reply: object = None

    def __init__(self, **_):
        self.chat = SimpleNamespace(completions=self)
        self.embeddings = self

    def create(self, **request):
        _FakeOpenAI.requests.append(request)
        return _FakeOpenAI.reply

    def close(self):
        pass


def _completion(content, finish):
    usage = SimpleNamespace(
        prompt_tokens=9,
        completion_tokens=512,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=500),
    )
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        model="m-0101", usage=usage, choices=[SimpleNamespace(message=message, finish_reason=finish)]
    )


def test_llm_probe_keeps_registry_settings_and_flags_truncation(monkeypatch) -> None:
    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("KEY", "secret-value")
    config = check_api.LLMConfig.from_mapping(
        {
            "model": "m",
            "base_url": "http://x",
            "api_key_env": "KEY",
            "reasoning_effort": "max",
            "stream": False,
            "token_parameter": "max_completion_tokens",
            "max_tokens": 512,
        }
    )
    _FakeOpenAI.reply = _completion("", "length")
    probe = check_api.probe_llm(config)
    request = _FakeOpenAI.requests[-1]
    assert request["reasoning_effort"] == "max" and request["max_completion_tokens"] == 512
    assert probe.status == "TRUNCATED" and probe.tokens == "in=9 out=512 reasoning=500"
    _FakeOpenAI.reply = _completion('{"ok": true}', "stop")
    assert check_api.probe_llm(config) == check_api.Probe("OK", "m-0101", "in=9 out=512 reasoning=500")
    _FakeOpenAI.reply = _completion("ok", "stop")
    assert check_api.probe_llm(config).status == "MISMATCH"


def test_llm_probe_reads_stream_and_its_usage(monkeypatch) -> None:
    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("KEY", "x")
    chunk = lambda text, finish=None: SimpleNamespace(  # noqa: E731
        model="q",
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)],
    )
    usage = SimpleNamespace(prompt_tokens=9, completion_tokens=4, completion_tokens_details=None)
    _FakeOpenAI.reply = [
        chunk('{"ok"'),
        chunk(":true}", "stop"),
        SimpleNamespace(model="q", usage=usage, choices=[]),
    ]
    config = check_api.LLMConfig.from_mapping(
        {"model": "q", "base_url": "u", "api_key_env": "KEY", "stream": True}
    )
    assert check_api.probe_llm(config) == check_api.Probe("OK", "q", "in=9 out=4")
    assert _FakeOpenAI.requests[-1]["stream_options"] == {"include_usage": True}


def test_embedding_probe_requires_256_finite_values(monkeypatch) -> None:
    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("KEY", "x")
    config = check_api.EmbeddingConfig.from_mapping(
        {"model": "e", "base_url": "u", "api_key_env": "KEY", "dimensions": 256}
    )

    def reply(vector):
        return SimpleNamespace(
            model="e",
            usage=SimpleNamespace(prompt_tokens=12),
            data=[SimpleNamespace(index=0, embedding=vector)],
        )

    _FakeOpenAI.reply = reply([0.1] * 256)
    assert check_api.probe_embedding(config).status == "OK"
    assert _FakeOpenAI.requests[-1]["dimensions"] == 256
    _FakeOpenAI.reply = reply([0.1] * 1024)
    assert "1024 dimensions" in check_api.probe_embedding(config).detail
    _FakeOpenAI.reply = reply([0.1] * 255 + [float("nan")])
    assert check_api.probe_embedding(config).status == "MISMATCH"


def test_errors_are_redacted() -> None:
    text = check_api.redact(
        "401 Incorrect API key secret-value; Bearer abcdef123456 sk-zz9988776655", ["secret-value"]
    )
    assert "secret-value" not in text and "abcdef123456" not in text and "zz9988776655" not in text
