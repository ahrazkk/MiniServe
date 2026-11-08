"""API contract tests: the official `openai` SDK against a real server (tiny model, CPU).

The tiny model's outputs are random-looking text; we only check the *protocol*: response shapes,
streaming, [DONE], usage counts, errors, backpressure and cleanup when a client disconnects.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import openai
import pytest
import torch
import uvicorn
from conftest import build_hf_model
from fastapi.testclient import TestClient

from miniserve.api.server import create_app
from miniserve.config import EngineConfig
from miniserve.engine.engine import AsyncEngine, LLMEngine
from miniserve.model.loader import from_hf_model

MAX_LEN = 512


def _tiny_engine(tokenizer: Any) -> LLMEngine:
    # The real tokenizer's ids go up to 151,668, so the tiny model gets Qwen3's real vocab size.
    hf = build_hf_model(
        dtype=torch.float32,
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
    )
    cfg = EngineConfig(
        model="tiny",
        device="cpu",
        dtype="float32",
        block_size=16,
        max_model_len=MAX_LEN,
        num_gpu_blocks=128,
        max_num_seqs=8,
    )
    model = from_hf_model(hf, torch.device("cpu"), torch.float32, MAX_LEN + 8)
    return LLMEngine(cfg, model=model, eos_token_ids=[151645, 151643])


@pytest.fixture(scope="module")
def server(qwen_tokenizer: Any) -> Iterator[tuple[str, LLMEngine]]:
    engine = _tiny_engine(qwen_tokenizer)
    app = create_app(AsyncEngine(engine), qwen_tokenizer, "tiny", MAX_LEN, max_waiting=64)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not srv.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", engine
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def client(server: tuple[str, LLMEngine]) -> openai.OpenAI:
    return openai.OpenAI(base_url=server[0] + "/v1", api_key="unused")


FIXED = {"ignore_eos": True}  # fixed output length, so usage counts are predictable


def test_completion(client: openai.OpenAI) -> None:
    r = client.completions.create(
        model="tiny", prompt="Hello there", max_tokens=8, temperature=0, extra_body=FIXED
    )
    assert isinstance(r.choices[0].text, str)
    assert r.choices[0].finish_reason == "length"
    assert r.usage is not None and r.usage.completion_tokens == 8 and r.usage.prompt_tokens == 2


def test_completion_from_token_ids(client: openai.OpenAI) -> None:
    r = client.completions.create(
        model="tiny", prompt=[9707, 1879, 0], max_tokens=4, extra_body=FIXED
    )
    assert r.usage is not None and r.usage.prompt_tokens == 3


def test_streaming_completion_with_usage(client: openai.OpenAI) -> None:
    stream = client.completions.create(
        model="tiny",
        prompt="Count:",
        max_tokens=12,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=FIXED,
    )
    chunks = list(stream)
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 12
    finals = [c.choices[0].finish_reason for c in chunks if c.choices]
    assert finals[-1] == "length" and all(f is None for f in finals[:-1])


def test_non_streaming_matches_streaming(client: openai.OpenAI) -> None:
    kwargs: dict[str, Any] = {
        "model": "tiny",
        "prompt": "Same text",
        "max_tokens": 10,
        "temperature": 0,
        "extra_body": FIXED,
    }
    full = client.completions.create(**kwargs).choices[0].text
    streamed = "".join(
        c.choices[0].text for c in client.completions.create(**kwargs, stream=True) if c.choices
    )
    assert full == streamed


def test_raw_stream_is_sse_and_ends_with_done(server: tuple[str, LLMEngine]) -> None:
    body = {"prompt": "Hi", "max_tokens": 5, "stream": True, "ignore_eos": True}
    with httpx.stream("POST", server[0] + "/v1/completions", json=body, timeout=30) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln]
    assert all(ln.startswith("data: ") for ln in lines)
    assert lines[-1] == "data: [DONE]"


def test_chat_completion_both_modes(client: openai.OpenAI) -> None:
    messages: Any = [{"role": "user", "content": "Say hi"}]
    r = client.chat.completions.create(
        model="tiny", messages=messages, max_tokens=6, extra_body=FIXED
    )
    assert r.choices[0].message.role == "assistant"
    assert r.usage is not None and r.usage.completion_tokens == 6
    chunks = list(
        client.chat.completions.create(
            model="tiny", messages=messages, max_tokens=6, stream=True, extra_body=FIXED
        )
    )
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "length"


def test_stop_string_ends_generation_early(client: openai.OpenAI) -> None:
    r = client.completions.create(
        model="tiny", prompt="abc", max_tokens=40, temperature=0, extra_body=FIXED
    )
    text = r.choices[0].text
    if len(text) < 4:
        pytest.skip("random model produced too little text to pick a stop string from")
    stop = text[2:4]
    r2 = client.completions.create(
        model="tiny", prompt="abc", max_tokens=40, temperature=0, stop=[stop], extra_body=FIXED
    )
    assert r2.choices[0].finish_reason == "stop"
    assert r2.choices[0].text == text[: text.index(stop)]


def test_too_long_request_is_rejected(client: openai.OpenAI) -> None:
    with pytest.raises(openai.BadRequestError):
        client.completions.create(model="tiny", prompt=[100] * 500, max_tokens=100)


def test_unsupported_n_is_rejected(client: openai.OpenAI) -> None:
    with pytest.raises(openai.BadRequestError):
        client.completions.create(model="tiny", prompt="x", n=2)


def test_disconnect_aborts_and_frees_kv_blocks(server: tuple[str, LLMEngine]) -> None:
    url, engine = server
    body = {"prompt": "Tell me a long story", "max_tokens": 400, "stream": True, "ignore_eos": True}
    with httpx.stream("POST", url + "/v1/completions", json=body, timeout=30) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i >= 4:
                break  # hang up mid-generation
    deadline = time.time() + 10
    while time.time() < deadline:
        s = engine.stats()
        if s["num_requests_running"] == 0 and s["kv_blocks_free"] == s["kv_blocks_total"]:
            return
        time.sleep(0.05)
    pytest.fail(f"request not aborted after disconnect: {engine.stats()}")


def test_models_health_and_metrics(server: tuple[str, LLMEngine]) -> None:
    url = server[0]
    assert httpx.get(url + "/health").json() == {"status": "ok"}
    assert httpx.get(url + "/v1/models").json()["data"][0]["id"] == "tiny"
    assert "miniserve_kv_blocks_total" in httpx.get(url + "/metrics").text


def test_full_queue_returns_503(qwen_tokenizer: Any, server: tuple[str, LLMEngine]) -> None:
    # A second app around the same engine, never started, with no room in its waiting queue.
    app = create_app(AsyncEngine(server[1]), qwen_tokenizer, "tiny", MAX_LEN, max_waiting=0)
    r = TestClient(app).post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == 503
