"""The OpenAI-compatible HTTP layer: requests in, tokens out, streamed as Server-Sent Events.

Endpoints: POST /v1/completions, POST /v1/chat/completions (both with "stream": true or false),
GET /v1/models, /health, /metrics (Prometheus text) and /stats (JSON).

Streaming format, one event per piece of new text, then a terminator:

    data: {"id":"cmpl-1","choices":[{"index":0,"text":" Hello","finish_reason":null}]}

    data: [DONE]

We write the SSE bytes ourselves with a StreamingResponse instead of FastAPI's EventSourceResponse
so the stream is exactly what OpenAI sends (EventSourceResponse can add keep-alive comment lines,
which some clients, vLLM's benchmark among them, try to parse as JSON).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from miniserve.api.detokenizer import IncrementalDetokenizer, StopChecker
from miniserve.api.protocol import ChatCompletionRequest, CompletionRequest
from miniserve.engine.engine import AsyncEngine
from miniserve.engine.sequence import SamplingParams

DONE = "data: [DONE]\n\n"


def sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))}\n\n"


def usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def create_app(
    engine: AsyncEngine,
    tokenizer: Any,
    served_model_name: str,
    max_model_len: int,
    max_waiting: int,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine.start()
        yield
        engine.shutdown()

    app = FastAPI(title="MiniServe", lifespan=lifespan)
    vocab_size = engine.engine.model.cfg.vocab_size

    @app.exception_handler(HTTPException)
    async def openai_error(_: Request, exc: HTTPException) -> JSONResponse:
        kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
        body = {"error": {"message": exc.detail, "type": kind, "code": exc.status_code}}
        return JSONResponse(body, status_code=exc.status_code)

    # ---- helpers --------------------------------------------------------------------------

    def check_capacity() -> None:
        # Backpressure: refuse new work instead of letting the queue (and latency) grow forever.
        if engine.num_waiting >= max_waiting:
            raise HTTPException(503, "server overloaded: too many waiting requests; retry later")

    def sampling_params(
        req: CompletionRequest | ChatCompletionRequest, prompt_len: int, max_tokens: int | None
    ) -> SamplingParams:
        if req.n != 1:
            raise HTTPException(400, "only n=1 is supported")
        if max_tokens is None:
            max_tokens = max_model_len - prompt_len
        if prompt_len + max_tokens > max_model_len:
            raise HTTPException(
                400,
                f"prompt ({prompt_len} tokens) + max_tokens ({max_tokens}) exceeds "
                f"max_model_len ({max_model_len})",
            )
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k if req.top_k > 0 else -1,
            seed=req.seed,
            ignore_eos=req.ignore_eos,
        )

    def completion_prompt_ids(prompt: str | list[int] | list[str] | list[list[int]]) -> list[int]:
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], int):
                ids = [int(t) for t in prompt]  # type: ignore[arg-type]
                if min(ids) < 0 or max(ids) >= vocab_size:
                    raise HTTPException(400, f"token ids must be in [0, {vocab_size})")
                return ids
            if len(prompt) != 1:
                raise HTTPException(400, "exactly one prompt per request is supported")
            return completion_prompt_ids(prompt[0])
        ids_: list[int] = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids_:
            raise HTTPException(400, "the prompt is empty")
        return ids_

    async def generate_text(
        request_id: str, prompt_ids: list[int], params: SamplingParams, stops: list[str]
    ) -> AsyncIterator[tuple[str, str | None, int]]:
        """Yield (new text, finish reason or None, output tokens so far)."""
        detok = IncrementalDetokenizer(tokenizer, prompt_ids)
        stopper = StopChecker(stops)
        done = False
        try:
            async with aclosing(engine.generate(request_id, prompt_ids, params)) as outputs:
                async for out in outputs:
                    text, hit = stopper.add(detok.add(out.new_token_ids))
                    if hit:  # a stop string: closing `outputs` aborts the request in the engine
                        done = True
                        yield text, "stop", out.num_output_tokens
                        return
                    if out.finished:
                        done = True
                        yield text + stopper.flush(), out.finish_reason, out.num_output_tokens
                        return
                    if text:
                        yield text, None, out.num_output_tokens
        except ValueError as exc:  # the engine rejected the request
            raise HTTPException(400, str(exc)) from exc
        finally:
            if not done:  # client disconnected mid-stream: free the KV blocks now
                engine.abort(request_id)

    # ---- routes ---------------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "owned_by": "miniserve",
                    "max_model_len": max_model_len,
                }
            ],
        }

    @app.get("/stats")
    async def stats() -> dict[str, float]:
        return engine.engine.stats()

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        lines = [f"miniserve_{k} {v}" for k, v in engine.engine.stats().items()]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Any:
        if req.echo or req.logprobs:
            raise HTTPException(400, "echo and logprobs are not supported")
        prompt_ids = completion_prompt_ids(req.prompt)
        params = sampling_params(req, len(prompt_ids), req.max_tokens)
        check_capacity()
        rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())

        def chunk(text: str, reason: str | None) -> dict[str, Any]:
            return {
                "id": rid,
                "object": "text_completion",
                "created": created,
                "model": served_model_name,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": reason}],
            }

        if req.stream:

            async def stream() -> AsyncIterator[str]:
                n_out = 0
                async for text, reason, n in generate_text(
                    rid, prompt_ids, params, req.stop_strings
                ):
                    n_out = n
                    yield sse(chunk(text, reason))
                if req.include_usage:
                    yield sse(
                        {**chunk("", None), "choices": [], "usage": usage(len(prompt_ids), n_out)}
                    )
                yield DONE

            return StreamingResponse(stream(), media_type="text/event-stream")

        parts, reason_, n_out_ = [], None, 0
        async for text, reason, n in generate_text(rid, prompt_ids, params, req.stop_strings):
            parts.append(text)
            reason_, n_out_ = reason or reason_, n
        body = chunk("".join(parts), reason_)
        body["usage"] = usage(len(prompt_ids), n_out_)
        return body

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        messages = [{"role": m.role, "content": m.text()} for m in req.messages]
        # Qwen3's template can emit <think>...</think> reasoning; benchmarks turn it off.
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        prompt_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)
        params = sampling_params(req, len(prompt_ids), req.max_completion_tokens or req.max_tokens)
        check_capacity()
        rid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

        def chunk(delta: dict[str, str], reason: str | None) -> dict[str, Any]:
            return {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": served_model_name,
                "choices": [
                    {"index": 0, "delta": delta, "logprobs": None, "finish_reason": reason}
                ],
            }

        if req.stream:

            async def stream() -> AsyncIterator[str]:
                n_out, first = 0, True
                async for text, reason, n in generate_text(
                    rid, prompt_ids, params, req.stop_strings
                ):
                    n_out = n
                    if first:  # the role arrives with the first token, as OpenAI and vLLM do
                        yield sse(chunk({"role": "assistant", "content": ""}, None))
                        first = False
                    yield sse(chunk({"content": text} if text else {}, reason))
                if req.include_usage:
                    yield sse(
                        {**chunk({}, None), "choices": [], "usage": usage(len(prompt_ids), n_out)}
                    )
                yield DONE

            return StreamingResponse(stream(), media_type="text/event-stream")

        parts, reason_, n_out_ = [], None, 0
        async for text, reason, n in generate_text(rid, prompt_ids, params, req.stop_strings):
            parts.append(text)
            reason_, n_out_ = reason or reason_, n
        return {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(parts)},
                    "logprobs": None,
                    "finish_reason": reason_,
                }
            ],
            "usage": usage(len(prompt_ids), n_out_),
        }

    return app
