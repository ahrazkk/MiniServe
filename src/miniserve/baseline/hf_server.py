"""M1, the naive baseline: Hugging Face `generate()` behind the same OpenAI-style endpoint.

One request at a time, on purpose: this is the "before" picture every optimization is measured
against. It still uses a KV cache (generate() does by default); what it lacks is batching, so the
GPU reads all 1.2 GB of weights for every single token of every single request.

It speaks the same /v1/completions protocol (streaming and not) as MiniServe, so the same load
generator measures both.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from miniserve.api.detokenizer import IncrementalDetokenizer, StopChecker
from miniserve.api.protocol import CompletionRequest
from miniserve.api.server import DONE, sse, usage
from miniserve.model.loader import read_eos_token_ids


class _TokenStreamer(BaseStreamer):
    """Hands each new token id from generate()'s thread to the asyncio side."""

    def __init__(self, loop: asyncio.AbstractEventLoop, q: asyncio.Queue[int | None]) -> None:
        self.loop, self.q, self.skip_prompt = loop, q, True

    def put(self, value: torch.Tensor) -> None:
        if self.skip_prompt:  # the first call is the prompt itself
            self.skip_prompt = False
            return
        for token in value.flatten().tolist():
            self.loop.call_soon_threadsafe(self.q.put_nowait, token)

    def end(self) -> None:
        self.loop.call_soon_threadsafe(self.q.put_nowait, None)


class _StopFlag(StoppingCriteria):
    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return torch.full(
            (input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device
        )


def create_baseline_app(
    model_dir: str, device: str = "cuda", served_model_name: str | None = None
) -> FastAPI:
    name = served_model_name or model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
    eos_ids = sorted(read_eos_token_ids(model_dir))
    max_len = model.config.max_position_embeddings
    queue_lock = asyncio.Lock()  # one request at a time: this IS the naive baseline
    gpu_lock = threading.Lock()  # and the GPU is never shared, even if a client disconnects

    def run_generate(
        prompt_ids: list[int],
        req: CompletionRequest,
        max_tokens: int,
        streamer: BaseStreamer | None,
        stop: threading.Event,
    ) -> list[int]:
        sampling = req.temperature > 1e-5
        gen = GenerationConfig(
            max_new_tokens=max_tokens,
            min_new_tokens=max_tokens if req.ignore_eos else 0,
            do_sample=sampling,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
            **(
                {
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "top_k": req.top_k if req.top_k > 0 else 0,
                }
                if sampling
                else {}
            ),
        )
        ids = torch.tensor([prompt_ids], device=device)
        with gpu_lock, torch.inference_mode():
            if req.seed is not None:
                torch.manual_seed(req.seed)
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                generation_config=gen,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([_StopFlag(stop)]),
            )
        new: list[int] = out[0, len(prompt_ids) :].tolist()
        return new

    app = FastAPI(title="MiniServe HF baseline")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "hf"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Any:
        if isinstance(req.prompt, str):
            prompt_ids = tokenizer.encode(req.prompt, add_special_tokens=False)
        elif req.prompt and isinstance(req.prompt[0], int):
            prompt_ids = [int(t) for t in req.prompt]  # type: ignore[arg-type]
        else:
            raise HTTPException(400, "send one prompt, as a string or a list of token ids")
        max_tokens = req.max_tokens or 16
        if len(prompt_ids) + max_tokens > max_len:
            raise HTTPException(400, "prompt + max_tokens is too long")
        rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())

        def chunk(text: str, reason: str | None) -> dict[str, Any]:
            return {
                "id": rid,
                "object": "text_completion",
                "created": created,
                "model": name,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": reason}],
            }

        def finish_reason(n: int, last: int | None) -> str:
            return (
                "stop" if (last in eos_ids and not req.ignore_eos) or n < max_tokens else "length"
            )

        if req.stream:

            async def stream() -> AsyncIterator[str]:
                stop = threading.Event()
                async with queue_lock:
                    loop = asyncio.get_running_loop()
                    q: asyncio.Queue[int | None] = asyncio.Queue()
                    job = asyncio.ensure_future(
                        asyncio.to_thread(
                            run_generate, prompt_ids, req, max_tokens, _TokenStreamer(loop, q), stop
                        )
                    )
                    detok = IncrementalDetokenizer(tokenizer, prompt_ids)
                    stopper = StopChecker(req.stop_strings)
                    n, last, hit = 0, None, False
                    try:
                        while (token := await q.get()) is not None:
                            n, last = n + 1, token
                            text, hit = stopper.add(detok.add([token]))
                            if hit:
                                stop.set()
                                yield sse(chunk(text, "stop"))
                                break
                            if text:
                                yield sse(chunk(text, None))
                        if not hit:
                            yield sse(chunk(stopper.flush(), finish_reason(n, last)))
                        if req.include_usage:
                            yield sse(
                                {
                                    **chunk("", None),
                                    "choices": [],
                                    "usage": usage(len(prompt_ids), n),
                                }
                            )
                        yield DONE
                    finally:
                        stop.set()  # client gone or stop string: let generate() end early
                        await job

            return StreamingResponse(stream(), media_type="text/event-stream")

        async with queue_lock:
            out = await asyncio.to_thread(
                run_generate, prompt_ids, req, max_tokens, None, threading.Event()
            )
        text = tokenizer.decode(out, skip_special_tokens=True)
        stopper = StopChecker(req.stop_strings)
        emitted, hit = stopper.add(text)
        body = chunk(
            emitted + ("" if hit else stopper.flush()),
            "stop" if hit else finish_reason(len(out), out[-1] if out else None),
        )
        body["usage"] = usage(len(prompt_ids), len(out))
        return body

    return app
