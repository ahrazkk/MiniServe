"""The engine: the scheduler, the KV block manager and the model runner, driven one step at a time.

    while there is work:
        plan    = scheduler.schedule()        # who runs this step, and how many tokens each
        tokens  = runner.execute(plan)        # one forward pass + sampling on the GPU
        updates = scheduler.update(plan, tokens)  # append tokens, stop checks, free finished

`LLMEngine` is synchronous and single-threaded (tests and offline use call `step()` directly).
`AsyncEngine` runs that loop in a background thread so the asyncio HTTP server never blocks on
GPU work; the two sides talk through a thread-safe inbox and one asyncio queue per request.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from functools import partial

import torch

from miniserve.config import EngineConfig
from miniserve.engine.block_manager import BlockManager, cdiv
from miniserve.engine.scheduler import Scheduler
from miniserve.engine.sequence import SamplingParams, Sequence
from miniserve.model.loader import load_qwen3, read_eos_token_ids
from miniserve.model.qwen3 import Qwen3ForCausalLM
from miniserve.quant.int8 import quantize_int8_
from miniserve.runner.model_runner import ModelRunner
from miniserve.spec.ngram import propose_ngram

logger = logging.getLogger(__name__)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass
class RequestOutput:
    request_id: str
    new_token_ids: list[int]
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_output_tokens: int
    num_cached_tokens: int


class LLMEngine:
    def __init__(
        self,
        cfg: EngineConfig,
        model: Qwen3ForCausalLM | None = None,
        eos_token_ids: Iterable[int] | None = None,
    ) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.dtype = DTYPES[cfg.dtype]
        if model is None:
            max_positions = cfg.max_model_len + cfg.num_speculative_tokens + 1
            t0 = time.perf_counter()
            model = load_qwen3(cfg.model, self.device, self.dtype, max_positions)
            logger.info("loaded %s in %.1f s", cfg.model, time.perf_counter() - t0)
            if eos_token_ids is None:
                eos_token_ids = read_eos_token_ids(cfg.model)
        self.model = model
        self.eos_token_ids = frozenset(eos_token_ids or ())
        if cfg.quantization == "int8":
            saved = quantize_int8_(model)
            logger.info("INT8 weight-only quantization saved %.0f MiB", saved / 2**20)

        self.runner = ModelRunner(cfg, model.cfg, model, self.device, self.dtype)
        num_blocks = cfg.num_gpu_blocks or self.runner.profile_num_blocks()
        if num_blocks * cfg.block_size < cfg.max_model_len:
            raise ValueError("the KV pool can't hold even one max_model_len request")
        self.runner.allocate_kv_cache(num_blocks)
        self.block_manager = BlockManager(num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        proposer = (
            partial(propose_ngram, n_max=cfg.ngram_max, n_min=cfg.ngram_min)
            if cfg.speculative == "ngram"
            else None
        )
        self.scheduler = Scheduler(cfg, self.block_manager, proposer)

        self.num_steps = 0
        self.num_prompt_tokens = 0  # prompt tokens of admitted requests
        self.num_generated_tokens = 0
        self.num_finished = 0
        kv_mib = num_blocks * self.runner.bytes_per_block / 2**20
        per_request = cdiv(cfg.max_model_len, cfg.block_size)
        logger.info(
            "KV cache: %d blocks x %d tokens = %d tokens (%.0f MiB); %d full-length requests fit",
            num_blocks,
            cfg.block_size,
            num_blocks * cfg.block_size,
            kv_mib,
            num_blocks // per_request,
        )

    # ---- requests -------------------------------------------------------------------------

    def add_request(self, request_id: str, prompt_ids: list[int], params: SamplingParams) -> None:
        if not prompt_ids:
            raise ValueError("the prompt is empty")
        if len(prompt_ids) >= self.cfg.max_model_len:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens; max_model_len is {self.cfg.max_model_len}"
            )
        vocab = self.model.cfg.vocab_size
        if min(prompt_ids) < 0 or max(prompt_ids) >= vocab:
            raise ValueError(f"token ids must be in [0, {vocab})")
        self.num_prompt_tokens += len(prompt_ids)
        self.scheduler.add(Sequence(request_id, prompt_ids, params, self.eos_token_ids))

    def abort_request(self, request_id: str) -> bool:
        return self.scheduler.abort(request_id) is not None

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    # ---- the step -------------------------------------------------------------------------

    def step(self) -> list[RequestOutput]:
        plan = self.scheduler.schedule()
        if not plan.scheduled:
            return []
        tokens = self.runner.execute(plan.scheduled)
        updates = self.scheduler.update(plan, tokens)
        self.num_steps += 1
        outputs = []
        for seq, new_tokens in updates:
            self.num_generated_tokens += len(new_tokens)
            if seq.is_finished:
                self.num_finished += 1
            if new_tokens or seq.is_finished:
                outputs.append(
                    RequestOutput(
                        request_id=seq.request_id,
                        new_token_ids=new_tokens,
                        finished=seq.is_finished,
                        finish_reason=seq.finish_reason,
                        num_prompt_tokens=seq.num_prompt_tokens,
                        num_output_tokens=seq.num_output_tokens,
                        num_cached_tokens=seq.num_cached_tokens,
                    )
                )
        return outputs

    def generate(
        self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]
    ) -> list[list[int]]:
        """Offline helper: run every prompt to completion; returns output token ids in order."""
        plist = params if isinstance(params, list) else [params] * len(prompts)
        outputs: dict[str, list[int]] = {}
        for i, (prompt, p) in enumerate(zip(prompts, plist, strict=True)):
            rid = f"offline-{self.num_finished}-{i}"
            outputs[rid] = []
            self.add_request(rid, prompt, p)
        while self.has_unfinished_requests():
            for out in self.step():
                if out.request_id in outputs:
                    outputs[out.request_id].extend(out.new_token_ids)
        return list(outputs.values())

    def warmup(self) -> None:
        """One small request end to end, so the first real request doesn't pay one-time costs."""
        self.generate([[1, 2, 3, 4, 5, 6, 7, 8]], SamplingParams(max_tokens=4, ignore_eos=True))

    def stats(self) -> dict[str, float]:
        bm, sched = self.block_manager, self.scheduler
        return {
            "num_requests_running": len(sched.running),
            "num_requests_waiting": len(sched.waiting),
            "kv_blocks_total": bm.num_blocks,
            "kv_blocks_free": bm.num_free_blocks,
            "kv_blocks_cached": len(bm.cached),
            "kv_usage": 1.0 - bm.num_free_blocks / bm.num_blocks,
            "engine_steps_total": self.num_steps,
            "preemptions_total": sched.num_preemptions,
            "prompt_tokens_total": self.num_prompt_tokens,
            "generation_tokens_total": self.num_generated_tokens,
            "requests_finished_total": self.num_finished,
            "prefix_cache_hit_tokens_total": self.prefix_cache_hits,
            "spec_draft_tokens_total": self.runner.spec_drafted,
            "spec_accepted_tokens_total": self.runner.spec_accepted,
        }

    @property
    def prefix_cache_hits(self) -> int:
        return self.scheduler.num_prefix_hit_tokens


class AsyncEngine:
    """Runs `LLMEngine.step()` forever in one background thread.

    The engine loop is blocking GPU work, so it must not run on the asyncio event loop (that
    would freeze every HTTP connection during each step). Inbox: a thread-safe `queue.Queue` of
    add/abort commands. Outbox: one `asyncio.Queue` per request, filled from the engine thread
    with `loop.call_soon_threadsafe`, the only safe way to touch asyncio objects from another
    thread.
    """

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._inbox: queue.Queue[tuple[str, object]] = queue.Queue()
        self._streams: dict[
            str, tuple[asyncio.AbstractEventLoop, asyncio.Queue[RequestOutput | BaseException]]
        ] = {}
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
            self._thread.start()

    def shutdown(self) -> None:
        self._running = False
        self._inbox.put(("stop", None))
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    @property
    def num_waiting(self) -> int:
        return len(self.engine.scheduler.waiting) + self._inbox.qsize()

    async def generate(
        self, request_id: str, prompt_ids: list[int], params: SamplingParams
    ) -> AsyncIterator[RequestOutput]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[RequestOutput | BaseException] = asyncio.Queue()
        self._streams[request_id] = (loop, q)
        self._inbox.put(("add", (request_id, prompt_ids, params)))
        finished = False
        try:
            while True:
                item = await q.get()
                if isinstance(item, BaseException):
                    finished = True
                    raise item
                if item.finished:
                    finished = True
                yield item
                if finished:
                    return
        finally:
            self._streams.pop(request_id, None)
            if not finished:  # the client went away (or stopped reading): free its KV blocks
                self.abort(request_id)

    def abort(self, request_id: str) -> None:
        self._inbox.put(("abort", request_id))

    # ---- engine thread --------------------------------------------------------------------

    def _loop(self) -> None:
        engine = self.engine
        while self._running:
            try:
                # Idle: block until a command arrives. Busy: just drain what's there.
                cmd = self._inbox.get(block=not engine.has_unfinished_requests(), timeout=1.0)
                self._handle(cmd)
                while True:
                    self._handle(self._inbox.get_nowait())
            except queue.Empty:
                pass
            if not engine.has_unfinished_requests():
                continue
            try:
                outputs = engine.step()
            except Exception as exc:  # a bug in one step must not leave clients hanging forever
                logger.exception("engine step failed; aborting all requests")
                self._fail_all(exc)
                continue
            for out in outputs:
                self._publish(out.request_id, out)

    def _handle(self, cmd: tuple[str, object]) -> None:
        kind, payload = cmd
        if kind == "add":
            request_id, prompt_ids, params = payload  # type: ignore[misc]
            try:
                self.engine.add_request(request_id, prompt_ids, params)
            except ValueError as exc:
                self._publish(request_id, exc)
        elif kind == "abort":
            self.engine.abort_request(payload)  # type: ignore[arg-type]
        elif kind == "stop":
            self._running = False

    def _publish(self, request_id: str, item: RequestOutput | BaseException) -> None:
        stream = self._streams.get(request_id)
        if stream is not None:
            loop, q = stream
            loop.call_soon_threadsafe(q.put_nowait, item)

    def _fail_all(self, exc: Exception) -> None:
        sched = self.engine.scheduler
        for seq in [*sched.running, *sched.waiting]:
            sched.abort(seq.request_id)
            self._publish(seq.request_id, RuntimeError(f"engine error: {exc!r}"))
