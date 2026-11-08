"""An async load generator for OpenAI-compatible servers, measuring what `vllm bench serve` does.

    python bench/loadgen.py --concurrency 16 --num-prompts 500 --input-len 256 --output-len 128

Per request it records the send time, the arrival time of every streamed chunk, and the token
count from the final `usage` chunk. From those (same definitions as vLLM's benchmark):

    TTFT  time to first token      first chunk - send
    TPOT  time per output token    (latency - TTFT) / (output tokens - 1), per request
    ITL   inter-token latency      gaps between consecutive chunks, all requests pooled
    E2EL  end-to-end latency       last chunk - send

Closed loop (--concurrency N): N clients, each sends its next request when the last finishes;
shows capacity. Open loop (--request-rate R): Poisson arrivals at R req/s whatever happens;
shows queueing under real traffic. Results (every request + a summary + the environment) are
saved as JSON in results/. While running it polls the server's /stats to record the peak number
of concurrent sequences and the server-side counters (preemptions, prefix-cache hits, ...).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from promptsets import random_prompts, shared_prefix_prompts


@dataclass
class RequestResult:
    prompt_len: int
    send_time: float = 0.0
    ttft: float = 0.0
    latency: float = 0.0
    output_tokens: int = 0
    itl: list[float] = field(default_factory=list)
    success: bool = False
    error: str = ""


async def send(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], prompt_len: int
) -> RequestResult:
    r = RequestResult(prompt_len=prompt_len)
    t0 = time.perf_counter()
    r.send_time = t0
    last = t0
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                r.error = f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}"
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                obj = json.loads(data)
                now = time.perf_counter()
                if obj.get("choices"):
                    if r.ttft == 0.0:
                        r.ttft = now - t0
                    else:
                        r.itl.append(now - last)
                    last = now
                elif obj.get("usage"):
                    r.output_tokens = obj["usage"]["completion_tokens"]
        r.latency = last - t0
        r.success = r.ttft > 0
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        r.error = repr(exc)
    return r


async def poll_stats(
    client: httpx.AsyncClient, base: str, peak: dict[str, float], stop: asyncio.Event
) -> None:
    while not stop.is_set():
        try:
            s = (await client.get(base + "/stats", timeout=2)).json()
            peak["running"] = max(peak.get("running", 0), s["num_requests_running"])
            peak["kv_usage"] = max(peak.get("kv_usage", 0.0), s["kv_usage"])
        except (httpx.HTTPError, ValueError, KeyError):
            return  # server has no /stats (e.g. the HF baseline)
        await asyncio.sleep(0.25)


async def get_stats(client: httpx.AsyncClient, base: str) -> dict[str, float] | None:
    try:
        resp = await client.get(base + "/stats", timeout=5)
        return dict(resp.json()) if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


async def run(
    args: argparse.Namespace, warm: list[list[int]], prompts: list[list[int]]
) -> dict[str, Any]:
    url = args.base_url + args.endpoint
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=limits) as client:

        def payload(p: list[int]) -> dict[str, Any]:
            return {
                "model": args.model,
                "prompt": p,
                "max_tokens": args.output_len,
                "temperature": args.temperature,
                "stream": True,
                "ignore_eos": args.ignore_eos,
                "stream_options": {"include_usage": True},
            }

        for p in warm:  # warm-up requests (CUDA init, allocator growth): not measured
            await send(client, url, payload(p), len(p))

        before = await get_stats(client, args.base_url)
        peak: dict[str, float] = {}
        stop = asyncio.Event()
        poller = asyncio.create_task(poll_stats(client, args.base_url, peak, stop))
        sem = asyncio.Semaphore(args.concurrency or len(prompts))
        rng = random.Random(args.seed)

        async def one(p: list[int]) -> RequestResult:
            async with sem:
                return await send(client, url, payload(p), len(p))

        t_start = time.perf_counter()
        tasks = []
        for p in prompts:
            tasks.append(asyncio.create_task(one(p)))
            if args.request_rate:  # open loop: exponential gaps = Poisson arrivals
                await asyncio.sleep(rng.expovariate(args.request_rate))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t_start
        stop.set()
        await poller
        after = await get_stats(client, args.base_url)

    server = None
    if before and after:
        server = {k: after[k] - before[k] for k in after if k.endswith("_total")}
        server["peak_running"] = peak.get("running", 0)
        server["peak_kv_usage"] = peak.get("kv_usage", 0.0)
        server["kv_blocks_total"] = after["kv_blocks_total"]
    return {
        "summary": summarize(results, duration, args),
        "server": server,
        "requests": [asdict(r) for r in results],
    }


def pct(values: list[float], ps: tuple[int, ...] = (50, 90, 99)) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values) * 1000.0  # seconds -> ms
    out = {"mean": float(arr.mean())}
    out.update({f"p{p}": float(np.percentile(arr, p)) for p in ps})
    return out


def summarize(
    results: list[RequestResult], duration: float, args: argparse.Namespace
) -> dict[str, Any]:
    ok = [r for r in results if r.success]
    out_tokens = sum(r.output_tokens for r in ok)
    in_tokens = sum(r.prompt_len for r in ok)
    tpots = [(r.latency - r.ttft) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
    good = [
        r
        for r, t in zip(
            ok, [(r.latency - r.ttft) / max(r.output_tokens - 1, 1) for r in ok], strict=True
        )
        if r.ttft * 1000 <= args.goodput_ttft_ms and t * 1000 <= args.goodput_tpot_ms
    ]
    return {
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "errors": sorted({r.error for r in results if r.error})[:5],
        "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_throughput": out_tokens / duration,
        "total_token_throughput": (in_tokens + out_tokens) / duration,
        "goodput_req_s": len(good) / duration,
        "goodput_slo": {"ttft_ms": args.goodput_ttft_ms, "tpot_ms": args.goodput_tpot_ms},
        "ttft_ms": pct([r.ttft for r in ok]),
        "tpot_ms": pct(tpots),
        "itl_ms": pct([x for r in ok for x in r.itl]),
        "e2el_ms": pct([r.latency for r in ok]),
        "mean_output_tokens": out_tokens / max(len(ok), 1),
    }


def environment() -> dict[str, Any]:
    def sh(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except OSError:
            return "unavailable"

    commit = sh(["git", "rev-parse", "--short", "HEAD"]) or "uncommitted"
    dirty = bool(sh(["git", "status", "--porcelain"]))
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": sh(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "commit": commit + ("-dirty" if dirty and commit != "uncommitted" else ""),
    }


def print_summary(label: str, s: dict[str, Any], server: dict[str, Any] | None) -> None:
    def row(name: str, d: dict[str, float]) -> str:
        return (
            f"  {name:5s} mean {d.get('mean', 0):8.1f}  p50 {d.get('p50', 0):8.1f}  "
            f"p90 {d.get('p90', 0):8.1f}  p99 {d.get('p99', 0):8.1f} ms"
        )

    print(f"\n=== {label} ===")
    print(f"  completed {s['completed']} (failed {s['failed']}) in {s['duration_s']:.1f} s")
    print(
        f"  output throughput {s['output_throughput']:9.1f} tok/s | requests "
        f"{s['request_throughput']:.2f}/s | goodput {s['goodput_req_s']:.2f}/s"
    )
    for k in ("ttft", "tpot", "itl", "e2el"):
        print(row(k.upper(), s[f"{k}_ms"]))
    if server:
        print(
            f"  server: peak running {server['peak_running']}, preemptions "
            f"{server.get('preemptions_total', 0)}, prefix hits "
            f"{server.get('prefix_cache_hit_tokens_total', 0)} tokens, spec accepted "
            f"{server.get('spec_accepted_tokens_total', 0)}/"
            f"{server.get('spec_draft_tokens_total', 0)}"
        )
    if s["errors"]:
        print(f"  errors: {s['errors']}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--endpoint", default="/v1/completions")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--dataset", choices=["random", "shared-prefix"], default="random")
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--prefix-len", type=int, default=1024)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=0, help="closed loop: N clients")
    p.add_argument("--request-rate", type=float, default=0.0, help="open loop: Poisson req/s")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goodput-ttft-ms", type=float, default=500.0)
    p.add_argument("--goodput-tpot-ms", type=float, default=50.0)
    p.add_argument("--label", default="run")
    p.add_argument("--system-config", default="", help="free text recorded with the results")
    p.add_argument("--result-dir", default="results")
    args = p.parse_args()

    n = args.num_prompts + args.warmup
    if args.dataset == "random":
        prompts = random_prompts(n, args.input_len, args.seed)
    else:
        prompts = shared_prefix_prompts(n, args.prefix_len, args.input_len, args.seed)
    result = asyncio.run(run(args, prompts[: args.warmup], prompts[args.warmup :]))
    print_summary(args.label, result["summary"], result["server"])

    record = {"label": args.label, "config": vars(args), "environment": environment(), **result}
    out = Path(args.result_dir) / f"{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
