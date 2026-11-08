"""`miniserve` command line.

miniserve serve    --model ~/models/Qwen3-0.6B --port 8000     # the engine + OpenAI API
miniserve baseline --model ~/models/Qwen3-0.6B --port 8000     # naive HF baseline (M1)
miniserve generate --model ~/models/Qwen3-0.6B --prompt "Hi"  # quick offline check
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

from miniserve.config import EngineConfig

DEFAULT_MODEL = os.environ.get("MINISERVE_MODEL", "/root/models/Qwen3-0.6B")


def _engine_args(p: argparse.ArgumentParser) -> None:
    d = EngineConfig.model_fields
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--served-model-name", default=None)
    p.add_argument(
        "--dtype", default=d["dtype"].default, choices=["bfloat16", "float16", "float32"]
    )
    p.add_argument("--device", default=d["device"].default)
    p.add_argument("--seed", type=int, default=d["seed"].default)
    p.add_argument("--block-size", type=int, default=d["block_size"].default)
    p.add_argument("--max-model-len", type=int, default=d["max_model_len"].default)
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=d["gpu_memory_utilization"].default
    )
    p.add_argument("--num-gpu-blocks", type=int, default=None)
    p.add_argument(
        "--kv-reservation",
        default="paged",
        choices=["paged", "full"],
        help="'full' reserves max_model_len per request (the pre-paging baseline)",
    )
    p.add_argument("--max-num-seqs", type=int, default=d["max_num_seqs"].default)
    p.add_argument(
        "--max-num-batched-tokens", type=int, default=d["max_num_batched_tokens"].default
    )
    p.add_argument("--no-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    p.add_argument("--no-prefix-caching", dest="enable_prefix_caching", action="store_false")
    p.add_argument("--max-waiting", type=int, default=d["max_waiting"].default)
    p.add_argument("--speculative", default="none", choices=["none", "ngram"])
    p.add_argument(
        "--num-speculative-tokens", type=int, default=d["num_speculative_tokens"].default
    )
    p.add_argument("--ngram-max", type=int, default=d["ngram_max"].default)
    p.add_argument("--ngram-min", type=int, default=d["ngram_min"].default)
    p.add_argument("--quantization", default="none", choices=["none", "int8"])


def _server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)


def engine_config(args: argparse.Namespace) -> EngineConfig:
    fields: dict[str, Any] = {k: v for k, v in vars(args).items() if k in EngineConfig.model_fields}
    return EngineConfig(**fields)


def serve(args: argparse.Namespace) -> None:
    import uvicorn
    from transformers import AutoTokenizer

    from miniserve.api.server import create_app
    from miniserve.engine.engine import AsyncEngine, LLMEngine

    cfg = engine_config(args)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    engine = LLMEngine(cfg)
    engine.warmup()
    name = cfg.served_model_name or cfg.model
    app = create_app(AsyncEngine(engine), tokenizer, name, cfg.max_model_len, cfg.max_waiting)
    logging.getLogger(__name__).info("serving %s on http://%s:%d", name, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


def baseline(args: argparse.Namespace) -> None:
    import uvicorn

    from miniserve.baseline.hf_server import create_baseline_app

    app = create_baseline_app(args.model, args.device, args.served_model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from miniserve.engine.engine import LLMEngine
    from miniserve.engine.sequence import SamplingParams

    cfg = engine_config(args)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    engine = LLMEngine(cfg)
    prompts = args.prompt or ["The capital of Canada is"]
    ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    t0 = time.perf_counter()
    outputs = engine.generate(ids, params)
    dt = time.perf_counter() - t0
    for prompt, out in zip(prompts, outputs, strict=True):
        print(f"--- {prompt!r}\n{tokenizer.decode(out, skip_special_tokens=True)}")
    n = sum(len(o) for o in outputs)
    print(f"\n{n} tokens in {dt:.2f} s ({n / dt:.1f} tok/s), engine stats: {engine.stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="miniserve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="serve the MiniServe engine over an OpenAI-compatible API")
    _engine_args(s)
    _server_args(s)
    b = sub.add_parser("baseline", help="serve the naive Hugging Face baseline")
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--served-model-name", default=None)
    b.add_argument("--device", default="cuda")
    _server_args(b)
    g = sub.add_parser("generate", help="generate offline from the command line")
    _engine_args(g)
    g.add_argument("--prompt", action="append")
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    {"serve": serve, "baseline": baseline, "generate": generate}[args.command](args)


if __name__ == "__main__":
    main()
