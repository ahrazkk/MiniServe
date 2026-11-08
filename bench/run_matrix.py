"""One command per system: start its server, sweep concurrency with loadgen, stop the server.

    python bench/run_matrix.py --system miniserve --tag miniserve --concurrency 1 4 16 32 64
    python bench/run_matrix.py --system hf-baseline --tag hf-baseline --concurrency 1 4 16 \
        --num-prompts 64
    python bench/run_matrix.py --system miniserve --tag fixed-slots \
        --server-args "--kv-reservation full --no-prefix-caching"

Every run's JSON lands in results/ (named <tag>_in<IN>_out<OUT>_c<C>_r<RUN>.json); the server log
and an nvidia-smi log (clocks, temperature, power: watch for thermal throttling) go to
results/logs/.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def wait_healthy(url: str, server: subprocess.Popen[bytes], timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError("server exited during start-up; see its log in results/logs/")
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("server did not become healthy")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--system", choices=["miniserve", "hf-baseline"], required=True)
    p.add_argument("--tag", required=True, help="prefix for result files")
    p.add_argument("--model", default="/root/models/Qwen3-0.6B")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server-args", default="", help="extra flags for `miniserve serve`")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    p.add_argument(
        "--num-prompts",
        type=int,
        default=500,
        help="prompts per run at concurrency >= 16 (fewer below, noted in the JSON)",
    )
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--loadgen-args", default="", help="extra flags for loadgen.py")
    args = p.parse_args()

    logs = ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sub = "serve" if args.system == "miniserve" else "baseline"
    cmd = [
        "miniserve",
        sub,
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--served-model-name",
        "Qwen/Qwen3-0.6B",
        *shlex.split(args.server_args),
    ]
    print("server:", " ".join(cmd), flush=True)
    server_log = open(logs / f"{args.tag}_server.log", "w")  # noqa: SIM115
    server = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, cwd=ROOT)
    gpu_log = open(logs / f"{args.tag}_gpu.csv", "w")  # noqa: SIM115
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,"
            "temperature.gpu,power.draw",
            "--format=csv",
            "-l",
            "1",
        ],
        stdout=gpu_log,
    )
    url = f"http://127.0.0.1:{args.port}"
    try:
        wait_healthy(url, server)
        for c in args.concurrency:
            n = args.num_prompts if c >= 16 else min(args.num_prompts, max(20, 16 * c))
            for r in range(args.runs):
                label = f"{args.tag}_in{args.input_len}_out{args.output_len}_c{c}_r{r}"
                # A distinct seed per run: the server lives across runs, and repeating prompts
                # would hand the prefix cache free hits from the previous run.
                seed = 1000 * c + r
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "bench" / "loadgen.py"),
                        "--base-url",
                        url,
                        "--seed",
                        str(seed),
                        "--concurrency",
                        str(c),
                        "--num-prompts",
                        str(n),
                        "--input-len",
                        str(args.input_len),
                        "--output-len",
                        str(args.output_len),
                        "--label",
                        label,
                        "--system-config",
                        " ".join(cmd),
                        "--result-dir",
                        str(ROOT / "results"),
                        *shlex.split(args.loadgen_args),
                    ],
                    check=True,
                    cwd=ROOT,
                )
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        monitor.terminate()
        server_log.close()
        gpu_log.close()


if __name__ == "__main__":
    main()
