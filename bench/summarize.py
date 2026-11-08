"""Turn results/*.json into the tables used by docs/RESULTS.md and the README.

    python bench/summarize.py           # markdown tables
    python bench/summarize.py --js      # plus a RESULTS_JS= line: the W1 numbers as JSON for charts

For each system and concurrency it reports the run with the *median* output throughput (and
lists every run's throughput, so the spread is visible). Nothing here is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = [  # (result tag, display name, chart color slot)
    ("ms", "MiniServe", "s1"),
    ("fixed-slots", "MiniServe, fixed KV slots", "s3"),
    ("hf-baseline", "HF baseline", "s2"),
]


def load(tag: str, input_len: int = 256, output_len: int = 128) -> dict[int, dict[str, Any]]:
    runs: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for f in sorted((ROOT / "results").glob(f"{tag}_in{input_len}_out{output_len}_c*_r*.json")):
        d = json.loads(f.read_text())
        runs.setdefault(int(d["config"]["concurrency"]), []).append((f.name, d))
    rows: dict[int, dict[str, Any]] = {}
    for c, rs in sorted(runs.items()):
        rs.sort(key=lambda r: r[1]["summary"]["output_throughput"])
        name, d = rs[(len(rs) - 1) // 2]  # median run (the lower one for an even count)
        s, srv = d["summary"], d.get("server") or {}
        rows[c] = {
            "file": name,
            "runs": len(rs),
            "tput_all": [round(r[1]["summary"]["output_throughput"], 1) for r in rs],
            "n": s["completed"],
            "failed": s["failed"],
            "tput": s["output_throughput"],
            "req": s["request_throughput"],
            "goodput": s["goodput_req_s"],
            "ttft50": s["ttft_ms"]["p50"],
            "ttft99": s["ttft_ms"]["p99"],
            "tpot50": s["tpot_ms"]["p50"],
            "tpot99": s["tpot_ms"]["p99"],
            "itl99": s["itl_ms"]["p99"],
            "e2el_mean": s["e2el_ms"]["mean"],
            "peak": srv.get("peak_running"),
            "preempt": srv.get("preemptions_total"),
            "prefix_hits": srv.get("prefix_cache_hit_tokens_total"),
            "date": d["environment"]["date"],
            "commit": d["environment"]["commit"],
        }
    return rows


def table(name: str, rows: dict[int, dict[str, Any]]) -> str:
    out = [
        f"### {name}",
        "",
        "| Clients | Requests | Output tok/s (all runs) | TTFT P50 / P99 ms | TPOT P50 / P99 ms "
        "| ITL P99 ms | Peak running | Preemptions | Little's law check | File |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c, r in rows.items():
        little = r["req"] * r["e2el_mean"] / 1000  # concurrency ~= throughput x latency
        runs = ", ".join(f"{t:.1f}" for t in r["tput_all"])
        peak = "-" if r["peak"] is None else f"{r['peak']:.0f}"
        pre = "-" if r["preempt"] is None else f"{r['preempt']:.0f}"
        out.append(
            f"| {c} | {r['n']} | **{r['tput']:.1f}** ({runs}) "
            f"| {r['ttft50']:.0f} / {r['ttft99']:.0f} "
            f"| {r['tpot50']:.1f} / {r['tpot99']:.1f} | {r['itl99']:.0f} | {peak} | {pre} "
            f"| {little:.1f} | `{r['file']}` |"
        )
    return "\n".join(out) + "\n"


def ticks(top: float, n: int = 5) -> tuple[float, list[float]]:
    raw = top * 1.1 / n
    step = min(
        (s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if s >= raw),
        default=raw,
    )
    hi = step * n
    return hi, [step * i for i in range(n + 1)]


def js(data: dict[str, dict[int, dict[str, Any]]]) -> str:
    conc = sorted({c for rows in data.values() for c in rows})
    tmax, tt = ticks(max(r["tput"] for rows in data.values() for r in rows.values()))
    pmax, pt = ticks(max(r["tpot50"] for rows in data.values() for r in rows.values()))
    ttft_hi = max(r["ttft50"] for rows in data.values() for r in rows.values())
    decades = [10.0]
    while decades[-1] < ttft_hi:
        decades.append(decades[-1] * 10)
    systems = []
    for tag, name, cls in SYSTEMS:
        rows = {
            c: {
                k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in r.items()
                if k
                in ("n", "tput", "ttft50", "ttft99", "tpot50", "tpot99", "itl99", "peak", "preempt")
            }
            for c, r in data.get(tag, {}).items()
        }
        systems.append({"name": name, "cls": cls, "rows": rows})
    obj = {
        "conc": conc,
        "systems": systems,
        "tputMax": tmax,
        "tputTicks": tt,
        "tpotMax": pmax,
        "tpotTicks": pt,
        "ttftMax": decades[-1],
        "ttftTicks": decades,
    }
    return json.dumps(obj, separators=(",", ":"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--js", action="store_true")
    args = p.parse_args()
    data = {tag: load(tag) for tag, _, _ in SYSTEMS}
    print("## W1: 256-token prompts, 128 output tokens, closed loop\n")
    for tag, name, _ in SYSTEMS:
        if data[tag]:
            print(table(name, data[tag]))
    on, off = load("prefix-on", 128), load("prefix-off", 128)
    if on and off:
        print("## W3: 1,024-token shared prefix + 128 unique tokens, 128 output tokens\n")
        print(table("Prefix caching on", on))
        print(table("Prefix caching off", off))
    if args.js:
        print("RESULTS_JS=" + js(data))


if __name__ == "__main__":
    main()
