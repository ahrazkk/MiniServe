"""Draw the README charts from results/*.json (nothing typed in by hand).

python bench/plot.py        # writes docs/img/throughput.svg, ttft.svg, step_time.svg
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from summarize import ROOT, SYSTEMS, load  # noqa: E402

OUT = ROOT / "docs" / "img"
COLORS = {"s1": "#2a78d6", "s2": "#d95926", "s3": "#178f64"}  # colorblind-checked palette
INK, MUTED, GRID = "#1f1f1f", "#6c6a64", "#e4e2db"


def style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c6bd")
    ax.tick_params(colors=MUTED)


def by_concurrency(key: str, metric: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor="white")
    data = {tag: load(tag) for tag, _, _ in SYSTEMS}
    conc = sorted({c for rows in data.values() for c in rows})
    for tag, name, cls in SYSTEMS:
        rows = data[tag]
        xs = sorted(rows)
        ys = [rows[c][key] for c in xs]
        ax.plot(xs, ys, marker="o", lw=2.2, ms=6, color=COLORS[cls], label=name)
        ax.annotate(
            f"{ys[-1]:,.0f}"
            if key == "tput"
            else f"{ys[-1] / 1000:,.1f} s"
            if ys[-1] >= 1000
            else f"{ys[-1]:,.0f} ms",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(8, -4),
            color=COLORS[cls],
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(conc, [str(c) for c in conc])
    ax.set_xlim(conc[0] * 0.8, conc[-1] * 1.6)
    if key == "ttft50":
        ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper left")
    style(ax, metric, "concurrent clients", "tokens / s" if key == "tput" else "ms (log scale)")
    fig.text(
        0.01,
        0.01,
        "RTX 3060 Laptop (6 GB), Qwen3-0.6B BF16, 256 in / 128 out, greedy, streamed over HTTP",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def step_time() -> plt.Figure:
    d = json.loads((ROOT / "results" / "step_time_v1.json").read_text())["rows"]
    b = [r["batch"] for r in d]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="white")
    ax.plot(
        b,
        [r["ms_per_step"] for r in d],
        marker="o",
        lw=2.2,
        color=COLORS["s1"],
        label="measured decode step",
    )
    ax.plot(
        b,
        [r["memory_bound_ms"] for r in d],
        ls="--",
        lw=1.6,
        color=MUTED,
        label="memory-bandwidth floor (computed)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(b, [str(x) for x in b])
    ax.set_ylim(0, 62)
    ax.legend(frameon=False, loc="upper right")
    style(
        ax,
        "Decode step time vs batch size: flat, and far above the floor",
        "sequences in the batch",
        "ms per step",
    )
    fig.text(
        0.01,
        0.01,
        "Engine only (no HTTP). The gap is CPU kernel-launch overhead: "
        "~1,700 launches per step at ~28 us each under WSL2.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    charts = {
        "throughput.svg": by_concurrency(
            "tput", "Output throughput: batching vs one request at a time"
        ),
        "ttft.svg": by_concurrency("ttft50", "Median time to first token"),
        "step_time.svg": step_time(),
    }
    for name, fig in charts.items():
        fig.savefig(OUT / name, format="svg", facecolor="white")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
