"""Prompt-lookup (n-gram) drafts: guess the next tokens by finding the current ending earlier on.

If the last 3 tokens are "the KV cache" and that phrase appeared before, propose whatever came
after it last time. It needs no draft model and costs microseconds, and it works well whenever the
output copies the input: summaries, code edits, RAG answers, "rewrite this paragraph".
"""

from __future__ import annotations

import numpy as np


def propose_ngram(token_ids: list[int], k: int, n_max: int = 3, n_min: int = 1) -> list[int]:
    """Up to k drafts: what followed the latest earlier match of the text's last n tokens."""
    if k <= 0:
        return []
    arr = np.asarray(token_ids, dtype=np.int64)
    length = len(arr)
    for n in range(n_max, n_min - 1, -1):
        if length <= n:
            continue
        suffix = arr[length - n :]
        # Every n-gram that starts before the suffix itself does.
        windows = np.lib.stride_tricks.sliding_window_view(arr[: length - 1], n)
        hits = np.flatnonzero((windows == suffix).all(axis=1))
        if hits.size:
            start = int(hits[-1]) + n  # most recent match; propose what followed it
            return [int(t) for t in arr[start : start + k]]
    return []
