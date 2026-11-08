"""Turning a stream of token ids back into a stream of text, correctly.

One token is not always valid text on its own: "🚀" is three byte-level tokens for Qwen, and
each one alone decodes to the replacement character "�". So we decode a small sliding window,
`ids[prefix_offset:]`, compare it with the text of the window before the newest tokens, and only
emit the new part once it no longer ends in "�". This is the same algorithm vLLM and TGI use.
"""

from __future__ import annotations

from typing import Protocol

_REPLACEMENT = "�"


class _Tokenizer(Protocol):
    def decode(self, token_ids: list[int], skip_special_tokens: bool = ...) -> str: ...


class IncrementalDetokenizer:
    def __init__(self, tokenizer: _Tokenizer, prompt_ids: list[int], context: int = 6) -> None:
        self.tokenizer = tokenizer
        # A few prompt tokens of context, so the first output token gets the right spacing.
        self.ids = list(prompt_ids[-context:])
        self.prefix_offset = 0  # start of the window we decode
        self.read_offset = len(self.ids)  # end of the text already emitted

    def add(self, new_ids: list[int]) -> str:
        """Append tokens; return the newly completed text (empty if a character is unfinished)."""
        if not new_ids:
            return ""
        self.ids.extend(new_ids)
        prefix_text = self._decode(self.ids[self.prefix_offset : self.read_offset])
        full_text = self._decode(self.ids[self.prefix_offset :])
        if len(full_text) > len(prefix_text) and not full_text.endswith(_REPLACEMENT):
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            return full_text[len(prefix_text) :]
        return ""

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class StopChecker:
    """Cut the output at the first stop string, never streaming any part of the stop string.

    While stop strings are set, the last (longest stop - 1) characters are held back: they might
    be the start of a stop string that the next token completes. They are released once it's
    clear they aren't, or at the end.
    """

    def __init__(self, stops: list[str]) -> None:
        self.stops = [s for s in stops if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.text = ""
        self.emitted = 0

    def add(self, delta: str) -> tuple[str, bool]:
        """Feed new text. Returns (text that is safe to send now, whether a stop string hit)."""
        old_len = len(self.text)
        self.text += delta
        if self.stops:
            start = max(0, old_len - self.hold)  # a stop string may straddle the old/new boundary
            hits = [i for s in self.stops if (i := self.text.find(s, start)) != -1]
            if hits:
                self.text = self.text[: min(hits)]
                return self.flush(), True
        safe_end = len(self.text) - self.hold
        if safe_end <= self.emitted:
            return "", False
        out = self.text[self.emitted : safe_end]
        self.emitted = safe_end
        return out, False

    def flush(self) -> str:
        out = self.text[self.emitted :]
        self.emitted = len(self.text)
        return out
