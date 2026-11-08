"""Streaming text must equal the full decode, even when characters span several tokens."""

from __future__ import annotations

from typing import Any

import pytest

from miniserve.api.detokenizer import IncrementalDetokenizer, StopChecker


@pytest.mark.parametrize(
    "text",
    ["Hello world!", "Hello 世界 🚀!", "Emoji run: 🎉🎉🔥 done", "混合 text, 数字 123 and ümlauts"],
)
def test_streamed_pieces_join_to_the_full_text(qwen_tokenizer: Any, text: str) -> None:
    prompt = qwen_tokenizer.encode("Say:", add_special_tokens=False)
    ids = qwen_tokenizer.encode(text, add_special_tokens=False)
    detok = IncrementalDetokenizer(qwen_tokenizer, prompt)
    pieces = [detok.add([t]) for t in ids]
    assert "".join(pieces) == qwen_tokenizer.decode(ids)
    assert all("�" not in p for p in pieces)  # never a half-built character


def test_stop_string_cuts_output_and_is_never_streamed() -> None:
    stopper = StopChecker(["STOP"])
    sent = []
    for piece in ["Hello ", "wor", "ld S", "TO", "P and more"]:
        out, hit = stopper.add(piece)
        sent.append(out)
        if hit:
            break
    assert "".join(sent) == "Hello world "
    assert hit


def test_text_held_back_for_a_possible_stop_is_released() -> None:
    stopper = StopChecker(["###"])
    out1, _ = stopper.add("a#")  # "#" might start "###"
    out2, _ = stopper.add("b")  # it didn't
    assert out1 + out2 + stopper.flush() == "a#b"


def test_no_stop_strings_streams_everything_immediately() -> None:
    stopper = StopChecker([])
    assert stopper.add("abc") == ("abc", False)
