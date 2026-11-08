"""Checks against the real Qwen3-0.6B on the GPU. Local only:  pytest -m gpu -s

In BF16, a sequence's logits differ slightly between Hugging Face and us (fused matmuls, other
attention kernels) and between batch shapes (different reduction orders), so these tests report
*rates* and only assert generous floors; the exact-equality proofs run in float64 on the CPU.
"""

from __future__ import annotations

import os
import random

import pytest
import torch
from helpers import full_logits

from miniserve.config import EngineConfig
from miniserve.engine.engine import LLMEngine
from miniserve.engine.sequence import SamplingParams
from miniserve.model.loader import load_qwen3

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]
MODEL = os.environ.get("MINISERVE_MODEL", "/root/models/Qwen3-0.6B")
PROMPTS = [
    "The capital of Canada is",
    "def fibonacci(n):",
    "Once upon a time, in a small village,",
    "The three laws of thermodynamics are",
    "Translate to French: I love programming.",
    "SELECT name FROM users WHERE",
    "The mitochondria is the",
    "To make a cup of tea, first",
    "In 1969, Neil Armstrong",
    "A binary search tree is",
    "import numpy as np\nx = np.",
    "The best way to learn a language is",
    "Toronto is a city in",
    "GPU memory bandwidth matters because",
    "Q: What is 12 times 12?\nA:",
    "The quick brown fox",
    "Photosynthesis converts",
    "HTTP status code 404 means",
    "My favourite programming language is",
    "The KV cache stores",
]


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


def test_next_token_matches_hugging_face(tokenizer) -> None:  # type: ignore[no-untyped-def]
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    ours = load_qwen3(MODEL, torch.device("cuda"), torch.bfloat16, 512)
    matches, worst = 0, 0.0
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tokenizer.encode(prompt)
            ref = hf(torch.tensor([ids], device="cuda")).logits[0, -1].float()
            got = full_logits(ours, ids)[-1]
            matches += int(ref.argmax() == got.argmax())
            worst = max(worst, (ref - got).abs().max().item())
    print(f"\ntop-1 agreement {matches}/{len(PROMPTS)}, max |logit diff| {worst:.3f}")
    del hf, ours
    torch.cuda.empty_cache()
    assert matches >= len(PROMPTS) - 1


def test_greedy_generations_match_hugging_face(tokenizer) -> None:  # type: ignore[no-untyped-def]
    from transformers import AutoModelForCausalLM

    n = 48
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    refs = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")
            out = hf.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=n,
                min_new_tokens=n,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=None,
            )
            refs.append(out[0, ids.shape[1] :].tolist())
    del hf
    torch.cuda.empty_cache()
    engine = LLMEngine(EngineConfig(model=MODEL, max_model_len=1024, gpu_memory_utilization=0.7))
    outs = engine.generate(
        [tokenizer.encode(p) for p in PROMPTS],
        SamplingParams(max_tokens=n, temperature=0, ignore_eos=True),
    )
    exact = sum(o == r for o, r in zip(outs, refs, strict=True))
    prefix = [
        next((i for i, (a, b) in enumerate(zip(o, r, strict=True)) if a != b), n)
        for o, r in zip(outs, refs, strict=True)
    ]
    print(
        f"\nexact {n}-token matches {exact}/{len(PROMPTS)}, mean matching prefix "
        f"{sum(prefix) / len(prefix):.1f} tokens"
    )
    assert exact >= len(PROMPTS) // 2


def test_no_kv_leaks_after_many_requests_with_aborts() -> None:
    engine = LLMEngine(EngineConfig(model=MODEL, max_model_len=1024, gpu_memory_utilization=0.7))
    rng = random.Random(0)
    for i in range(300):
        prompt = [rng.randrange(1000, 100000) for _ in range(rng.randrange(8, 300))]
        engine.add_request(
            str(i),
            prompt,
            SamplingParams(max_tokens=rng.randrange(1, 64), temperature=1.0, ignore_eos=True),
        )
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
        if steps % 7 == 0 and engine.scheduler.running:  # hang up on someone mid-generation
            engine.abort_request(engine.scheduler.running[-1].request_id)
    bm = engine.block_manager
    assert bm.num_free_blocks == bm.num_blocks
    bm.check_invariants([])
