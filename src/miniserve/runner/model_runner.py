"""Turn the scheduler's plan into one forward pass on the GPU, then sample the next tokens.

The runner
* owns the KV pool: one tensor [layers, 2, num_slots, kv_heads, head_dim] allocated at start-up
  and sized by profiling how much memory a worst-case step needs;
* flattens every scheduled sequence's new tokens into one batch (ids, positions, KV slots);
* computes logits only where a token will be sampled (each sequence's last token, or every
  position of a speculative verify), never for every prompt token;
* samples and verifies drafts with a single GPU->CPU sync per step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from miniserve.attention.backends import (
    MAX_GATHER_TOKENS,
    AttentionMetadata,
    SeqChunk,
    build_attention_metadata,
)
from miniserve.config import EngineConfig
from miniserve.engine.block_manager import cdiv
from miniserve.engine.scheduler import ScheduledSeq
from miniserve.engine.sequence import SamplingParams, Sequence
from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from miniserve.runner.sampler import compute_probs, exponential_noise, sample
from miniserve.spec.rejection import verify_greedy, verify_sampled
from miniserve.utils import to_device

logger = logging.getLogger(__name__)


@dataclass
class _StepInputs:
    input_ids: Tensor
    positions: Tensor
    attn: AttentionMetadata
    logits_rows: Tensor  # flat-batch rows whose logits we need: normal rows first, then spec rows
    sample_idx: list[int]  # scheduled index of each normally sampled sequence (one row each)
    spec: list[tuple[int, int]]  # (scheduled index, number of drafts): 1 + drafts rows each


class ModelRunner:
    def __init__(
        self,
        cfg: EngineConfig,
        model_cfg: Qwen3Config,
        model: Qwen3ForCausalLM,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.model = model
        self.device = device
        self.dtype = dtype
        self.block_size = cfg.block_size
        self.kv_cache: Tensor | None = None
        self.kv_caches: list[tuple[Tensor, Tensor]] = []
        self.num_blocks = 0
        self.spec_drafted = 0
        self.spec_accepted = 0
        self._pin = device.type == "cuda"

    # ---- the KV pool ----------------------------------------------------------------------

    @property
    def bytes_per_block(self) -> int:
        c = self.model_cfg
        return c.kv_bytes_per_token(self.dtype.itemsize) * self.block_size

    def allocate_kv_cache(self, num_blocks: int) -> None:
        c = self.model_cfg
        self.kv_cache, self.kv_caches = None, []  # release a previous (profiling) pool first
        # Zeros, not empty: masked-out slots still get multiplied by a zero attention weight,
        # and uninitialized memory can hold NaN bit patterns (0 x NaN = NaN).
        self.kv_cache = torch.zeros(
            c.num_hidden_layers,
            2,
            num_blocks * self.block_size,
            c.num_key_value_heads,
            c.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_caches = [
            (self.kv_cache[i, 0], self.kv_cache[i, 1]) for i in range(c.num_hidden_layers)
        ]
        self.num_blocks = num_blocks

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """Size the KV pool: what's left of the GPU after weights and a worst-case step.

        Same recipe as vLLM: run dummy steps at the largest sizes we allow, record peak memory,
        and give the KV pool  total * gpu_memory_utilization - everything else.
        """
        if self.device.type != "cuda":
            raise ValueError("KV pool profiling needs CUDA; set num_gpu_blocks instead")
        cfg, bs = self.cfg, self.block_size
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        weights = torch.cuda.memory_allocated()

        prefill_len = min(cfg.max_num_batched_tokens, cfg.max_model_len - 1)
        ctx = max(1, MAX_GATHER_TOKENS // cfg.max_num_seqs - 1)
        pool_blocks = cdiv(max(prefill_len, ctx + 1), bs)  # every dummy sequence shares this pool
        self.allocate_kv_cache(pool_blocks)
        pool = torch.cuda.memory_allocated() - weights
        torch.cuda.reset_peak_memory_stats()

        # 1) The biggest prefill: one prompt filling the whole token budget.
        big = Sequence("profile-prefill", [0] * prefill_len, SamplingParams())
        big.block_table = list(range(cdiv(prefill_len, bs)))
        self.execute([ScheduledSeq(big, prefill_len)])
        # 2) The biggest decode: max_num_seqs sequences filling one attention gather group,
        #    sampled with top-k/top-p (the sort is the sampler's biggest temporary).
        decodes = []
        for i in range(cfg.max_num_seqs):
            s = Sequence(f"profile-{i}", [0] * (ctx + 1), SamplingParams(top_k=50, top_p=0.9))
            s.num_computed = ctx
            s.block_table = [b % pool_blocks for b in range(cdiv(ctx + 1, bs))]
            decodes.append(ScheduledSeq(s, 1))
        self.execute(decodes)
        torch.cuda.synchronize()
        activations = torch.cuda.max_memory_allocated() - weights - pool

        self.kv_cache, self.kv_caches = None, []
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free  # our weights + the CUDA context + every other process on the GPU
        budget = total * cfg.gpu_memory_utilization - used - activations
        num_blocks = int(budget // self.bytes_per_block)
        logger.info(
            "memory profile: total %.0f MiB, in use %.0f MiB (weights %.0f), peak activations "
            "%.0f MiB -> KV pool %.0f MiB = %d blocks = %d tokens",
            total / 2**20,
            used / 2**20,
            weights / 2**20,
            activations / 2**20,
            num_blocks * self.bytes_per_block / 2**20,
            num_blocks,
            num_blocks * bs,
        )
        if num_blocks * bs < cfg.max_model_len:
            raise RuntimeError(
                f"only {max(num_blocks, 0)} KV blocks fit; not even one {cfg.max_model_len}-token "
                "request. Raise gpu_memory_utilization or lower max_model_len / budgets."
            )
        return num_blocks

    # ---- one step -------------------------------------------------------------------------

    def _to_device(self, data: list[int]) -> Tensor:
        return to_device(data, self.device)

    def prepare(self, scheduled: list[ScheduledSeq]) -> _StepInputs:
        """Flatten all scheduled tokens into one batch, the way Orca and vLLM do.

        Example: A decoding (1 token), B decoding (1 token), C prefilling 5 tokens:
            input_ids = [a7, b3, c0, c1, c2, c3, c4]
            positions = [ 7,  3,  0,  1,  2,  3,  4]   each token's position in its own sequence
            logits for rows 0 (A), 1 (B) and 6 (C), the last token of each sequence
        """
        bs = self.block_size
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        chunks: list[SeqChunk] = []
        normal_rows: list[int] = []
        spec_rows: list[int] = []
        sample_idx: list[int] = []
        spec: list[tuple[int, int]] = []
        for idx, s in enumerate(scheduled):
            seq, start, n = s.seq, s.seq.num_computed, s.num_new_tokens
            if seq.spec_token_ids:
                tokens = [seq.token_ids[-1], *seq.spec_token_ids]
            else:
                tokens = seq.token_ids[start : start + n]
            flat = len(input_ids)
            input_ids.extend(tokens)
            positions.extend(range(start, start + n))
            table = seq.block_table
            slots.extend(table[p // bs] * bs + p % bs for p in range(start, start + n))
            chunks.append(SeqChunk(flat, n, start, table))
            if seq.spec_token_ids:
                spec.append((idx, len(seq.spec_token_ids)))
                spec_rows.extend(range(flat, flat + n))
            elif start + n == seq.num_tokens:  # this chunk reaches the end: sample a token
                sample_idx.append(idx)
                normal_rows.append(flat + n - 1)
        slot_mapping = self._to_device(slots)
        return _StepInputs(
            input_ids=self._to_device(input_ids),
            positions=self._to_device(positions),
            attn=build_attention_metadata(chunks, slot_mapping, bs),
            logits_rows=self._to_device(normal_rows + spec_rows),
            sample_idx=sample_idx,
            spec=spec,
        )

    @torch.inference_mode()
    def execute(self, scheduled: list[ScheduledSeq]) -> list[list[int]]:
        """Run one step; return each scheduled sequence's new token ids (often 0 or 1 of them)."""
        results: list[list[int]] = [[] for _ in scheduled]
        if not scheduled:
            return results
        inp = self.prepare(scheduled)
        hidden = self.model(inp.input_ids, inp.positions, self.kv_caches, inp.attn)
        if inp.logits_rows.numel() == 0:
            return results  # only mid-prompt chunks this step: nothing to sample yet
        logits = self.model.compute_logits(hidden[inp.logits_rows])  # [rows, vocab] float32

        n_normal = len(inp.sample_idx)
        pending: list[Tensor] = []
        if n_normal:
            seqs = [scheduled[i].seq for i in inp.sample_idx]
            gens = [self._generator(s) for s in seqs]
            pending.append(sample(logits[:n_normal], [s.params for s in seqs], gens))
        if inp.spec:
            pending.extend(self._verify(logits[n_normal:], scheduled, inp.spec))
        flat = torch.cat(pending).tolist()  # the one GPU -> CPU sync of the step

        for j, idx in enumerate(inp.sample_idx):
            results[idx] = [flat[j]]
        if inp.spec:
            num = len(inp.spec)
            accepted = flat[n_normal : n_normal + num]
            next_tokens = flat[n_normal + num :]
            for (idx, k), a, t in zip(inp.spec, accepted, next_tokens, strict=True):
                results[idx] = [*scheduled[idx].seq.spec_token_ids[:a], t]
                self.spec_drafted += k
                self.spec_accepted += a
        return results

    def _verify(
        self, logits: Tensor, scheduled: list[ScheduledSeq], spec: list[tuple[int, int]]
    ) -> tuple[Tensor, Tensor]:
        """Check every speculating sequence's drafts against the target's own predictions."""
        seqs = [scheduled[idx].seq for idx, _ in spec]
        width = max(k for _, k in spec)
        rows, start = [], 0
        for _, k in spec:  # pad to [S, width + 1] rows by repeating each sequence's last row
            rows.append([start + min(j, k) for j in range(width + 1)])
            start += k + 1
        target_logits = logits[to_device(rows, self.device)]  # [S, width + 1, V]
        drafts = to_device(
            [s.spec_token_ids + [0] * (width - len(s.spec_token_ids)) for s in seqs], self.device
        )
        num_drafts = to_device([k for _, k in spec], self.device)
        accepted, next_tokens = verify_greedy(target_logits, drafts, num_drafts)
        if all(s.params.greedy for s in seqs):
            return accepted, next_tokens

        num, vocab = len(seqs), logits.shape[-1]
        params = [s.params for s in seqs for _ in range(width + 1)]
        probs = compute_probs(target_logits.reshape(-1, vocab), params).view(num, width + 1, vocab)
        gens = [self._generator(s) for s in seqs]
        uniform = torch.rand(num, width, device=self.device)
        for i, g in enumerate(gens):
            if g is not None:
                uniform[i] = torch.rand(width, generator=g, device=self.device)
        noise = exponential_noise((num, vocab), gens, self.device)
        acc_s, next_s = verify_sampled(probs, drafts, num_drafts, None, uniform, noise)
        greedy = to_device([s.params.greedy for s in seqs], self.device, torch.bool)
        return torch.where(greedy, accepted, acc_s), torch.where(greedy, next_tokens, next_s)

    def _generator(self, seq: Sequence) -> torch.Generator | None:
        if seq.params.seed is None:
            return None
        if seq.generator is None:
            seq.generator = torch.Generator(device=self.device).manual_seed(seq.params.seed)
        return seq.generator
