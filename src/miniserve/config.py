"""Every engine setting in one place. The defaults are the ones the benchmarks use."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DType = Literal["bfloat16", "float16", "float32", "float64"]


class EngineConfig(BaseModel):
    # Model
    model: str = Field(description="Hugging Face model directory (config.json + *.safetensors)")
    served_model_name: str | None = None
    dtype: DType = "bfloat16"
    device: str = "cuda"
    seed: int = 0

    # KV-cache memory
    block_size: int = Field(16, gt=0, description="Tokens per KV block (vLLM's default is 16)")
    max_model_len: int = Field(4096, gt=0, description="Longest prompt + output we accept")
    gpu_memory_utilization: float = Field(0.80, gt=0.0, le=1.0)
    num_gpu_blocks: int | None = Field(
        None, gt=0, description="Fix the KV pool size (skip profiling)"
    )
    kv_reservation: Literal["paged", "full"] = Field(
        "paged",
        description="'paged' hands out blocks on demand. 'full' reserves max_model_len tokens per "
        "request up front, like servers before PagedAttention (the baseline for the ablation).",
    )

    # Scheduling
    max_num_seqs: int = Field(64, gt=0, description="Most sequences in one batch")
    max_num_batched_tokens: int = Field(2048, gt=0, description="Token budget per engine step")
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    max_waiting: int = Field(
        1024, ge=0, description="Waiting-queue cap; beyond it the API sends 503"
    )

    # Speculative decoding (prompt lookup / n-gram drafts)
    speculative: Literal["none", "ngram"] = "none"
    num_speculative_tokens: int = Field(4, ge=1, le=7)
    ngram_max: int = Field(3, ge=1)
    ngram_min: int = Field(1, ge=1)

    # Weights
    quantization: Literal["none", "int8"] = "none"

    @model_validator(mode="after")
    def _check(self) -> EngineConfig:
        # Without chunking, a whole prompt has to fit in one step's token budget.
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            self.max_num_batched_tokens = self.max_model_len
        if self.ngram_min > self.ngram_max:
            raise ValueError("ngram_min must be <= ngram_max")
        return self
