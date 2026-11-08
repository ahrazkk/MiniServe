"""Load Hugging Face Qwen3 weights into our modules, fusing q/k/v and gate/up on the way in.

Checkpoint names look like `model.layers.0.self_attn.q_proj.weight`. Our modules use the same
names minus the `model.` prefix, except that q/k/v live in one `qkv_proj` matrix and gate/up in
one `gate_up_proj` matrix, so those tensors are copied into row slices of the fused weight.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn

from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM

# checkpoint projection -> (our fused module, which slice of it)
_FUSED = {
    "q_proj": ("qkv_proj", 0),
    "k_proj": ("qkv_proj", 1),
    "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _shard_rows(cfg: Qwen3Config) -> dict[str, list[tuple[int, int]]]:
    q = cfg.num_attention_heads * cfg.head_dim
    kv = cfg.num_key_value_heads * cfg.head_dim
    i = cfg.intermediate_size
    return {
        "qkv_proj": [(0, q), (q, q + kv), (q + kv, q + 2 * kv)],
        "gate_up_proj": [(0, i), (i, 2 * i)],
    }


def load_weights(model: Qwen3ForCausalLM, weights: Iterable[tuple[str, Tensor]]) -> None:
    """Copy (name, tensor) pairs from a Hugging Face Qwen3 checkpoint into `model`."""
    params = dict(model.named_parameters())
    rows = _shard_rows(model.cfg)
    loaded: set[tuple[str, int | None]] = set()
    for name, tensor in weights:
        if name == "lm_head.weight" and model.lm_head is None:
            continue  # tied embeddings: the LM head is the embedding matrix
        name = name.removeprefix("model.")
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] in _FUSED:
            fused, shard = _FUSED[parts[-2]]
            target = ".".join([*parts[:-2], fused, parts[-1]])
            start, end = rows[fused][shard]
            params[target].data[start:end].copy_(tensor)
            loaded.add((target, shard))
        elif name in params:
            params[name].data.copy_(tensor)
            loaded.add((name, None))
        else:
            raise ValueError(
                f"unexpected weight {name!r}: is this really a dense Qwen3 checkpoint?"
            )
    for pname in params:
        module = pname.split(".")[-2]
        shards = range(len(rows[module])) if module in rows else [None]
        missing = [s for s in shards if (pname, s) not in loaded]
        if missing:
            raise ValueError(f"checkpoint is missing {pname} (shards {missing})")


def load_qwen3(
    model_dir: str | Path, device: torch.device, dtype: torch.dtype, max_positions: int
) -> Qwen3ForCausalLM:
    cfg = Qwen3Config.from_pretrained(model_dir)
    # Build on the meta device (no memory, no random init), then allocate real storage directly
    # in the target dtype, so a 1.2 GB model never briefly exists as a 2.4 GB float32 copy.
    with default_dtype(dtype), torch.device("meta"):
        model = Qwen3ForCausalLM(cfg, max_positions)
    model = model.to_empty(device=device)
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")

    def tensors() -> Iterator[tuple[str, Tensor]]:
        for f in files:
            with safe_open(str(f), framework="pt", device=str(device)) as sf:
                for key in sf.keys():  # noqa: SIM118 (safe_open is not a dict)
                    if key == "lm_head.weight" and cfg.tie_word_embeddings:
                        continue  # don't even read it
                    yield key, sf.get_tensor(key)

    load_weights(model, tensors())
    model.rotary.build(device)
    return model.eval()


def from_hf_model(
    hf_model: nn.Module, device: torch.device, dtype: torch.dtype, max_positions: int
) -> Qwen3ForCausalLM:
    """Build our model from an in-memory Hugging Face Qwen3ForCausalLM (used by the tests)."""
    cfg = Qwen3Config.from_dict(hf_model.config.to_dict())  # type: ignore[operator]
    with default_dtype(dtype), torch.device(device):
        model = Qwen3ForCausalLM(cfg, max_positions)
    load_weights(model, ((k, v.to(dtype)) for k, v in hf_model.state_dict().items()))
    model.rotary.build(device)
    return model.eval()


def read_eos_token_ids(model_dir: str | Path) -> frozenset[int]:
    """Every end-of-sequence id the model declares (Qwen3: <|im_end|> and <|endoftext|>)."""
    ids: set[int] = set()
    for fname in ("generation_config.json", "config.json"):
        path = Path(model_dir) / fname
        if path.exists():
            value = json.loads(path.read_text()).get("eos_token_id")
            if isinstance(value, int):
                ids.add(value)
            elif isinstance(value, list):
                ids.update(int(v) for v in value)
    return frozenset(ids)
