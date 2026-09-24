"""Load Llama-3.2 weights from Hugging Face into TTTTransformer.

Name mapping (HF -> ours). HF stores nn.Linear weights as [out, in] and so do we,
so no transpose is needed; only the module paths differ.

    model.embed_tokens.weight                     -> embed_tokens.weight
    model.layers.{i}.self_attn.q_proj.weight      -> blocks.{i}.attn.wq.weight
    model.layers.{i}.self_attn.k_proj.weight      -> blocks.{i}.attn.wk.weight
    model.layers.{i}.self_attn.v_proj.weight      -> blocks.{i}.attn.wv.weight
    model.layers.{i}.self_attn.o_proj.weight      -> blocks.{i}.attn.wo.weight
    model.layers.{i}.mlp.gate_proj.weight         -> blocks.{i}.mlp.w1.weight
    model.layers.{i}.mlp.up_proj.weight           -> blocks.{i}.mlp.w3.weight
    model.layers.{i}.mlp.down_proj.weight         -> blocks.{i}.mlp.w2.weight
    model.layers.{i}.input_layernorm.weight       -> blocks.{i}.seq_norm.weight
    model.layers.{i}.post_attention_layernorm.w   -> blocks.{i}.ffn_norm.weight
    model.norm.weight                             -> norm.weight
    lm_head.weight                                -> lm_head.weight (absent when tied)

A LoRA-wrapped projection keeps its base matrix at `.weight`, so the same mapping
applies whether or not LoRA is enabled.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ttt.config import LoRAConfig, ModelConfig, RopeConfig
from ttt.model.transformer import TTTTransformer

# Ungated mirror with byte-identical weights and tokenizer, used when the gated
# meta-llama repo returns 403. Verified: same config, same token ids.
DEFAULT_REPO = "meta-llama/Llama-3.2-1B"
MIRROR_REPO = "unsloth/Llama-3.2-1B"


def model_config_from_hf(hf_config: dict, *, window_size: int, chunk_size: int,
                         fast_blocks: int, lora: LoRAConfig | None = None,
                         token_rates: bool = False) -> ModelConfig:
    """Build our ModelConfig from an HF config.json, asserting the fields we rely on."""
    assert hf_config["model_type"] == "llama", f"expected llama, got {hf_config['model_type']}"
    assert hf_config["hidden_act"] == "silu", "SwiGLUMLP assumes silu"
    assert not hf_config.get("attention_bias", False), "we build bias-free projections"
    assert not hf_config.get("mlp_bias", False), "we build bias-free projections"

    rs = hf_config.get("rope_scaling") or {}
    if rs:
        assert rs.get("rope_type") == "llama3", f"unsupported rope_scaling {rs}"
        rope = RopeConfig(
            theta=float(hf_config["rope_theta"]), scaling="llama3",
            factor=float(rs["factor"]), low_freq_factor=float(rs["low_freq_factor"]),
            high_freq_factor=float(rs["high_freq_factor"]),
            original_max_position=int(rs["original_max_position_embeddings"]),
        )
    else:
        rope = RopeConfig(theta=float(hf_config["rope_theta"]), scaling="none")

    return ModelConfig(
        vocab_size=int(hf_config["vocab_size"]),
        hidden_size=int(hf_config["hidden_size"]),
        intermediate_size=int(hf_config["intermediate_size"]),
        num_layers=int(hf_config["num_hidden_layers"]),
        num_heads=int(hf_config["num_attention_heads"]),
        num_kv_heads=int(hf_config["num_key_value_heads"]),
        rms_norm_eps=float(hf_config["rms_norm_eps"]),
        tie_word_embeddings=bool(hf_config["tie_word_embeddings"]),
        window_size=window_size, chunk_size=chunk_size, fast_blocks=fast_blocks,
        rope=rope, lora=lora or LoRAConfig(rank=0), token_rates=token_rates,
    )


def hf_key_map(num_layers: int, tied: bool) -> dict[str, str]:
    """HF parameter name -> our parameter name."""
    m = {"model.embed_tokens.weight": "embed_tokens.weight", "model.norm.weight": "norm.weight"}
    if not tied:
        m["lm_head.weight"] = "lm_head.weight"
    for i in range(num_layers):
        h, o = f"model.layers.{i}.", f"blocks.{i}."
        m[h + "self_attn.q_proj.weight"] = o + "attn.wq.weight"
        m[h + "self_attn.k_proj.weight"] = o + "attn.wk.weight"
        m[h + "self_attn.v_proj.weight"] = o + "attn.wv.weight"
        m[h + "self_attn.o_proj.weight"] = o + "attn.wo.weight"
        m[h + "mlp.gate_proj.weight"] = o + "mlp.w1.weight"
        m[h + "mlp.up_proj.weight"] = o + "mlp.w3.weight"
        m[h + "mlp.down_proj.weight"] = o + "mlp.w2.weight"
        m[h + "input_layernorm.weight"] = o + "seq_norm.weight"
        m[h + "post_attention_layernorm.weight"] = o + "ffn_norm.weight"
    return m


def load_hf_state_dict(repo_id: str, *, cache_dir: str | None = None) -> tuple[dict, dict]:
    """Download (or reuse) the safetensors shards. Returns (state_dict, config_json)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(snapshot_download(repo_id, cache_dir=cache_dir,
                                  allow_patterns=["*.safetensors", "config.json", "*.json"]))
    cfg = json.loads((path / "config.json").read_text())
    shards = sorted(path.glob("*.safetensors"))
    assert shards, f"no safetensors found in {path}"
    sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        sd.update(load_file(str(shard)))
    return sd, cfg


def load_into_model(model: TTTTransformer, hf_state: dict, *, strict: bool = True) -> None:
    """Copy HF tensors into `model` in place, asserting every shape matches."""
    tied = model.cfg.tie_word_embeddings
    mapping = hf_key_map(model.cfg.num_layers, tied)
    own = dict(model.named_parameters())
    missing, copied = [], 0
    for hf_name, our_name in mapping.items():
        if hf_name not in hf_state:
            missing.append(hf_name)
            continue
        src, dst = hf_state[hf_name], own[our_name]
        assert src.shape == dst.shape, f"{hf_name} {tuple(src.shape)} -> {our_name} {tuple(dst.shape)}"
        with torch.no_grad():
            dst.copy_(src.to(dst.dtype))
        copied += 1
    if strict:
        assert not missing, f"missing HF tensors: {missing[:5]}"
        expected = len(mapping)
        assert copied == expected, f"copied {copied} of {expected} tensors"


def build_llama_ttt(repo_id: str = MIRROR_REPO, *, max_seq_len: int, window_size: int = 8192,
                    chunk_size: int = 1024, fast_blocks: int = 4,
                    lora: LoRAConfig | None = None, dtype: torch.dtype = torch.float32,
                    cache_dir: str | None = None, token_rates: bool = False) -> TTTTransformer:
    """Fully-loaded TTTTransformer with pretrained Llama-3.2 weights."""
    hf_state, hf_cfg = load_hf_state_dict(repo_id, cache_dir=cache_dir)
    cfg = model_config_from_hf(hf_cfg, window_size=window_size, chunk_size=chunk_size,
                               fast_blocks=fast_blocks, lora=lora, token_rates=token_rates)
    model = TTTTransformer(cfg, max_seq_len=max_seq_len).to(dtype)
    load_into_model(model, hf_state)
    return model
