"""Guards for the first DeepSeek V4.1 small-operator SFT path."""

import json
import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("torch is required for DeepSeek V4.1 SFT guard tests", allow_module_level=True)
    raise

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safetensors.torch import save_file  # noqa: E402

from torchturbo.models.deepseek_v41.configuration_deepseek_v41 import (  # noqa: E402
    DeepseekV41TextConfig,
)
from torchturbo.models.deepseek_v41.model import TurboDeepseekV41ForCausalLM  # noqa: E402
from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import (  # noqa: E402
    DeepseekV41Indexer,
    DeepseekV41MoE,
    HostOffloadEngram,
)


def _text_config() -> DeepseekV41TextConfig:
    return DeepseekV41TextConfig(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        compress_ratios=[1],
        index_n_heads=2,
        index_head_dim=8,
        hc_mult=2,
        engram_layer_ids=[0],
        engram_num_embeddings=[64],
        engram_max_ngram_size=2,
        engram_n_heads=2,
        engram_head_dim=4,
    )


def test_moe_keeps_layer_idx_for_swiglu_limit():
    moe = DeepseekV41MoE(_text_config(), layer_idx=3, vision_enabled=False)
    assert moe.layer_idx == 3


def test_sp_greater_than_one_is_rejected():
    model = object.__new__(TurboDeepseekV41ForCausalLM)
    model.on_parallelized({"sp": {"sp_size": 1}})
    try:
        model.on_parallelized({"sp": {"sp_size": 2}})
    except NotImplementedError as exc:
        assert "sp_size=2" in str(exc)
    else:
        raise AssertionError("sp_size=2 should be rejected")


def test_indexer_parameters_are_frozen():
    holder = torch.nn.Module()
    holder.indexer = DeepseekV41Indexer(
        _text_config(), compress_ratio=1, owns_k=True, is_candidate_source=True
    )
    assert any(param.requires_grad for param in holder.indexer.parameters())
    TurboDeepseekV41ForCausalLM._freeze_indexers(holder)
    assert all(not param.requires_grad for param in holder.indexer.parameters())


def _write_engram_shard(directory: Path, engram: HostOffloadEngram) -> dict[str, torch.Tensor]:
    directory.mkdir(parents=True, exist_ok=True)
    # Build from shape/dtype so this also works while the module is still meta.
    payload = {
        f"layers.{engram.layer_idx}.engram.wkv.weight": torch.arange(
            engram.wkv.weight.numel(), dtype=engram.wkv.weight.dtype
        ).reshape(tuple(engram.wkv.weight.shape)),
        f"layers.{engram.layer_idx}.engram.q_weight": torch.full(
            tuple(engram.q_weight.shape), 0.25, dtype=engram.q_weight.dtype
        ),
        f"layers.{engram.layer_idx}.engram.k_weight": torch.full(
            tuple(engram.k_weight.shape), 0.5, dtype=engram.k_weight.dtype
        ),
    }
    filename = "model-00001-of-00001.safetensors"
    save_file(payload, directory / filename)
    index = {"weight_map": {key: filename for key in payload}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    return payload


def test_engram_load_params_replaces_meta_tensors(tmp_path):
    with torch.device("meta"):
        engram = HostOffloadEngram(_text_config(), layer_idx=0, layer_hash_index=0)
    assert engram.wkv.weight.is_meta
    expected = _write_engram_shard(tmp_path, engram)

    engram.load_params(str(tmp_path), device=torch.device("cpu"))

    assert not engram.wkv.weight.is_meta
    assert engram.wkv.weight.device.type == "cpu"
    torch.testing.assert_close(
        engram.wkv.weight, expected["layers.0.engram.wkv.weight"]
    )
    torch.testing.assert_close(engram.q_weight, expected["layers.0.engram.q_weight"])
    torch.testing.assert_close(engram.k_weight, expected["layers.0.engram.k_weight"])


def test_engram_load_params_copies_into_real_tensors(tmp_path):
    engram = HostOffloadEngram(_text_config(), layer_idx=0, layer_hash_index=0)
    assert not engram.wkv.weight.is_meta
    expected = _write_engram_shard(tmp_path, engram)

    engram.load_params(str(tmp_path), device=torch.device("cpu"))

    torch.testing.assert_close(
        engram.wkv.weight, expected["layers.0.engram.wkv.weight"]
    )
