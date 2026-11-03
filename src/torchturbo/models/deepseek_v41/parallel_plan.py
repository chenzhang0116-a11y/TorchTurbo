"""FSDP2 parallel plan for DeepSeek V4.1.

Mirrors the deepseek_v4 plan (MoE replacement + EP-sharded experts) and adds
the engram replacement: the full-table ``DeepseekV41Engram`` becomes a
``HostOffloadEngram`` whose table stays on the host (EP-sharded, rows
fetched on demand), so the 2x183GiB tables never enter FSDP2 (a sharded
parameter would be all-gathered on every forward) nor the generic weight
loader (full-tensor broadcasts).
"""

from typing import Any

import torch

from ...parallel import FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.moe.moe_layer import MoEConfig, MoELayer, MoESubmodules

# v4.1 has no hc_head params (delayed hyper-connections); the engram module
# is no_shard at module level, so its params are not listed here either.
_NO_SHARD_PARAM_NAMES = [
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
    "attn_sink",
]


def _is_moe_module(module: Any) -> bool:
    return getattr(module, "experts", None) is not None and getattr(module, "gate", None) is not None


def _text_config_of(hf_config):
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


def _dtype_from_config(text_config):
    dtype = getattr(text_config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype = dtype.removeprefix("torch.")
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype, torch.float32)
    return torch.float32


def _replace_moe_only(
    original_module,
    hf_config,
    sp_info,
    ep_info,
    moe_submodules,
    **kwargs,
):
    if not _is_moe_module(original_module):
        return original_module

    text_config = _text_config_of(hf_config)
    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    swiglu_limit = getattr(text_config, "swiglu_limit", 0.0)
    num_layers = getattr(text_config, "num_hidden_layers", 0)
    moe_swiglu_limit = [swiglu_limit] * num_layers if swiglu_limit else None
    dsv41_config_kwargs = get_base_moe_config(
        hf_config=text_config,
        ep_size=ep_size,
        num_moe_experts=text_config.n_routed_experts,
        params_dtype=_dtype_from_config(text_config),
        moe_router_topk_scaling_factor=text_config.routed_scaling_factor,
        moe_router_score_function=text_config.scoring_func,
        moe_router_dtype="fp32",
        moe_swiglu_limit=moe_swiglu_limit,
        moe_router_enable_expert_bias=True,
    )
    moe_config = MoEConfig(**dsv41_config_kwargs)
    layer_number = getattr(original_module, "layer_idx", None)
    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        layer_number=layer_number,
    )

    if hasattr(original_module, "shared_experts"):
        moe_layer.shared_experts = original_module.shared_experts

    return moe_layer


def _replace_engram_only(
    original_module,
    hf_config,
    sp_info,
    ep_info,
    engine_config,
    **kwargs,
):
    from .modeling_deepseek_v41 import DeepseekV41Engram, HostOffloadEngram

    if not isinstance(original_module, DeepseekV41Engram):
        return original_module
    return HostOffloadEngram(
        _text_config_of(hf_config),
        original_module.layer_idx,
        original_module.layer_hash_index,
        ep_group=ep_info["ep_group"],
    )


def get_parallel_plan(**kwargs) -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ..modules.moe.experts import GroupedMLP
    from ..modules.moe.module import ModuleSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "layers.*": FSDPPlanSpec(activation_checkpoint=True),
            "layers.*.ffn.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "layers.*.ffn": FSDPPlanSpec(
                replace_module=_replace_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "layers.*.engram": FSDPPlanSpec(
                replace_module=_replace_engram_only,
                # The host-offload table is a CPU parameter and wkv/q/k read
                # their shard manually, so the whole module stays out of
                # FSDP; grads (once unfrozen) sync through the no_shard AVG
                # hooks for wkv/q/k and the table's own sparse path.
                # NOTE: engine cpu_offload FULL would try to move the host
                # table back to the device (unsupported combination).
                no_shard=True,
            ),
            **{name: FSDPPlanSpec(no_shard=True) for name in _NO_SHARD_PARAM_NAMES},
            **{
                f"**.{name}": FSDPPlanSpec(no_shard=True)
                for name in _NO_SHARD_PARAM_NAMES
            },
        },
        sp_size=1,
        ep_size=1,
    )
