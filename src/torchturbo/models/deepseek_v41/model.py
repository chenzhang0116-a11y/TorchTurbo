"""Turbo wrapper for DeepSeek V4.1 (registration, weight converter, hooks)."""

from functools import wraps

import torch
from transformers import AutoModelForCausalLM

from ..base import ParameterGroup, TurboVLMModel
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from .configuration_deepseek_v41 import DeepseekV41Config
from .modeling_deepseek_v41 import DeepseekV41ForCausalLM, RowShardedEngram


class DeepseekV41WeightConverter(BaseWeightConverter):
    """Checkpoint names are the native inference format; expert weights and
    the router bias are the only renames needed."""

    def __init__(self, model, parallel_context, model_path):
        super().__init__(model, parallel_context, model_path)
        if hasattr(self.config, "text_config"):
            # Nested VLM config: shape helpers read text fields.
            self.config = self.config.text_config
            self.model_type = self.config.model_type

    def get_expert_gate_name(self):
        return "w1"

    def get_expert_up_name(self):
        return "w3"

    def get_expert_down_name(self):
        return "w2"

    def get_hf_expert_mlp_name(self):
        return "ffn"

    def get_hf_gate_bias_name(self):
        return "gate.bias"


@register_model(
    "deepseek_v41", config_cls=DeepseekV41Config, auto_class=AutoModelForCausalLM
)
class TurboDeepseekV41ForCausalLM(TurboVLMModel, DeepseekV41ForCausalLM):
    config_class = DeepseekV41Config
    base_model_prefix = ""

    def __init__(self, config: DeepseekV41Config):
        self._pre_init_patches(config)
        super().__init__(config)
        self._post_init_patches()
        self._model_path = None

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def get_weight_converter(self, parallel_context, model_path):
        # Stash the path for on_materialized (engram hash + row-sharded
        # tables), which runs after checkpoint loading.
        self._model_path = model_path
        return DeepseekV41WeightConverter(self, parallel_context, model_path)

    def get_lm_head(self):
        return self.head

    def get_norm(self):
        return self.norm

    def get_sequence_parallel_input_slicing_spec(self):
        # The packed RL/SFT path slices input_ids before embedding (v4-style),
        # which matches the TurboModel default; declare it explicitly because
        # TurboVLMModel raises NotImplementedError here.
        from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec

        return SequenceParallelInputSlicingSpec(
            self,
            type(self).forward,
            sliced_inputs=(("input_ids", 1), ("position_ids", -1)),
        )

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        if self.config.vision_enabled:
            groups[ParameterGroup.VISION_TOWER] = self.vision.named_parameters(
                prefix="vision"
            )
            groups[ParameterGroup.MULTI_MODAL_PROJECTOR] = (
                self.aligner.named_parameters(prefix="aligner")
            )
        groups[ParameterGroup.LANGUAGE_MODEL] = (
            (name, param)
            for name, param in self.named_parameters()
            if not name.startswith(("vision.", "aligner."))
        )
        return groups

    @wraps(DeepseekV41ForCausalLM.forward)
    def forward(self, *args, **kwargs):
        lm_head = self._modules["head"]
        noop_lm_head = self.head
        if noop_lm_head is lm_head:
            return super().forward(*args, **kwargs)

        def head_forward(x, norm):
            return noop_lm_head(norm(x).float())

        object.__setattr__(self, "head", head_forward)
        try:
            return super().forward(*args, **kwargs)
        finally:
            object.__setattr__(self, "head", noop_lm_head)

    @staticmethod
    def _pre_init_patches(config):
        from torchturbo.models.modules.moe.moe_layer import MoELayer

        # MoELayer returns (output, aux); the v4.1 modeling contract is a
        # plain tensor (same adaptation as deepseek_v4).
        origin_attr = "_torchturbo_deepseek_v41_origin_forward"
        origin_moe_forward = getattr(MoELayer, origin_attr, MoELayer.forward)
        MoELayer._torchturbo_deepseek_v41_origin_forward = origin_moe_forward

        def moe_forward(self, hidden_states, input_ids=None):
            output, _ = origin_moe_forward(self, hidden_states, input_ids)
            return output

        MoELayer.forward = moe_forward
        config._attn_implementation = "eager"

    def _post_init_patches(self):
        self._use_flash_attention_2 = True
        self.config._attn_implementation = "flash_attention_2"
        # Phase 1 is text-only: the vision path is never exercised, so the
        # tower/projector stay frozen (weights still load for exchange).
        for module_name in ("vision", "aligner"):
            module = getattr(self, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)
        for param_name in ("image_start", "image_end", "image_newline"):
            param = getattr(self, param_name, None)
            if param is not None:
                param.requires_grad_(False)

    def on_materialized(self):
        """Build engram runtime state after parameters are on a real device.

        The n-gram hash buffers need the tokenizer; the row-sharded engram
        tables are read straight from the checkpoint shards (they are not
        parameters, so the generic loader never sees them).
        """
        from torchturbo.runtime import get_device_runtime

        device = get_device_runtime().device
        if self.engram_hash is not None:
            self.engram_hash.build(device, self._model_path)
        for module in self.modules():
            if isinstance(module, RowShardedEngram):
                if self._model_path is not None:
                    module.load_table(self._model_path, device)
                else:
                    module.init_table_random()
