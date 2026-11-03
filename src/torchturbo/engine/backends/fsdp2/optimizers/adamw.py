"""AdamW construction and optimizer-state CPU offload for FSDP2."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import chain, groupby
from typing import Any, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from .....logging import get_logger
from .....runtime import DeviceRuntime, TurboDevice
from ....optimizers.plan import ParamGroup
from ..config import CPUOffloadConfig, CPUOffloadMode
from .state import wrap_cpu_state

__all__ = [
    "AdamWSwap",
    "prepare_fsdp_adamw_optimizer",
]

logger = get_logger(__name__)


def prepare_fsdp_adamw_optimizer(
    groups: Sequence[ParamGroup],
    optimizer_kwargs: Mapping[str, Any],
    *,
    device_runtime: DeviceRuntime,
    cpu_offload: CPUOffloadConfig,
    split_param_groups: Callable[
        [Sequence[Mapping[str, Any]]],
        list[dict[str, Any]],
    ],
) -> Optimizer:
    """Build native AdamW, or AdamWSwap under optimizer-state offload.

    Native AdamW plans and the Muon fallback both use this function, so they
    apply the same offload policy and FSDP topology split.
    """
    param_groups = split_param_groups(
        tuple(group.get_optimizer_param_group() for group in groups)
    )
    options = dict(optimizer_kwargs)
    if cpu_offload.enabled and cpu_offload.mode is CPUOffloadMode.OPTIMIZER:
        # AdamWSwap always runs the fused kernel and has no fused option.
        options.pop("fused", None)
        return AdamWSwap(
            param_groups,
            **options,
            device_runtime=device_runtime,
            optimizer_memory_fraction=cpu_offload.optimizer_memory_fraction,
            param_names={
                param: name
                for group in groups
                for name, param in group.params_by_fqn.items()
            },
        )
    return torch.optim.AdamW(param_groups, **options)


class DeviceWorkspace:
    """Byte-backed temporary device storage that yields typed tensor views."""

    _ALIGNMENT_BYTES = 16

    def __init__(
        self,
        device_runtime: DeviceRuntime,
        memory_fraction: float,
    ) -> None:
        """Reserve temporary device storage from currently free memory."""
        self._storage: torch.Tensor | None = None
        if device_runtime.turbo_device is TurboDevice.CPU:
            return

        device = device_runtime.device
        free_memory, _ = device_runtime.memory_info()
        capacity = int(free_memory * memory_fraction)
        capacity = self._reduce_capacity_to_minimum(capacity, device)
        if capacity < self._ALIGNMENT_BYTES:
            raise RuntimeError(
                f"workspace capacity {capacity} bytes is too small "
                f"(memory_fraction={memory_fraction}); increase "
                "optimizer_memory_fraction or reduce device memory usage"
            )
        self._storage = torch.empty(capacity, dtype=torch.uint8, device=device)
        logger.info_rank0(
            "FSDP2 workspace reserved: device=%s, capacity=%d MiB, "
            "memory_fraction=%.3f",
            device.type,
            capacity // (1024 * 1024),
            memory_fraction,
        )

    @staticmethod
    def _reduce_capacity_to_minimum(capacity: int, device: torch.device) -> int:
        if not (dist.is_available() and dist.is_initialized()):
            return capacity

        capacity_tensor = torch.tensor([capacity], dtype=torch.int64, device=device)
        dist.all_reduce(
            capacity_tensor,
            op=dist.ReduceOp.MIN,
            group=dist.group.WORLD,
        )
        return int(capacity_tensor.item())

    @property
    def is_available(self) -> bool:
        return self._storage is not None

    def _require_storage(self) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("workspace is unavailable")
        return self._storage

    @property
    def device(self) -> torch.device:
        return self._require_storage().device

    @property
    def capacity_bytes(self) -> int:
        return self._require_storage().numel()

    @classmethod
    def align(cls, offset: int) -> int:
        alignment = cls._ALIGNMENT_BYTES
        return (offset + alignment - 1) // alignment * alignment

    def view_like(self, offset: int, tensor: torch.Tensor) -> torch.Tensor:
        nbytes = tensor.numel() * tensor.element_size()
        if offset % tensor.element_size() != 0:
            raise ValueError("workspace offset is not aligned for tensor dtype")
        if offset + nbytes > self.capacity_bytes:
            raise RuntimeError("workspace capacity exceeded")
        raw = self._require_storage().narrow(0, offset, nbytes)
        return raw.view(tensor.dtype).view(tensor.shape)


class AdamWSwapMixin:
    """Run AdamW with CPU-resident optimizer state."""

    state_keys = ("exp_avg", "exp_avg_sq")

    def __init__(
        self,
        device_runtime: DeviceRuntime,
        optimizer_memory_fraction: float,
        param_names: Mapping[torch.Tensor, str] | None = None,
    ) -> None:
        self.device_runtime = device_runtime
        self.optimizer_memory_fraction = optimizer_memory_fraction
        self.param_to_name_map = (
            dict(param_names) if param_names is not None else {}
        )
        self.param_to_cpu_states_map = {}
        self.param_to_step_map = {}
        self.param_to_state_bytes_map = {}
        self.workspace: DeviceWorkspace | None = None
        self._workspace_adamw_logged = False
        self._checkpoint_meshes: dict[DeviceMesh, DeviceMesh] = {}
        self._opt_states_initialization()

    def _acquire_workspace(self) -> DeviceWorkspace | None:
        """Allocate the per-step workspace for AdamW state staging."""
        self.release_workspace()
        self.device_runtime.synchronize()
        workspace = DeviceWorkspace(
            self.device_runtime,
            self.optimizer_memory_fraction,
        )
        self.workspace = workspace if workspace.is_available else None
        return self.workspace

    def release_workspace(self) -> None:
        """Synchronize and release the workspace."""
        if self.workspace is None:
            return
        self.device_runtime.synchronize()
        capacity_mib = self.workspace.capacity_bytes // (1024 * 1024)
        self.workspace = None
        logger.info_rank0(
            "FSDP2 workspace released: capacity=%d MiB",
            capacity_mib,
        )

    def _opt_states_initialization(self) -> None:
        """Zero the swap state and publish it under ``self.state``, to avoid
        the helper faking a zero-grad / lr-0 ``step()``, which would decay the
        CPU moments and bump the step counters on every checkpoint save/load.
        """
        for group in self.param_groups:
            for param in group["params"]:
                local_param = param.to_local() if isinstance(param, DTensor) else param
                cpu_state = {
                    key: torch.empty_like(
                        local_param,
                        device="cpu",
                        pin_memory=self.optimizer_memory_fraction > 0,
                    ).zero_()
                    for key in self.state_keys
                }
                step = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=(
                        "cpu"
                        if self.optimizer_memory_fraction == 0
                        else local_param.device
                    ),
                )
                state = {"step": step, **cpu_state}
                self.state[param] = state
                self.param_to_cpu_states_map[param] = state
                self.param_to_step_map[param] = step
                self.param_to_state_bytes_map[param] = sum(
                    tensor.untyped_storage().nbytes() for tensor in cpu_state.values()
                )

    def _run_adamw(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        states: list[dict[str, torch.Tensor]],
        steps: list[torch.Tensor],
        group: dict,
    ) -> None:
        for step in steps:
            step.add_(1)
        beta1, beta2 = group["betas"]
        torch._fused_adamw_(
            params,
            grads,
            [state["exp_avg"] for state in states],
            [state["exp_avg_sq"] for state in states],
            [],
            steps,
            lr=group["lr"],
            beta1=beta1,
            beta2=beta2,
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            amsgrad=False,
            maximize=group["maximize"],
        )

    def _step_on_device(self, params_with_groups) -> None:
        if self._acquire_workspace() is None:
            raise RuntimeError(
                "workspace is unavailable; AdamW device update requires a "
                "CUDA/NPU runtime"
            )
        self._ensure_workspace_capacity(params_with_groups)
        self._step_on_device_workspace(params_with_groups)

    def _ensure_workspace_capacity(self, params_with_groups) -> None:
        assert self.workspace is not None
        error = self._workspace_capacity_error(params_with_groups)
        if not (dist.is_available() and dist.is_initialized()):
            if error is not None:
                raise RuntimeError(error)
            return

        status = torch.tensor(
            [error is None], dtype=torch.int64, device=self.workspace.device
        )
        dist.all_reduce(
            status,
            op=dist.ReduceOp.MIN,
            group=dist.group.WORLD,
        )
        if int(status.item()) == 1:
            return

        if error is not None:
            raise RuntimeError(error)
        raise RuntimeError(
            "Per-step workspace capacity validation failed on another rank"
        )

    def _workspace_capacity_error(self, params_with_groups) -> str | None:
        if not params_with_groups:
            return None

        param, _ = max(
            params_with_groups,
            key=lambda item: self.param_to_state_bytes_map[item[0]],
        )
        if self._workspace_state_fits(self.param_to_cpu_states_map[param]):
            return None

        state_bytes = self.param_to_state_bytes_map[param]
        return (
            f"optimizer state for parameter '{self._parameter_name(param)}' "
            "does not fit in the per-step workspace "
            f"({state_bytes} bytes required per state pair, "
            f"workspace={self.workspace.capacity_bytes} bytes)"
        )

    def _parameter_name(self, param: torch.Tensor) -> str:
        return self.param_to_name_map.get(param, "<unnamed parameter>")

    def _wrap_checkpoint_tensor(
        self, tensor: torch.Tensor, param: torch.Tensor
    ) -> torch.Tensor:
        return wrap_cpu_state(tensor, param, self._checkpoint_meshes)

    def _workspace_state_fits(self, state: dict[str, torch.Tensor]) -> bool:
        assert self.workspace is not None
        second_base = self.workspace.align(self.workspace.capacity_bytes // 2)
        capacities = (second_base, self.workspace.capacity_bytes - second_base)
        return all(
            state[key].numel() * state[key].element_size() <= capacity
            for key, capacity in zip(self.state_keys, capacities)
        )

    def _stage_workspace_batch(self, params_with_groups, index: int):
        assert self.workspace is not None
        first_key, second_key = self.state_keys
        second_base = self.workspace.align(self.workspace.capacity_bytes // 2)
        first_capacity = second_base
        second_capacity = self.workspace.capacity_bytes - second_base
        if first_capacity <= 0 or second_capacity <= 0:
            return [], index

        first_offset = 0
        second_offset = 0
        entries = []
        while index < len(params_with_groups):
            param, group = params_with_groups[index]
            cpu_state = self.param_to_cpu_states_map[param]
            first_offset = self.workspace.align(first_offset)
            second_offset = self.workspace.align(second_offset)
            first_size = (
                cpu_state[first_key].numel() * cpu_state[first_key].element_size()
            )
            second_size = (
                cpu_state[second_key].numel() * cpu_state[second_key].element_size()
            )
            if (
                first_offset + first_size > first_capacity
                or second_offset + second_size > second_capacity
            ):
                break

            state = {
                first_key: self.workspace.view_like(first_offset, cpu_state[first_key]),
                second_key: self.workspace.view_like(
                    second_base + second_offset, cpu_state[second_key]
                ),
            }
            entries.append((param, group, state))
            first_offset += first_size
            second_offset += second_size
            index += 1
        return entries, index

    def _step_device_entries(self, entries) -> None:
        for _, group_entries in groupby(entries, key=lambda entry: id(entry[1])):
            batch = list(group_entries)
            group = batch[0][1]
            params = [
                param.to_local() if isinstance(param, DTensor) else param
                for param, _, _ in batch
            ]
            grads = [
                param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                for param, _, _ in batch
            ]
            self._run_adamw(
                params,
                grads,
                [state for _, _, state in batch],
                [self.param_to_step_map[param] for param, _, _ in batch],
                group,
            )

    def _step_on_device_workspace(self, params_with_groups) -> None:
        """Run AdamW with reusable state views from the fixed workspace slab."""
        assert self.workspace is not None

        index = 0
        if not self._workspace_adamw_logged:
            logger.info_rank0(
                "FSDP2 workspace: AdamW is using a %d MiB slab for %d parameters",
                self.workspace.capacity_bytes // (1024 * 1024),
                len(params_with_groups),
            )
            self._workspace_adamw_logged = True
        while index < len(params_with_groups):
            entries, next_index = self._stage_workspace_batch(
                params_with_groups, index
            )
            if not entries:
                param = params_with_groups[index][0]
                state_bytes = self.param_to_state_bytes_map[param]
                raise RuntimeError(
                    "optimizer state for parameter "
                    f"'{self._parameter_name(param)}' does not fit in the "
                    "per-step workspace "
                    f"({state_bytes} bytes required per state pair, "
                    f"workspace={self.workspace.capacity_bytes} bytes)"
                )

            for param, _, state in entries:
                for key in self.state_keys:
                    state[key].copy_(
                        self.param_to_cpu_states_map[param][key], non_blocking=True
                    )
            self.device_runtime.synchronize()

            self._step_device_entries(entries)

            for param, _, state in entries:
                for key in self.state_keys:
                    self.param_to_cpu_states_map[param][key].copy_(
                        state[key], non_blocking=True
                    )
            self.device_runtime.synchronize()
            index = next_index

    def _step_on_host(self, params_with_groups) -> None:
        # One parameter at a time. Pinning a whole FSDP group of params and
        # grads overflows rtMallocHost when every local rank shares one host.
        for param, group in params_with_groups:
            local_param = param.to_local() if isinstance(param, DTensor) else param
            local_grad = (
                param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
            )
            cpu_param = torch.empty_like(local_param, device="cpu", pin_memory=True)
            cpu_grad = torch.empty_like(local_grad, device="cpu", pin_memory=True)
            cpu_param.copy_(local_param.detach())
            cpu_grad.copy_(local_grad)
            self._run_adamw(
                [cpu_param],
                [cpu_grad],
                [self.param_to_cpu_states_map[param]],
                [self.param_to_step_map[param]],
                group,
            )
            local_param.copy_(cpu_param)
            del cpu_param, cpu_grad


class AdamWSwap(torch.optim.AdamW, AdamWSwapMixin):
    """Fused AdamW that keeps optimizer state on CPU between steps."""

    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        maximize: bool = False,
        device_runtime: DeviceRuntime,
        optimizer_memory_fraction: float,
        param_names: Mapping[torch.Tensor, str] | None = None,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            foreach=False,
            maximize=maximize,
            capturable=False,
            differentiable=False,
            fused=True,
        )
        # Let GradScaler unscale and check overflow before the low-level fused call.
        self._step_supports_amp_scaling = False
        AdamWSwapMixin.__init__(
            self, device_runtime, optimizer_memory_fraction, param_names
        )

    def state_dict(self, *args, **kwargs):
        """Return a standard AdamW-format checkpoint view of the swap state.

        Sharded moments are wrapped in metadata-only CPU DTensors while the
        state and param_groups keys remain standard packed integer IDs.
        """
        state_dict = super().state_dict(*args, **kwargs)
        params = chain.from_iterable(group["params"] for group in self.param_groups)
        for (param_id, entry), param in zip(state_dict["state"].items(), params):
            checkpoint_entry = dict(entry)
            for key in self.state_keys:
                checkpoint_entry[key] = self._wrap_checkpoint_tensor(
                    entry[key], param
                )
            state_dict["state"][param_id] = checkpoint_entry
        return state_dict

    def load_state_dict(self, state_dict):
        """Restore swap state from a standard AdamW-format state dict.

        Unlike the base implementation, the loaded tensors are copied into the
        existing buffers instead of being adopted.
        """
        state_dict = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            replacement = hook(self, state_dict)
            if replacement is not None:
                state_dict = replacement
        saved_groups = state_dict["param_groups"]
        groups = self.param_groups
        if len(groups) != len(saved_groups):
            raise ValueError(
                f"loaded state dict has {len(saved_groups)} parameter groups; "
                f"optimizer has {len(groups)}"
            )
        if [len(group["params"]) for group in groups] != [
            len(group["params"]) for group in saved_groups
        ]:
            raise ValueError(
                "loaded state dict has a different number of parameters per "
                "group than the optimizer"
            )

        for group, saved in zip(groups, saved_groups):
            for key, value in saved.items():
                if key != "params":
                    group[key] = value

        current_params = [p for group in groups for p in group["params"]]
        saved_ids = [p for group in saved_groups for p in group["params"]]
        param_map = dict(zip(saved_ids, current_params))

        for saved_id, saved_state in state_dict["state"].items():
            param = param_map.get(saved_id)
            if param is None:
                raise ValueError(
                    "loaded state dict contains state for unknown parameter "
                    f"id {saved_id!r}"
                )
            cpu_state = self.param_to_cpu_states_map.get(param)
            if cpu_state is None:
                raise ValueError(
                    "loaded state dict contains state for a parameter without "
                    "swap state"
                )
            for key, value in saved_state.items():
                if isinstance(value, DTensor):
                    value = value.to_local()
                if key == "step":
                    step = self.param_to_step_map[param]
                    if not torch.is_tensor(value):
                        value = torch.tensor(value)
                    self._validate_state_shape(param, key, value, step)
                    step.copy_(value)
                elif key in self.state_keys:
                    self._validate_state_shape(param, key, value, cpu_state[key])
                    cpu_state[key].copy_(value)
                # Other keys (e.g. amsgrad ``max_exp_avg_sq``) have no swap
                # counterpart and are ignored.
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)

    def _validate_state_shape(
        self,
        param: torch.Tensor,
        key: str,
        value: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        if value.shape != destination.shape:
            raise ValueError(
                f"loaded optimizer state '{key}' for parameter "
                f"'{self._parameter_name(param)}' has shape "
                f"{tuple(value.shape)}, expected {tuple(destination.shape)}"
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        try:
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            params_with_groups = [
                (param, group)
                for group in self.param_groups
                for param in group["params"]
                if param.grad is not None
            ]
            if self.optimizer_memory_fraction == 0:
                self._step_on_host(params_with_groups)
            else:
                self._step_on_device(params_with_groups)
        finally:
            self.release_workspace()
        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        """Release a stale workspace when AMP skips the optimizer update."""
        self.release_workspace()
        return super().zero_grad(set_to_none=set_to_none)
