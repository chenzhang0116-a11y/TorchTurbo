#!/usr/bin/env python3
"""DeepSeek V4.1 small-operator SFT smoke test (standalone, needs torchrun).

Drives TorchTurboEngine directly: causal cross-entropy on packed sequences,
forward, backward, gradient clipping, AdamW. Phase 1 pins SP=1 with the fused
sparse attention off.

Usage:
    torchrun --nproc_per_node=16 scripts/runs/deepseek_v41/run_sft_mini.py \\
        --model /path/to/deepseek-v4.1-pruned \\
        --steps 1 --seq_len 512 --sp_size 1 --ep_size 16

    # Forward and loss only, no backward
    torchrun --nproc_per_node=16 scripts/runs/deepseek_v41/run_sft_mini.py \\
        --model /path/to/deepseek-v4.1-pruned --forward_only
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _torchturbo_src() -> str:
    """Prefer TORCHTURBO_SRC_DIR, then a sibling checkout, then the in-repo src."""
    env_src = os.environ.get("TORCHTURBO_SRC_DIR")
    if env_src:
        return env_src
    sibling = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "torchturbo", "src"))
    if os.path.isdir(os.path.join(sibling, "torchturbo")):
        return sibling
    return os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "src"))


_REPO_SRC = _torchturbo_src()
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def _is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _set_device(local_rank: int) -> torch.device:
    if _is_npu():
        torch.npu.set_device(local_rank)
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _log(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message, flush=True)


def _parse_seqlens(args: argparse.Namespace) -> list[int]:
    if not args.seqlens:
        if args.seq_len <= 0:
            raise ValueError(f"--seq_len must be positive, got {args.seq_len}")
        return [args.seq_len]
    lengths = [int(part) for part in args.seqlens.split(",") if part.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"--seqlens must be positive integers, got {args.seqlens!r}")
    return lengths


def build_sft_batch(
    lengths: list[int],
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """One packed micro-batch shared by every rank.

    ``attention_mask`` carries segment ids (0 would be padding). Labels match
    ``input_ids``; the first token of each later segment is ``-100`` so the
    engine's global shift does not train across a packed boundary.
    """
    total = sum(lengths)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    high = max(vocab_size, 4)
    input_ids = torch.randint(3, high, (1, total), generator=generator)
    labels = input_ids.clone()
    cursor = 0
    attention_mask = torch.empty(1, total, dtype=torch.long)
    position_ids = torch.empty(1, total, dtype=torch.long)
    for segment_id, length in enumerate(lengths, start=1):
        attention_mask[0, cursor : cursor + length] = segment_id
        position_ids[0, cursor : cursor + length] = torch.arange(length)
        if segment_id > 1:
            labels[0, cursor] = -100
        cursor += length
    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "position_ids": position_ids.to(device),
        "labels": labels.to(device),
    }
    return batch


def _vocab_size(model_path: str) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text = getattr(config, "text_config", config)
    return int(text.vocab_size)


def _build_engine(args: argparse.Namespace):
    import torchturbo.models  # noqa: F401  registers deepseek_v41
    from torchturbo.engine import (
        ActivationCheckpointingConfig,
        CPUOffloadConfig,
        CPUOffloadMode,
        EngineConfig,
        FSDPConfig,
        LossChunkConfig,
        TorchTurboEngine,
    )
    from torchturbo.parallel import FSDPParallelPlan

    engine = TorchTurboEngine(
        EngineConfig(
            parallel_plan=FSDPParallelPlan(sp_size=args.sp_size, ep_size=args.ep_size),
            # Head is not its own FSDP unit yet; chunked CE stays off.
            loss_chunk=LossChunkConfig(enabled=False),
            # CSA2 slots cross checkpoint regions and need the non-reentrant path.
            activation_checkpointing=ActivationCheckpointingConfig(
                enabled=True,
                use_reentrant=False,
            ),
            # Adam moments stay on CPU. FULL offload is incompatible with the
            # host-offload engram tables.
            backend_config=FSDPConfig(
                cpu_offload=CPUOffloadConfig(
                    enabled=True,
                    mode=CPUOffloadMode.OPTIMIZER,
                ),
            ),
        )
    )
    if args.scratch:
        model = engine.build_model(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_weights=False,
        )
        engine.model = model
        engine.model_name_or_path = args.model
    else:
        model = engine.prepare_model(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    return engine, model


def run_step(engine, model, batch: dict[str, torch.Tensor], args: argparse.Namespace, step: int) -> None:
    model.train(not args.forward_only)
    if args.forward_only:
        with torch.no_grad():
            loss = engine.backend.compute_loss(model, dict(batch))
        _log(f"step {step} forward_only loss={loss.item():.6f}")
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        return

    engine.optimizer.zero_grad(set_to_none=True)
    loss = engine.backend.compute_loss(model, dict(batch))
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
    loss.backward()
    grad_norm = engine.backend.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
    grad_norm_value = float(grad_norm)
    if grad_norm_value != grad_norm_value:
        raise RuntimeError(f"non-finite grad_norm at step {step}")
    engine.optimizer.step()
    _log(
        f"step {step} loss={loss.item():.6f} grad_norm={grad_norm_value:.6f} "
        f"lr={args.lr}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 small-operator SFT smoke")
    parser.add_argument("--model", type=str, required=True, help="pruned-4layer or full checkpoint dir")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512, help="used when --seqlens is empty")
    parser.add_argument(
        "--seqlens",
        type=str,
        default="",
        help="comma-separated packed segment lengths, e.g. 128,256",
    )
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--ep_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.25e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch", action="store_true", help="skip checkpoint load")
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="compute loss and skip backward / optimizer step",
    )
    args = parser.parse_args()

    if args.sp_size != 1:
        raise SystemExit(
            "DeepSeek V4.1 sequence parallelism is not implemented. "
            f"Got --sp_size {args.sp_size}; this script requires --sp_size 1."
        )
    if args.steps <= 0:
        raise SystemExit(f"--steps must be positive, got {args.steps}")

    local_rank = _local_rank()
    device = _set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl" if _is_npu() else "nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size % args.ep_size != 0:
        raise SystemExit(
            f"world_size ({world_size}) must be divisible by --ep_size ({args.ep_size})"
        )

    lengths = _parse_seqlens(args)
    _log("=" * 60)
    _log("  DeepSeek V4.1 small-operator SFT")
    _log("=" * 60)
    _log(f"  Model: {args.model}")
    _log(f"  Steps: {args.steps}")
    _log(f"  Packed lengths: {lengths} (tokens={sum(lengths)})")
    _log(f"  SP size: {args.sp_size}")
    _log(f"  EP size: {args.ep_size}")
    _log(f"  World size: {world_size}")
    _log(f"  Forward only: {args.forward_only}")
    _log(f"  Scratch: {args.scratch}")
    _log("=" * 60)

    import torchturbo.models  # noqa: F401

    vocab_size = _vocab_size(args.model)
    batch = build_sft_batch(lengths, vocab_size, args.seed, device)
    dist.barrier()

    engine, model = _build_engine(args)
    if not args.forward_only:
        engine.prepare_optimizer(
            torch.optim.AdamW,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    for step in range(args.steps):
        run_step(engine, model, batch, args, step)
        dist.barrier()

    _log(f"[Rank {rank}] Done.")
    dist.barrier()


if __name__ == "__main__":
    main()
