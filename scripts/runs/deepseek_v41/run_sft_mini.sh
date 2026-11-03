#!/bin/bash
# DeepSeek V4.1 small-operator SFT smoke.
#
# One node, 16 dies, SP=1 / EP=16. Fused sparse attention stays off.
# The Python entry runs packed causal CE: forward, backward, clip, AdamW.
#
#   bash scripts/runs/deepseek_v41/run_sft_mini.sh
#   STEPS=3 SEQ_LEN=1024 bash scripts/runs/deepseek_v41/run_sft_mini.sh
#   FORWARD_ONLY=1 bash scripts/runs/deepseek_v41/run_sft_mini.sh

set -o pipefail
set -eux

MODEL=${MODEL:-/models/deepseek-v4.1-flash-pruned}
SEQ_LEN=${SEQ_LEN:-512}
SEQLENS=${SEQLENS:-}
STEPS=${STEPS:-1}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-16}
LR=${LR:-1.25e-5}
FORWARD_ONLY=${FORWARD_ONLY:-0}
SCRATCH=${SCRATCH:-0}

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
# v4.1 first path is the torch small-operator implementation.
export TORCHTURBO_ENABLE_NPU_SFA_LI=0

export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="${SCRIPT_DIR}/run_sft_mini.py"
# Deployed layout: <root>/scripts next to <root>/torchturbo/src.
# In-repo layout: torchturbo/scripts/runs/deepseek_v41 -> ../../../src.
if [ -z "${TORCHTURBO_SRC_DIR:-}" ]; then
    if [ -d "${SCRIPT_DIR}/../torchturbo/src/torchturbo" ]; then
        TORCHTURBO_SRC_DIR="$(cd "${SCRIPT_DIR}/../torchturbo/src" && pwd)"
    elif [ -d "${SCRIPT_DIR}/../../../src/torchturbo" ]; then
        TORCHTURBO_SRC_DIR="$(cd "${SCRIPT_DIR}/../../../src" && pwd)"
    fi
fi
[ -n "${TORCHTURBO_SRC_DIR:-}" ] && [ -d "${TORCHTURBO_SRC_DIR}/torchturbo" ] || {
    echo "bad TORCHTURBO_SRC_DIR: ${TORCHTURBO_SRC_DIR:-<empty>}" >&2
    echo "expected ${SCRIPT_DIR}/../torchturbo/src" >&2
    exit 1
}
export TORCHTURBO_SRC_DIR
export PYTHONPATH="${TORCHTURBO_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [ "${SP_SIZE}" != "1" ]; then
    echo "DeepSeek V4.1 SFT smoke requires SP_SIZE=1, got ${SP_SIZE}" >&2
    exit 1
fi

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPROC_PER_NODE:-${NPUS_PER_NODE}} \
    --nnodes ${NNODES:-1} \
    --node_rank ${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT}
"

EXTRA_ARGS=()
if [ -n "${SEQLENS}" ]; then
    EXTRA_ARGS+=(--seqlens "${SEQLENS}")
else
    EXTRA_ARGS+=(--seq_len "${SEQ_LEN}")
fi
if [ "${FORWARD_ONLY}" = "1" ]; then
    EXTRA_ARGS+=(--forward_only)
fi
if [ "${SCRATCH}" = "1" ]; then
    EXTRA_ARGS+=(--scratch)
fi

torchrun ${DISTRIBUTED_ARGS} "${RUN_PY}" \
    --model "${MODEL}" \
    --steps "${STEPS}" \
    --sp_size "${SP_SIZE}" \
    --ep_size "${EP_SIZE}" \
    --lr "${LR}" \
    "${EXTRA_ARGS[@]}"
