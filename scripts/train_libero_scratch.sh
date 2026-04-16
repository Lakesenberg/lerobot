#!/bin/bash
# =============================================================================
# 从零训练 SmolVLA on LIBERO（从 SmolVLM2-500M-Video-Instruct VLM 权重开始）
#
# 用法：
#   cd /root/project/lq/lerobot
#   bash scripts/train_libero_scratch.sh [suite] [steps] [batch_size] [gpu_id]
#
# 示例：
#   bash scripts/train_libero_scratch.sh libero_spatial 100000 64 6
# =============================================================================

set -e

SUITE=${1:-"libero_spatial"}
STEPS=${2:-100000}
BATCH_SIZE=${3:-64}
GPU_ID=${4:-6}

LEROBOT_TRAIN=/root/miniconda/envs/lerobot/bin/lerobot-train

declare -A SUITE_DATA_DIRS=(
    ["libero_spatial"]="/root/storage/DATA/libero_lerobot/libero_spatial_no_noops"
    ["libero_object"]="/root/storage/DATA/libero_lerobot/libero_object_no_noops"
    ["libero_goal"]="/root/storage/DATA/libero_lerobot/libero_goal_no_noops"
    ["libero_10"]="/root/storage/DATA/libero_lerobot/libero_10_no_noops"
    ["libero_90"]="/root/storage/DATA/libero_lerobot/libero_90_no_noops"
)

DATA_DIR="${SUITE_DATA_DIRS[$SUITE]}"
if [ -z "$DATA_DIR" ]; then
    echo "[ERROR] 未知 suite: $SUITE，可选: ${!SUITE_DATA_DIRS[@]}"
    exit 1
fi

OUTPUT_DIR="/root/project/lq/lerobot/outputs/smolvla_scratch_${SUITE}"
JOB_NAME="smolvla_scratch_${SUITE}"

echo "[INFO] ============================================================"
echo "[INFO] 从零训练模式（SmolVLM2 VLM backbone + 随机 expert）"
echo "[INFO] Suite:      ${SUITE}"
echo "[INFO] Data dir:   ${DATA_DIR}"
echo "[INFO] Steps:      ${STEPS}"
echo "[INFO] Batch size: ${BATCH_SIZE}"
echo "[INFO] GPU:        ${GPU_ID}"
echo "[INFO] Output:     ${OUTPUT_DIR}"
echo "[INFO] ============================================================"

if [ ! -f "${DATA_DIR}/meta/info.json" ]; then
    echo "[ERROR] 数据集不存在: ${DATA_DIR}/meta/info.json"
    exit 1
fi

# resume 逻辑
RESUME_FLAGS=""
if [ -d "${OUTPUT_DIR}/checkpoints" ]; then
    LAST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoints/*/pretrained_model 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CKPT" ] && [ -f "${LAST_CKPT}/train_config.json" ]; then
        echo "[INFO] 找到 checkpoint，开启 resume: ${LAST_CKPT}"
        RESUME_FLAGS="--resume=true --config_path=${LAST_CKPT}/train_config.json"
    else
        echo "[WARN] 输出目录存在但无有效 checkpoint，清理后重新训练..."
        rm -rf "${OUTPUT_DIR}"
    fi
elif [ -d "${OUTPUT_DIR}" ]; then
    echo "[WARN] 输出目录存在但无 checkpoints，清理后重新训练..."
    rm -rf "${OUTPUT_DIR}"
fi

echo "[INFO] 开始训练..."

CUDA_VISIBLE_DEVICES=${GPU_ID} \
HF_DATASETS_OFFLINE=1 \
${LEROBOT_TRAIN} \
    --policy.type=smolvla \
    --policy.load_vlm_weights=true \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.tokenizer_max_length=48 \
    --policy.num_steps=10 \
    --policy.vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
    --policy.num_vlm_layers=16 \
    --policy.attention_mode=cross_attn \
    --policy.optimizer_lr=0.0001 \
    --policy.optimizer_grad_clip_norm=10.0 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.repo_id="username/smolvla_scratch_${SUITE}" \
    --dataset.repo_id="local/libero_${SUITE}" \
    --dataset.root="${DATA_DIR}" \
    --batch_size=${BATCH_SIZE} \
    --steps=${STEPS} \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --wandb.enable=true \
    --save_freq=20000 \
    ${RESUME_FLAGS}

echo "[INFO] 训练完成，checkpoint: ${OUTPUT_DIR}"
