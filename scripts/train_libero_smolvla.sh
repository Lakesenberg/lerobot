#!/bin/bash
# =============================================================================
# 使用 conda lerobot 0.4.0 环境训练 SmolVLA on LIBERO
#
# 用法：
#   cd /root/project/lq/lerobot
#   bash scripts/train_libero_smolvla.sh [suite] [steps] [batch_size] [gpu_id]
#
# 示例：
#   bash scripts/train_libero_smolvla.sh libero_spatial 100000 64 5
# =============================================================================

set -e

# ── 参数 ──────────────────────────────────────────────────────────────────────
SUITE=${1:-"libero_spatial"}
STEPS=${2:-100000}
BATCH_SIZE=${3:-64}
GPU_ID=${4:-5}

LEROBOT_TRAIN=/root/miniconda/envs/lerobot/bin/lerobot-train

# 数据目录映射
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

OUTPUT_DIR="/root/project/lq/lerobot/outputs/lerobot_smolvla_${SUITE}"
JOB_NAME="lerobot_smolvla_${SUITE}"

# a2c2 checkpoint（迁移后的版本，含 policy_preprocessor.json）
A2C2_CKPT="/root/project/lq/a2c2-libero/libero_smolvla_scratch/checkpoints/080000/pretrained_model_migrated"

# ── 前置检查 ──────────────────────────────────────────────────────────────────
echo "[INFO] ============================================================"
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

# ── resume 逻辑 ──────────────────────────────────────────────────────────────
# lerobot 0.4.0 resume 需要 --config_path 指向 checkpoint 中的 train_config.json
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

# ── 训练 ──────────────────────────────────────────────────────────────────────
echo "[INFO] 开始训练..."

CUDA_VISIBLE_DEVICES=${GPU_ID} \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
${LEROBOT_TRAIN} \
    --policy.path="${A2C2_CKPT}" \
    --policy.repo_id="username/lerobot_smolvla_${SUITE}" \
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
