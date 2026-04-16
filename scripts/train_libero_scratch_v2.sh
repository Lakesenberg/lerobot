#!/bin/bash
# =============================================================================
# 使用 a2c2-libero 源码从零训练 SmolVLA on LIBERO
# （基于 conda lerobot 环境 + a2c2 PYTHONPATH）
#
# 用法：
#   cd /root/project/lq/lerobot
#   bash scripts/train_libero_scratch_v2.sh [suite] [steps] [batch_size] [gpu_id]
# =============================================================================

set -e

SUITE=${1:-"libero_spatial"}
STEPS=${2:-100000}
BATCH_SIZE=${3:-64}
GPU_ID=${4:-4}

PYTHON=/root/miniconda/envs/lerobot/bin/python
A2C2_SRC=/root/project/lq/a2c2-libero/src
CONFIG=/root/project/lq/lerobot/configs/train_scratch_${SUITE}.json

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

# 从配置文件读取 output_dir
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['output_dir'])")

echo "[INFO] ============================================================"
echo "[INFO] 从零训练模式 (a2c2 源码)"
echo "[INFO] Suite:      ${SUITE}"
echo "[INFO] Data dir:   ${DATA_DIR}"
echo "[INFO] Steps:      ${STEPS}"
echo "[INFO] Batch size: ${BATCH_SIZE}"
echo "[INFO] GPU:        ${GPU_ID}"
echo "[INFO] Config:     ${CONFIG}"
echo "[INFO] Output:     ${OUTPUT_DIR}"
echo "[INFO] ============================================================"

if [ ! -f "${DATA_DIR}/meta/info.json" ]; then
    echo "[ERROR] 数据集不存在: ${DATA_DIR}/meta/info.json"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] 配置文件不存在: ${CONFIG}"
    exit 1
fi

echo "[INFO] 使用已有配置: ${CONFIG}"

# resume 检查
RESUME_FLAG=""
if [ -d "${OUTPUT_DIR}/checkpoints" ]; then
    LAST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoints/*/pretrained_model 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CKPT" ] && [ -f "${LAST_CKPT}/train_config.json" ]; then
        echo "[INFO] 找到 checkpoint，开启 resume: ${LAST_CKPT}"
        CONFIG="${LAST_CKPT}/train_config.json"
        RESUME_FLAG="--resume=true"
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
PYTHONPATH=${A2C2_SRC}:${PYTHONPATH} \
${PYTHON} -m lerobot.scripts.train \
    --config_path="${CONFIG}" \
    ${RESUME_FLAG}

echo "[INFO] 训练完成，checkpoint: ${OUTPUT_DIR}"
