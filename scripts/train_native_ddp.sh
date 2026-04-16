#!/bin/bash
# =============================================================================
# 使用 lerobot 原生训练脚本 + Accelerate DDP 进行多卡训练
#
# 用法：
#   # 单卡
#   bash scripts/train_native_ddp.sh libero_spatial 1
#
#   # 多卡（自动选取空闲 GPU）
#   bash scripts/train_native_ddp.sh libero_spatial 2
#
#   # 指定 GPU
#   bash scripts/train_native_ddp.sh libero_spatial 3 "1,3,7"
#
#   # 后台运行所有 suite
#   bash scripts/train_native_ddp.sh all 2
#
# 参数：
#   $1  suite 名称: libero_spatial / libero_object / libero_goal / libero_10 / all
#   $2  GPU 数量 (默认 1)
#   $3  GPU IDs (可选，如 "1,3,7"，不指定则自动选取空闲卡)
#   $4  每卡 batch_size (可选，默认 64)
#   $5  总训练步数 (可选，默认 100000)
# =============================================================================

set -euo pipefail

SUITE=${1:-"libero_spatial"}
NUM_GPUS=${2:-1}
GPU_IDS=${3:-""}
BATCH_SIZE=${4:-64}
STEPS=${5:-100000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${PROJECT_DIR}/configs/native"
OUTPUT_BASE="${PROJECT_DIR}/outputs"
LOG_DIR="${OUTPUT_BASE}/logs"

CONDA_ENV="${CONDA_ENV:-lerobot312}"
PYTHON="/root/miniconda/envs/${CONDA_ENV}/bin/python"
ACCELERATE="/root/miniconda/envs/${CONDA_ENV}/bin/accelerate"

mkdir -p "$LOG_DIR"

# ── 自动选取空闲 GPU ──────────────────────────────────────────────
auto_select_gpus() {
    local need=$1
    local free_gpus
    free_gpus=$($PYTHON -c "
import subprocess, csv, io
out = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
    text=True
)
free = [row[0].strip() for row in csv.reader(io.StringIO(out)) if int(row[1].strip()) < 1000]
print(','.join(free[:$need]))
")
    if [ -z "$free_gpus" ]; then
        echo "[ERROR] 找不到 $need 张空闲 GPU" >&2
        exit 1
    fi
    local actual
    actual=$(echo "$free_gpus" | tr ',' '\n' | wc -l)
    if [ "$actual" -lt "$need" ]; then
        echo "[WARN] 仅找到 $actual 张空闲 GPU ($free_gpus)，少于请求的 $need 张" >&2
    fi
    echo "$free_gpus"
}

# ── 单 suite 训练函数 ─────────────────────────────────────────────
train_suite() {
    local suite=$1
    local num_gpus=$2
    local gpu_ids=$3
    local batch_size=$4
    local steps=$5

    local config="${CONFIG_DIR}/train_${suite}.json"
    if [ ! -f "$config" ]; then
        echo "[ERROR] 配置文件不存在: $config"
        return 1
    fi

    # 输出目录包含 GPU 数量和 batch_size 信息
    local effective_bs=$((batch_size * num_gpus))
    local out_dir="${OUTPUT_BASE}/smolvla_${suite}_native_${num_gpus}gpu_bs${effective_bs}"

    # 如果目录已存在，检查是否有 checkpoint 可以 resume
    local resume_args=""
    if [ -d "${out_dir}/checkpoints" ]; then
        local last_ckpt
        last_ckpt=$(ls -d "${out_dir}"/checkpoints/*/pretrained_model 2>/dev/null | sort -V | tail -1 || true)
        if [ -n "$last_ckpt" ] && [ -f "${last_ckpt}/train_config.json" ]; then
            echo "[INFO] 发现已有 checkpoint，启用 resume: ${last_ckpt}"
            config="${last_ckpt}/train_config.json"
            resume_args="--resume=true"
        fi
    fi

    local log_file="${LOG_DIR}/train_${suite}_${num_gpus}gpu.log"

    echo "┌──────────────────────────────────────────────────────────────"
    echo "│ lerobot 原生 DDP 训练"
    echo "│ Suite:         ${suite}"
    echo "│ GPU:           ${gpu_ids} (${num_gpus} 卡)"
    echo "│ Per-GPU BS:    ${batch_size}"
    echo "│ Effective BS:  ${effective_bs}"
    echo "│ Steps:         ${steps}"
    echo "│ Config:        ${config}"
    echo "│ Output:        ${out_dir}"
    echo "│ Log:           ${log_file}"
    echo "└──────────────────────────────────────────────────────────────"

    local accel_args="--num_processes=${num_gpus} --mixed_precision=no"
    if [ "$num_gpus" -gt 1 ]; then
        accel_args="--multi_gpu ${accel_args}"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    ${ACCELERATE} launch \
        ${accel_args} \
        -m lerobot.scripts.lerobot_train \
        --config_path="${config}" \
        --output_dir="${out_dir}" \
        --batch_size="${batch_size}" \
        --steps="${steps}" \
        ${resume_args} \
        2>&1 | tee "${log_file}"

    echo "[INFO] ${suite} 训练完成，checkpoint: ${out_dir}"
}

# ── 主逻辑 ────────────────────────────────────────────────────────
if [ -z "$GPU_IDS" ]; then
    GPU_IDS=$(auto_select_gpus "$NUM_GPUS")
    echo "[INFO] 自动选取 GPU: ${GPU_IDS}"
fi

# 重新计算实际 GPU 数量（以 GPU_IDS 为准）
ACTUAL_NUM=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$ACTUAL_NUM" -ne "$NUM_GPUS" ]; then
    echo "[INFO] GPU IDs 数量 ($ACTUAL_NUM) 与 NUM_GPUS ($NUM_GPUS) 不一致，以 GPU IDs 为准"
    NUM_GPUS=$ACTUAL_NUM
fi

# 单卡时不需要 --multi_gpu 标志
if [ "$NUM_GPUS" -eq 1 ]; then
    SINGLE_GPU_MODE=true
else
    SINGLE_GPU_MODE=false
fi

if [ "$SUITE" == "all" ]; then
    echo "[INFO] 启动全部 4 个 suite 的训练..."
    for s in libero_spatial libero_object libero_goal libero_10; do
        echo ""
        echo "================================================================"
        echo "  开始 ${s}"
        echo "================================================================"
        train_suite "$s" "$NUM_GPUS" "$GPU_IDS" "$BATCH_SIZE" "$STEPS"
    done
    echo "[INFO] 全部训练完成！"
else
    train_suite "$SUITE" "$NUM_GPUS" "$GPU_IDS" "$BATCH_SIZE" "$STEPS"
fi
