#!/bin/bash
# v2 模型全量评测（朴素 + RTC，各 14 组 × 10 trials）
# 三个 suite 并行在不同 GPU 上跑
# 用法: bash scripts/run_eval_v2.sh

set -e

PYTHON="/root/miniconda/envs/lerobot/bin/python"
SCRIPT="scripts/eval_libero_rtc.py"
TRIALS=10
OUT="outputs/eval_v2"

cd /root/project/lq/lerobot

declare -A CKPTS=(
    ["libero_object"]="outputs/smolvla_scratch_libero_object_v2/checkpoints/100000/pretrained_model_migrated"
    ["libero_goal"]="outputs/smolvla_scratch_libero_goal_v2/checkpoints/100000/pretrained_model_migrated"
    ["libero_10"]="outputs/smolvla_scratch_libero_10_v2/checkpoints/100000/pretrained_model_migrated"
)

declare -A NORM_CKPTS=(
    ["libero_object"]="outputs/smolvla_scratch_libero_object_v2/checkpoints/100000/pretrained_model/model.safetensors"
    ["libero_goal"]="outputs/smolvla_scratch_libero_goal_v2/checkpoints/100000/pretrained_model/model.safetensors"
    ["libero_10"]="outputs/smolvla_scratch_libero_10_v2/checkpoints/100000/pretrained_model/model.safetensors"
)

eval_suite() {
    local SUITE=$1
    local GPU=$2
    local CKPT="${CKPTS[$SUITE]}"
    local NORM="${NORM_CKPTS[$SUITE]}"

    mkdir -p ${OUT}/naive ${OUT}/rtc ${OUT}/logs

    echo "[$(date '+%H:%M:%S')] GPU${GPU} ${SUITE} 朴素 chunk reuse 开始"
    CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl $PYTHON $SCRIPT \
        --checkpoint $CKPT --suite $SUITE \
        --norm_checkpoint $NORM \
        --mode rtc_ablation --trials $TRIALS --no_video --resume \
        --out_dir ${OUT}/naive \
        2>&1 | tee ${OUT}/logs/log_${SUITE}_naive.txt

    echo "[$(date '+%H:%M:%S')] GPU${GPU} ${SUITE} 官方 RTC 开始"
    CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl $PYTHON $SCRIPT \
        --checkpoint $CKPT --suite $SUITE \
        --norm_checkpoint $NORM \
        --mode rtc_ablation --use_rtc --trials $TRIALS --no_video --resume \
        --out_dir ${OUT}/rtc \
        2>&1 | tee ${OUT}/logs/log_${SUITE}_rtc.txt

    echo "[$(date '+%H:%M:%S')] GPU${GPU} ${SUITE} 全部完成！"
}

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] v2 全量评测开始"
echo "============================================"

eval_suite libero_object 4 &
eval_suite libero_goal   5 &
eval_suite libero_10     6 &

wait

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] v2 全量评测全部完成！"
echo "============================================"
echo "朴素结果: ${OUT}/naive/"
echo "RTC 结果: ${OUT}/rtc/"
