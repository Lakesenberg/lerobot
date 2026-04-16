#!/bin/bash
# 三个数据集的全量评测（朴素 + RTC，各 14 组 × 10 trials）
# 用法: CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl bash scripts/run_all_eval.sh

set -e

PYTHON="/root/miniconda/envs/lerobot/bin/python"
SCRIPT="scripts/eval_libero_rtc.py"
TRIALS=10
OUT="outputs/eval_full"

cd /root/project/lq/lerobot

declare -A CKPTS=(
    ["libero_object"]="outputs/smolvla_scratch_libero_object/checkpoints/100000/pretrained_model_migrated"
    ["libero_goal"]="outputs/smolvla_scratch_libero_goal/checkpoints/100000/pretrained_model_migrated"
    ["libero_10"]="outputs/smolvla_scratch_libero_10/checkpoints/100000/pretrained_model_migrated"
)

for SUITE in libero_object libero_goal libero_10; do
    CKPT="${CKPTS[$SUITE]}"
    echo "============================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始 ${SUITE}"
    echo "  Checkpoint: ${CKPT}"
    echo "============================================"

    mkdir -p ${OUT}/naive ${OUT}/rtc ${OUT}/logs

    # 朴素 chunk reuse
    echo "[$(date '+%H:%M:%S')] ${SUITE} 朴素 chunk reuse 14 组"
    $PYTHON $SCRIPT \
        --checkpoint $CKPT --suite $SUITE \
        --mode rtc_ablation --trials $TRIALS --no_video --resume \
        --out_dir ${OUT}/naive \
        2>&1 | tee ${OUT}/logs/log_${SUITE}_naive.txt

    # 官方 RTC
    echo "[$(date '+%H:%M:%S')] ${SUITE} 官方 RTC 14 组"
    $PYTHON $SCRIPT \
        --checkpoint $CKPT --suite $SUITE \
        --mode rtc_ablation --use_rtc --trials $TRIALS --no_video --resume \
        --out_dir ${OUT}/rtc \
        2>&1 | tee ${OUT}/logs/log_${SUITE}_rtc.txt

    echo "[$(date '+%H:%M:%S')] ${SUITE} 完成！"
    echo ""
done

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部评测完成！"
echo "============================================"
