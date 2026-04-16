#!/bin/bash
# run_spatial_eval.sh — libero_spatial 朴素 + 官方 RTC 全量评测（串行）
# 用法: CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl bash scripts/run_spatial_eval.sh

set -e

CKPT="/root/project/lq/a2c2-libero/libero_smolvla_scratch/checkpoints/080000/pretrained_model_migrated"
PYTHON="/root/miniconda/envs/lerobot/bin/python"
SCRIPT="scripts/eval_libero_rtc.py"
SUITE="libero_spatial"
TRIALS=10
OUT="outputs/eval_full"

cd /root/project/lq/lerobot

echo "============================================"
echo "[$(date '+%H:%M:%S')] 开始 ${SUITE} 朴素 chunk reuse 14 组消融"
echo "============================================"
$PYTHON $SCRIPT \
  --checkpoint $CKPT --suite $SUITE \
  --mode rtc_ablation --trials $TRIALS --no_video --resume \
  --out_dir ${OUT}/naive \
  2>&1 | tee ${OUT}/logs/log_${SUITE}_naive.txt

echo "============================================"
echo "[$(date '+%H:%M:%S')] 开始 ${SUITE} 官方 RTC 14 组消融"
echo "============================================"
$PYTHON $SCRIPT \
  --checkpoint $CKPT --suite $SUITE \
  --mode rtc_ablation --use_rtc --trials $TRIALS --no_video --resume \
  --out_dir ${OUT}/rtc \
  2>&1 | tee ${OUT}/logs/log_${SUITE}_rtc.txt

echo "============================================"
echo "[$(date '+%H:%M:%S')] ${SUITE} 全部完成！"
echo "============================================"
echo "朴素结果: ${OUT}/naive/results_${SUITE}.jsonl"
echo "RTC 结果: ${OUT}/rtc/results_${SUITE}.jsonl"
