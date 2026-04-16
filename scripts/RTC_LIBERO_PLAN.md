# RTC Libero 训练与评测计划

## 1. 当前训练状态

**正在运行的训练任务：** lerobot 0.4.0 官方流程在 libero_spatial 上微调 SmolVLA

| 项目 | 详情 |
|------|------|
| 训练入口 | `lerobot-train`（lerobot 0.4.0 官方 CLI） |
| GPU | GPU 5（A100-80GB） |
| 当前步数 | ~4000 / 100,000 |
| Loss | 0.037（已收敛） |
| 预计完成时间 | ~30 小时（每步 1.24s） |
| WandB | https://wandb.ai/1250576969-/lerobot/runs/pvlpe88z |
| 输出目录 | `/root/project/lq/lerobot/outputs/lerobot_smolvla_libero_spatial` |
| Checkpoint 保存间隔 | 每 20000 步 |

## 2. 数据集

所有数据集位于 `/root/storage/DATA/libero_lerobot/`，格式为 LeRobot v2.1（lerobot 0.4.0 原生支持）。

| 数据集 | Episodes | Frames | 任务数 | 用途 |
|--------|----------|--------|--------|------|
| `libero_spatial_no_noops` | 432 | 52,970 | 10 | **当前训练中** |
| `libero_object_no_noops` | 454 | 66,984 | 10 | 待训练 |
| `libero_goal_no_noops` | 428 | 52,042 | 10 | 待训练 |
| `libero_10_no_noops` | 379 | 101,469 | 10 | 待训练 |
| `libero_90_no_noops` | 3,921 | 569,249 | 90 | 待训练（大规模） |

数据特征：
- 机器人：Franka，7 DoF 动作（6D 末端位姿 + 1D 夹爪）
- 观测：前视图 + 腕部相机（256×256），8 维状态（位置+轴角+夹爪）
- 采集频率：20 FPS

## 3. 模型架构

**SmolVLA**（Small Vision-Language-Action）

| 参数 | 值 |
|------|-----|
| VLM 骨干 | SmolVLM2-500M-Video-Instruct |
| VLM 层数 | 16（从原始模型裁剪） |
| 总参数量 | 450M |
| 可训练参数 | 100M（仅 expert 层 + state_proj） |
| 动作生成 | Flow Matching（10 步去噪） |
| Chunk 大小 | 50 步 |
| 注意力模式 | Cross Attention |
| Expert 宽度 | 0.75x |

**初始权重来源：** a2c2-libero 在 libero_spatial 上从零训练的 SmolVLA（80000 步 checkpoint），已通过 `migrate_policy_normalization.py` 迁移为 lerobot 0.4.0 processor 格式。

路径：`/root/project/lq/a2c2-libero/libero_smolvla_scratch/checkpoints/080000/pretrained_model_migrated`

## 4. a2c2-libero 已有评测结果（libero_spatial）

> **注意：以下结果均未使用官方 RTC。** a2c2-libero 的评测代码仅在动作层面做朴素拼接（旧 chunk 的 delay 步 + 新 chunk 的 horizon 步），predict_action_chunk 调用时没有传入任何 RTC 参数，flow matching 去噪过程中没有引导梯度。这是纯 chunk 管理策略的消融，不是 RTC 消融。

a2c2-libero 使用**朴素 chunk 重用策略（无 RTC）** 在 libero_spatial 上的评测结果（10 trials/task，共 10 tasks）：

| execute_horizon | inference_delay | 成功率 |
|-----------------|-----------------|--------|
| 1 | 0 | 59% |
| 5 | 0 | **68%** |
| 10 | 0 | 60% |
| 30 | 0 | 62% |
| 40 | 0 | 51% |
| 50 | 0 | 47% |
| 10 | 1 | **67%** |
| 10 | 3 | 62% |
| 10 | 5 | **65%** |
| 10 | 10 | 59% |
| 40 | 1 | 54% |
| 40 | 3 | 55% |
| 40 | 5 | 45% |
| 40 | 10 | 42% |

关键发现：
- **最优配置**：`execute_horizon=5, delay=0`（68%）和 `execute_horizon=10, delay=1`（67%）
- 较小的 execute_horizon（5~10）普遍优于大值（40~50）
- delay > 0 在大 horizon 时会显著降低成功率

## 5. 下一步训练计划

### 阶段一：完成 libero_spatial 训练与评测（当前）

1. **等待训练完成**（~30h），checkpoint 会保存在 step 20000/40000/60000/80000/100000
2. **评测 lerobot 训练的 checkpoint**，复现 a2c2 的 14 组消融实验：
   ```bash
   CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl \
   /root/miniconda/envs/lerobot/bin/python scripts/eval_libero_rtc.py \
       --checkpoint outputs/lerobot_smolvla_libero_spatial/checkpoints/last/pretrained_model \
       --suite libero_spatial \
       --mode rtc_ablation \
       --trials 10
   ```
3. **对比 a2c2 checkpoint 在同一评测脚本下的结果**（排除评测代码差异）：
   ```bash
   CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl \
   /root/miniconda/envs/lerobot/bin/python scripts/eval_libero_rtc.py \
       --checkpoint /root/project/lq/a2c2-libero/libero_smolvla_scratch/checkpoints/080000/pretrained_model_migrated \
       --suite libero_spatial \
       --mode rtc_ablation \
       --trials 10
   ```

### 阶段二：扩展到其他 LIBERO suites

按数据量从小到大训练：

| 顺序 | Suite | 预计训练步数 | 预计耗时 |
|------|-------|-------------|---------|
| 1 | libero_goal | 100K | ~30h |
| 2 | libero_object | 100K | ~30h |
| 3 | libero_10 | 100K | ~30h |
| 4 | libero_90 | 200K | ~60h |

启动命令模板：
```bash
bash scripts/train_libero_smolvla.sh libero_goal 100000 64 5
bash scripts/train_libero_smolvla.sh libero_object 100000 64 5
bash scripts/train_libero_smolvla.sh libero_10 100000 64 5
bash scripts/train_libero_smolvla.sh libero_90 200000 64 5
```

每个 suite 训练完成后立即进行评测。

### 阶段三：论文实验矩阵

最终需要完成的完整实验（每组 10 trials × 10 tasks = 100 episodes）：

| 实验维度 | 说明 |
|---------|------|
| **Suites** | libero_spatial, libero_object, libero_goal, libero_10（必选），libero_90（可选） |
| **Chunk 策略** | baseline（全 chunk 顺序）vs 朴素 chunk 重用（不同 horizon/delay 组合） |
| **消融参数** | 14 组 (delay, horizon) 组合 |
| **Checkpoint 来源** | a2c2 训练 vs lerobot 官方训练（对比训练框架差异） |

## 6. 关键文件路径

| 文件 | 说明 |
|------|------|
| `scripts/train_libero_smolvla.sh` | 训练脚本 |
| `scripts/eval_libero_rtc.py` | 评测脚本（朴素 chunk 重用消融） |
| `scripts/COMPARISON.md` | a2c2-libero 与 lerobot 官方方案对比 |
| `outputs/eval_libero_rtc/` | 评测结果输出目录 |

## 7. 环境依赖

```
conda activate lerobot          # lerobot 0.4.0, Python 3.10
GPU: NVIDIA A100-SXM4-80GB      # 使用 GPU 5（空闲）
```
