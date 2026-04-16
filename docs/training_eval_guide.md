# SmolVLA LIBERO 训练与评估完整指南

> 适用环境：GPU03 服务器，8×A100-SXM4-80GB，conda 环境 `lerobot`
> 代码库：`/root/project/lq/lerobot`（a2c2-libero 源码位于 `/root/project/lq/a2c2-libero`）

---

## 1. 环境与目录结构

```bash
# 激活环境
conda activate lerobot

# 关键路径
PROJECT=/root/project/lq/lerobot
A2C2_SRC=/root/project/lq/a2c2-libero/src
PYTHON=/root/miniconda/envs/lerobot/bin/python

# 目录结构
lerobot/
├── configs/                           # 训练配置文件 (JSON)
│   └── train_scratch_libero_*.json
├── scripts/
│   ├── train_libero_scratch_v2.sh     # 训练启动脚本
│   ├── eval_libero_rtc.py             # 评测脚本（Naive + RTC）
│   ├── run_eval_v2.sh                 # 批量评测脚本
│   ├── run_spatial_eval.sh            # spatial 单独评测脚本
│   ├── migrate_checkpoint.py          # checkpoint 格式迁移（a2c2 → lerobot 0.4.0）
│   └── rtc_local/monkey_patch.py      # RTC monkey-patch
├── outputs/
│   ├── smolvla_scratch_libero_*_v2/   # v2 训练输出（checkpoint）
│   ├── eval_full/                     # spatial + v1 评测结果
│   ├── eval_v2/                       # v2 评测结果
│   └── train_log_scratch_*_v2.log     # 训练日志
└── docs/
    ├── experiment_report_libero_spatial.md  # 实验报告
    └── training_eval_guide.md               # 本文档
```

---

## 2. 训练

### 2.1 配置文件说明

配置文件位于 `configs/train_scratch_libero_{suite}.json`，关键字段：

```json
{
  "output_dir": "/root/project/lq/lerobot/outputs/smolvla_scratch_libero_object_v2",
  "job_name": "smolvla_scratch_libero_object_v2",
  "batch_size": 64,
  "steps": 100000,
  "eval_freq": 20000,
  "save_freq": 20000,
  "policy": {
    "type": "smolvla",
    "n_obs_steps": 1,
    "normalization_mapping": {
      "VISUAL": "IDENTITY",
      "STATE": "MEAN_STD",
      "ACTION": "MEAN_STD"
    }
  },
  "dataset": {
    "root": "/root/storage/DATA/libero_lerobot/libero_object_no_noops"
  }
}
```

**修改配置的常见需求：**

| 需求 | 修改字段 | 示例 |
|---|---|---|
| 修改 batch size | `batch_size` | `"batch_size": 128` |
| 修改训练步数 | `steps` | `"steps": 200000` |
| 修改保存频率 | `save_freq` | `"save_freq": 10000` |
| 修改输出目录 | `output_dir` + `job_name` | 确保两者一致 |
| 切换数据集 | `dataset.root` | 指向对应 suite 的数据路径 |

**⚠️ 关键：`normalization_mapping` 必须存在**，否则训练出的模型无法正确推理。

### 2.2 单卡训练（当前方式）

```bash
cd /root/project/lq/lerobot

# 用法：bash scripts/train_libero_scratch_v2.sh [suite] [steps] [batch_size] [gpu_id]
# 后台运行并记录日志：
nohup bash scripts/train_libero_scratch_v2.sh libero_object 100000 64 2 \
  > outputs/train_log_scratch_object_v3.log 2>&1 &
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| suite | libero_spatial | 可选：libero_spatial, libero_object, libero_goal, libero_10 |
| steps | 100000 | 训练总步数 |
| batch_size | 64 | 单卡 A100-80GB 可用 64，更大需多卡 |
| gpu_id | 4 | CUDA_VISIBLE_DEVICES |

**同时启动多个 suite（各占一张卡）：**

```bash
cd /root/project/lq/lerobot

nohup bash scripts/train_libero_scratch_v2.sh libero_object 100000 64 2 \
  > outputs/train_log_scratch_object_v3.log 2>&1 &

nohup bash scripts/train_libero_scratch_v2.sh libero_goal 100000 64 5 \
  > outputs/train_log_scratch_goal_v3.log 2>&1 &

nohup bash scripts/train_libero_scratch_v2.sh libero_10 100000 64 6 \
  > outputs/train_log_scratch_10_v3.log 2>&1 &
```

### 2.3 多卡训练（扩大 batch size）

当前 a2c2-libero 的 `train.py` **不原生支持 DDP/多卡**（代码中无 `DistributedDataParallel` 逻辑）。要用多卡扩大 batch size 有以下方案：

**方案 A：单卡尽量拉大 batch size + gradient accumulation（推荐）**

在配置文件中修改 `batch_size`，在单卡 A100-80GB 上测试最大能跑的 batch size（预估可达 96-128，取决于 chunk_size 和序列长度）：

```bash
# 先小规模测试是否 OOM
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/root/project/lq/a2c2-libero/src \
  /root/miniconda/envs/lerobot/bin/python -m lerobot.scripts.train \
  --config_path=configs/train_scratch_libero_object.json \
  --batch_size=128
# 如果不 OOM，改配置文件的 batch_size 后正式训练
```

**方案 B：torchrun 多卡（需要修改训练脚本）**

`torchrun` 在 lerobot 环境中可用，但 a2c2 的 `train.py` 未实现 DDP 初始化。如需多卡，需修改 `a2c2-libero/src/lerobot/scripts/train.py`，添加以下逻辑：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 包装模型
model = DDP(model, device_ids=[local_rank])

# 使用 DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
```

启动方式：

```bash
# 4卡训练，每卡 bs=64，有效 bs=256
PYTHONPATH=/root/project/lq/a2c2-libero/src \
  /root/miniconda/envs/lerobot/bin/torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  -m lerobot.scripts.train \
  --config_path=configs/train_scratch_libero_object.json
```

> **注意**：方案 B 需要开发工作，当前代码无法直接使用。

### 2.4 训练自动 resume

训练脚本支持自动 resume：如果 `output_dir/checkpoints/` 中有已有 checkpoint，会自动从最新 checkpoint 恢复训练。无需手动干预。

---

## 3. Checkpoint 迁移

a2c2 训练产出的 checkpoint 格式与 lerobot 0.4.0 评测脚本不兼容，**评测前必须迁移**。

```bash
cd /root/project/lq/lerobot

# 迁移单个 checkpoint
/root/miniconda/envs/lerobot/bin/python scripts/migrate_checkpoint.py \
  outputs/smolvla_scratch_libero_object_v2/checkpoints/100000/pretrained_model

# 输出：outputs/.../pretrained_model_migrated/
```

**迁移做了什么：**
1. 从 `model.safetensors` 提取归一化参数 → `policy_preprocessor*.safetensors` + `policy_postprocessor*.safetensors`
2. 修复 `config.json` 中图像 shape：HWC `[256,256,3]` → CHW `[3,256,256]`

**批量迁移：**

```bash
for suite in object goal 10; do
  /root/miniconda/envs/lerobot/bin/python scripts/migrate_checkpoint.py \
    outputs/smolvla_scratch_libero_${suite}_v2/checkpoints/100000/pretrained_model
done
```

---

## 4. 评测

### 4.1 单个 suite 快速评测

```bash
cd /root/project/lq/lerobot

# Naive（朴素 chunk reuse），单组参数
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl \
  /root/miniconda/envs/lerobot/bin/python scripts/eval_libero_rtc.py \
  --checkpoint outputs/smolvla_scratch_libero_object_v2/checkpoints/100000/pretrained_model_migrated \
  --norm_checkpoint outputs/smolvla_scratch_libero_object_v2/checkpoints/100000/pretrained_model/model.safetensors \
  --suite libero_object \
  --horizon 10 --delay 0 \
  --trials 10 \
  --out_dir outputs/eval_test
```

### 4.2 14 组参数消融（Naive + RTC）

```bash
# Naive 14 组消融
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl \
  /root/miniconda/envs/lerobot/bin/python scripts/eval_libero_rtc.py \
  --checkpoint <CKPT_MIGRATED_PATH> \
  --norm_checkpoint <ORIGINAL_MODEL_SAFETENSORS> \
  --suite libero_object \
  --mode rtc_ablation --trials 10 --no_video --resume \
  --out_dir outputs/eval_v3/naive

# RTC 14 组消融（加 --use_rtc）
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl \
  /root/miniconda/envs/lerobot/bin/python scripts/eval_libero_rtc.py \
  --checkpoint <CKPT_MIGRATED_PATH> \
  --norm_checkpoint <ORIGINAL_MODEL_SAFETENSORS> \
  --suite libero_object \
  --mode rtc_ablation --use_rtc --trials 10 --no_video --resume \
  --out_dir outputs/eval_v3/rtc
```

**评测参数说明：**

| 参数 | 说明 |
|---|---|
| `--checkpoint` | 迁移后的 checkpoint 路径（`pretrained_model_migrated`） |
| `--norm_checkpoint` | 原始 checkpoint 的 `model.safetensors`（含归一化参数） |
| `--suite` | libero_spatial / libero_object / libero_goal / libero_10 |
| `--mode rtc_ablation` | 跑 14 组 (horizon, delay) 参数组合 |
| `--use_rtc` | 启用官方 RTC guidance（不加则为 Naive） |
| `--trials` | 每个 task 的 rollout 次数（论文用 50，我们用 10） |
| `--no_video` | 不保存视频，加速评测 |
| `--resume` | 跳过已有结果，断点续跑 |
| `--out_dir` | 结果输出目录 |

### 4.3 多 suite 并行评测

参考 `scripts/run_eval_v2.sh`，核心逻辑是每个 suite 分配一张 GPU 后台运行：

```bash
cd /root/project/lq/lerobot

# 后台启动全量评测
nohup bash scripts/run_eval_v2.sh > outputs/eval_v2/logs/all_eval.log 2>&1 &

# 或手动并行启动
eval_one() {
  CUDA_VISIBLE_DEVICES=$2 MUJOCO_GL=egl \
    /root/miniconda/envs/lerobot/bin/python scripts/eval_libero_rtc.py \
    --checkpoint $3 --norm_checkpoint $4 \
    --suite $1 --mode rtc_ablation --trials 10 --no_video --resume \
    --out_dir outputs/eval_v3/naive \
    2>&1 | tee outputs/eval_v3/logs/log_${1}_naive.txt
}

eval_one libero_object 4 <CKPT> <NORM> &
eval_one libero_goal   5 <CKPT> <NORM> &
eval_one libero_10     6 <CKPT> <NORM> &
wait
```

### 4.4 评测耗时预估

| Suite | 每 episode | 每组 (10×10) | 14 组 Naive + 14 组 RTC |
|---|---|---|---|
| libero_spatial | ~105s | ~2.9h | ~81h (~3.4 天) |
| libero_object | ~105s | ~2.9h | ~81h |
| libero_goal | ~113s | ~3.1h | ~87h |
| libero_10 | ~195s | ~5.4h | ~151h (~6.3 天) |

---

## 5. 结果文件说明

### 5.1 评测结果路径

| 内容 | 路径 |
|---|---|
| Spatial (v1, a2c2 80K) Naive | `outputs/eval_full/naive/results_libero_spatial.jsonl` |
| Spatial (v1, a2c2 80K) RTC | `outputs/eval_full/rtc/results_libero_spatial.jsonl` |
| Object/Goal/10 (v1, 无 norm) | `outputs/eval_full/{naive,rtc}/results_libero_{object,goal,10}.jsonl` |
| Object/Goal/10 (v2, 修复) Naive | `outputs/eval_v2/naive/results_libero_{object,goal,10}.jsonl` |
| Object/Goal/10 (v2, 修复) RTC | `outputs/eval_v2/rtc/results_libero_{object,goal,10}.jsonl` |
| Spatial 带视频 | `outputs/eval_with_video/naive/results_libero_spatial.jsonl` |

### 5.2 结果文件格式

每个 `.jsonl` 文件每行一个 JSON，格式：

```json
{
  "suite": "libero_object",
  "execute_horizon": 10,
  "inference_delay": 0,
  "method": "chunk_reuse",
  "success_rate": 0.01,
  "num_trials_per_task": 10,
  "per_task_success": [0.0, 0.1, 0.0, ...]
}
```

### 5.3 快速查看结果

```bash
# 查看某个 suite 的所有结果
python3 -c "
import json
with open('outputs/eval_v2/naive/results_libero_goal.jsonl') as f:
    for line in f:
        d = json.loads(line)
        print(f'h={d[\"execute_horizon\"]:>2}, d={d[\"inference_delay\"]:>2}  SR={d[\"success_rate\"]:.1%}')
"

# 一键查看所有 suite
for s in libero_spatial libero_object libero_goal libero_10; do
  echo "=== ${s} ===";
  for mode in naive rtc; do
    f="outputs/eval_v2/${mode}/results_${s}.jsonl"
    [ -f "$f" ] && echo "  ${mode}: $(wc -l < $f) 组" || echo "  ${mode}: 无"
  done
done
```

### 5.4 训练 Checkpoint 路径

```
outputs/smolvla_scratch_libero_{suite}_v2/
└── checkpoints/
    ├── 020000/pretrained_model/          # a2c2 格式（训练原始输出）
    ├── 020000/pretrained_model_migrated/ # lerobot 0.4.0 格式（迁移后，用于评测）
    ├── 040000/...
    ├── 060000/...
    ├── 080000/...
    ├── 100000/...
    └── last -> 100000                    # 软链接到最新
```

---

## 6. 监控指南

### 6.1 监控训练进度

```bash
# 查看最新训练步数和 loss
for suite in object goal 10; do
  echo "=== libero_${suite} ==="
  grep "step:" outputs/train_log_scratch_${suite}_v2.log | tail -1
  echo ""
done
```

**日志格式说明：**
```
INFO step:73K smpl:5M ep:32K epch:69.56 loss:0.032 grdn:0.195 lr:2.5e-06 updt_s:0.847 data_s:0.002
```

| 字段 | 含义 |
|---|---|
| step | 当前训练步数 |
| smpl | 已处理样本数 |
| ep | 已处理 episode 数 |
| epch | 已训练 epoch 数 |
| loss | 当前 loss |
| grdn | 梯度范数 |
| lr | 学习率 |
| updt_s | 每步更新耗时（秒） |
| data_s | 数据加载耗时（秒） |

### 6.2 监控 GPU 状态

```bash
# 查看所有 GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 仅查看特定 GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader --id=2,5,6
```

### 6.3 监控评测进度

```bash
# 方法 1：查看结果文件行数（每完成一组参数写入一行）
for s in libero_object libero_goal libero_10; do
  echo -n "${s}: "
  f="outputs/eval_v2/naive/results_${s}.jsonl"
  [ -f "$f" ] && echo "$(wc -l < $f)/14 组" || echo "尚未开始"
done

# 方法 2：查看日志（注意 tee 有缓冲，用 cat -v 处理 \r）
tail -20 outputs/eval_v2/logs/log_libero_object_naive.txt | cat -v | tail -10

# 方法 3：检查进程是否在运行
ps aux | grep eval_libero_rtc | grep -v grep
```

### 6.4 预估训练剩余时间

```bash
# 从日志提取当前步数和速度，计算剩余时间
python3 -c "
import re, sys
log = 'outputs/train_log_scratch_object_v2.log'
target = 100000
with open(log) as f:
    lines = [l for l in f if 'step:' in l]
if lines:
    last = lines[-1]
    step = int(re.search(r'step:(\d+)K', last).group(1)) * 1000
    spd = float(re.search(r'updt_s:([\d.]+)', last).group(1))
    remain = (target - step) * spd / 3600
    print(f'当前: {step}/{target}, 速度: {spd}s/step, 预估剩余: {remain:.1f}h')
"
```

### 6.5 一键状态面板

```bash
# 复制此命令到终端，一次性查看所有状态
echo "====== 训练 ======" && \
for suite in object goal 10 spatial; do
  log="outputs/train_log_scratch_${suite}_v2.log"
  [ -f "$log" ] && echo "=== ${suite}: $(grep 'step:' $log | tail -1 | grep -oP 'step:\S+')" || true
done && \
echo "" && echo "====== 评测 ======" && \
for s in libero_object libero_goal libero_10 libero_spatial; do
  for mode in naive rtc; do
    f="outputs/eval_v2/${mode}/results_${s}.jsonl"
    [ -f "$f" ] && echo "${s} ${mode}: $(wc -l < $f)/14 组"
  done
done && \
echo "" && echo "====== GPU ======" && \
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 7. 完整工作流（从训练到评测）

```bash
cd /root/project/lq/lerobot

# ① 编辑配置（如需修改 batch_size、steps 等）
vim configs/train_scratch_libero_object.json

# ② 启动训练
nohup bash scripts/train_libero_scratch_v2.sh libero_object 100000 64 2 \
  > outputs/train_log_scratch_object_v3.log 2>&1 &

# ③ 监控训练
grep "step:" outputs/train_log_scratch_object_v3.log | tail -1

# ④ 训练完成后迁移 checkpoint
/root/miniconda/envs/lerobot/bin/python scripts/migrate_checkpoint.py \
  outputs/smolvla_scratch_libero_object_v3/checkpoints/100000/pretrained_model

# ⑤ 启动评测
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl \
  /root/miniconda/envs/lerobot/bin/python scripts/eval_libero_rtc.py \
  --checkpoint outputs/smolvla_scratch_libero_object_v3/checkpoints/100000/pretrained_model_migrated \
  --norm_checkpoint outputs/smolvla_scratch_libero_object_v3/checkpoints/100000/pretrained_model/model.safetensors \
  --suite libero_object \
  --mode rtc_ablation --trials 10 --no_video --resume \
  --out_dir outputs/eval_v3/naive

# ⑥ 查看结果
python3 -c "
import json
with open('outputs/eval_v3/naive/results_libero_object.jsonl') as f:
    for line in f:
        d = json.loads(line)
        print(f'h={d[\"execute_horizon\"]:>2}, d={d[\"inference_delay\"]:>2}  SR={d[\"success_rate\"]:.1%}')
"
```

---

## 8. 数据集路径

| Suite | 数据集路径 |
|---|---|
| libero_spatial | `/root/storage/DATA/libero_lerobot/libero_spatial_no_noops` |
| libero_object | `/root/storage/DATA/libero_lerobot/libero_object_no_noops` |
| libero_goal | `/root/storage/DATA/libero_lerobot/libero_goal_no_noops` |
| libero_10 | `/root/storage/DATA/libero_lerobot/libero_10_no_noops` |
| libero_90 | `/root/storage/DATA/libero_lerobot/libero_90_no_noops` |

---

## 9. 常见问题

### Q: 训练 loss 不降 / 评测 0%

- 检查 `normalization_mapping` 是否存在于配置文件中
- 检查数据集路径是否正确
- 单套件从零训练在 object/10 上极难收敛（社区验证），建议 finetune from smolvla_base

### Q: 评测时报 "Unexpected key(s) when loading model"

- 正常警告，不影响推理。迁移脚本未移除 `normalize_targets` 相关 key

### Q: 评测日志看起来卡住了

- `tee` 有缓冲延迟，进度条用 `\r` 覆盖显示。用 `cat -v` 查看或直接检查结果文件行数

### Q: 如何增大 batch size

- 单卡 A100-80GB 当前用 bs=64（~29GB 显存），可尝试 bs=96 或 bs=128
- 更大需多卡 DDP（当前训练脚本不支持，需修改代码）
- 替代方案：gradient accumulation（需修改训练代码添加 `accumulate_grad_batches` 参数）

### Q: 如何评测中间 checkpoint（如 60K 步）

```bash
# 先迁移该 checkpoint
/root/miniconda/envs/lerobot/bin/python scripts/migrate_checkpoint.py \
  outputs/smolvla_scratch_libero_object_v2/checkpoints/060000/pretrained_model

# 再用迁移后的路径评测
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl \
  /root/miniconda/envs/lerobot/bin/python scripts/eval_libero_rtc.py \
  --checkpoint outputs/smolvla_scratch_libero_object_v2/checkpoints/060000/pretrained_model_migrated \
  --norm_checkpoint outputs/smolvla_scratch_libero_object_v2/checkpoints/060000/pretrained_model/model.safetensors \
  --suite libero_object --horizon 10 --delay 0 --trials 10 \
  --out_dir outputs/eval_60k
```
