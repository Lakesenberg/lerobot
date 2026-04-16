# 两套方案对比：a2c2-libero vs 官方 RTC

## 整体架构定位

```
SmolVLA（基础模型，两套方案共用同一模型结构）
         │
         ├─── a2c2-libero 方案
         │      ├─ 朴素 chunk 重用（模拟 RTC 的效果）
         │      └─ Residual Transformer（额外训练一个残差修正网络）
         │
         └─── 官方 lerobot 方案
                └─ RTCProcessor（flow matching 引导，推理时加引导梯度）
```

---

## 核心技术对比

### 动作 chunk 执行逻辑

**a2c2-libero（`eval_libero/evaluation_libero.py` 第 259-262 行）**

```python
# 朴素拼接：用旧 chunk 的前 inference_delay 步 + 新 chunk 的后 execute_horizon 步
exec_entries = [pending_actions.popleft() for _ in range(inference_delay)]
exec_entries.extend(chunk_entries[inference_delay:execute_horizon])
action_plan = collections.deque(exec_entries)
```

- 简单数组拼接，新旧 chunk 在拼接点**可能发生数值跳变**
- 不对新 chunk 施加任何约束

**官方 RTC（`lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` 第 847-858 行）**

```python
# flow matching denoising 每一步：RTCProcessor 加引导梯度
if self._rtc_enabled():
    v_t = self.rtc_processor.denoise_step(
        x_t=x_t,
        v_t=v_t,
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
    )
```

- 在 denoising 每一步加 guidance term
- 强制新 chunk 的前 execution_horizon 步数值上平滑过渡
- 从物理机制上消除 chunk 边界跳变

### ActionQueue 管理

| 特性 | a2c2-libero | 官方 RTC |
|------|-------------|---------|
| 数据结构 | `collections.deque` | `ActionQueue`（线程安全） |
| 剩余动作追踪 | 手写 `pending_actions` | `action_queue.get_left_over()` |
| 真机异步支持 | 不支持 | 支持（线程安全设计） |
| 引导输入 | 不传给模型 | 作为 `prev_chunk_left_over` 传入 denoise |

---

## 文件对应关系

| 功能 | a2c2-libero | 官方 lerobot |
|------|-------------|-------------|
| 代码库 | `/root/project/lq/a2c2-libero/` | `/root/project/lq/lerobot/` |
| 环境 | `.venv`（lerobot 0.2.0 fork） | conda `lerobot`（lerobot 0.4.0） |
| 训练脚本 | `src/lerobot/scripts/train.py` | `scripts/train_libero_smolvla.sh` |
| 评测脚本 | `eval_libero/run_eval.py` | `scripts/eval_libero_rtc.py` |
| RTC 实现 | 手写（无引导） | `policies/rtc/`（flow matching 引导） |
| 残差网络 | `policies/residual_transformer/` | 无 |
| 数据集格式 | v2.1 | v3.0（需先转换） |
| 结果目录 | `result/libero_eval/` | `outputs/eval_libero_rtc/` |

---

## 已有结果（a2c2-libero，libero_spatial，80k steps）

| execute_horizon | inference_delay | 成功率 |
|:-:|:-:|:-:|
| 5 | 0 | **68%** ← 最优 |
| 10 | 1 | 67% |
| 10 | 5 | 65% |
| 10 | 3 | 62% |
| 30 | 0 | 62% |
| 10 | 0 | 60% |
| 1 | 0 | 59% |
| 10 | 10 | 59% |
| 50 | 0 | 47% |
| 40 | 10 | 42% |

> 评测数据位于 `a2c2-libero/result/libero_eval/results_libero_spatial.jsonl`

---

## 运行指引

### Step 1：数据格式转换（仅首次）

```bash
conda activate lerobot
cd /root/project/lq/lerobot

python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=local/libero_spatial \
    --root=/root/storage/DATA/libero_lerobot \
    --push-to-hub=false
```

### Step 2：安装 libero

```bash
conda activate lerobot
pip install -e /root/project/lq/a2c2-libero/third_party/libero
```

### Step 3：训练官方 SmolVLA on LIBERO

```bash
conda activate lerobot
cd /root/project/lq/lerobot
bash scripts/train_libero_smolvla.sh libero_spatial 100000 64 0
```

### Step 4：用官方 RTC 评测

```bash
conda activate lerobot
cd /root/project/lq/lerobot
MUJOCO_GL=egl python scripts/eval_libero_rtc.py \
    --checkpoint outputs/lerobot_smolvla_libero_spatial/checkpoints/last/pretrained_model \
    --suite libero_spatial \
    --mode rtc_ablation \
    --trials 10
```

### Step 5：对比两套结果

```bash
# a2c2 已有结果
cat /root/project/lq/a2c2-libero/result/libero_eval/results_libero_spatial.jsonl

# 官方 RTC 新结果
cat /root/project/lq/lerobot/outputs/eval_libero_rtc/results_libero_spatial.jsonl
```

---

## 论文章节建议

| 章节 | 内容 | 数据来源 |
|------|------|---------|
| Baseline | SmolVLA 全 chunk 执行（horizon=50, delay=0） | a2c2 或 lerobot 均可 |
| Method A | 朴素 chunk 重用（本文 a2c2 方案） | `a2c2-libero/result/` |
| Method B | 官方 RTC（flow matching 引导） | `lerobot/outputs/eval_libero_rtc/` |
| Method C（可选） | Method A + Residual Transformer | 需先训练 residual |
| 消融 | execute_horizon / inference_delay 对各方法的影响 | 各 14 组参数结果 |
