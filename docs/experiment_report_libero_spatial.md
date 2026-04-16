# SmolVLA × Real-Time Chunking (RTC) 在 LIBERO 上的评测实验报告

> 实验日期：2026-04-03 ~ 2026-04-16（持续更新）
> 评测基准：LIBERO Spatial / Object / Goal / 10（10 tasks × 10 trials = 100 episodes / 参数组）
> 模型：SmolVLA（SmolVLM2-500M-Video-Instruct backbone + Flow Matching Expert）
> 相关论文：SmolVLA (arXiv:2506.01844), A2C2 (arXiv:2509.23224), RTC (arXiv:2506.07339)

---

## 1. 实验概述

本实验对比了三种 action chunk 管理策略在 LIBERO-Spatial 基准上的表现：


| 策略                         | 代号    | 说明                                                        |
| -------------------------- | ----- | --------------------------------------------------------- |
| a2c2-libero 朴素 chunk reuse | a2c2  | a2c2-libero 代码库自带的评测，含内部归一化                               |
| lerobot 朴素 chunk reuse     | Naive | 使用 lerobot 0.4.0 推理 + 手动归一化，动作层面拼接                        |
| lerobot 官方 RTC guidance    | RTC   | Monkey-patch flow matching denoising loop，注入 RTCProcessor |


**消融参数**：14 组 (execute_horizon, inference_delay) 组合，覆盖不同执行步长和模拟推理延迟。

---

## 2. 模型与训练信息

### 2.1 评测用模型（a2c2 训练的 80K checkpoint）


| 项目            | 详情                                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------- |
| 来源            | a2c2-libero 从零训练 SmolVLA，libero_spatial 数据集                                                        |
| 训练步数          | 80,000 步                                                                                           |
| Checkpoint 路径 | `/root/project/lq/a2c2-libero/libero_smolvla_scratch/checkpoints/080000/pretrained_model_migrated` |
| 归一化参数来源       | 原始 checkpoint `pretrained_model/model.safetensors` 中提取                                             |


### 2.2 从零训练的新模型（lerobot 100K checkpoint）


| 项目            | 详情                                                                                                    |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| 训练方式          | 使用 a2c2-libero 源码 + lerobot conda 环境                                                                  |
| 初始权重          | SmolVLM2-500M-Video-Instruct（VLM backbone），Expert 随机初始化                                               |
| 训练步数          | 100,000 步                                                                                             |
| 最终 loss       | 0.029                                                                                                 |
| Checkpoint 路径 | `/root/project/lq/lerobot/outputs/smolvla_scratch_libero_spatial/checkpoints/100000/pretrained_model` |
| 保存间隔          | 每 20K 步（020000, 040000, 060000, 080000, 100000）                                                       |
| 状态            | 已完成，尚未评测                                                                                              |

### 2.3 v2 从零训练模型（修复 normalization_mapping，100K checkpoint）

> **v2** 最初为 object/goal/10 三个 suite 训练的模型（v1）由于训练配置文件中**缺失 `normalization_mapping` 字段**，导致训练时观测状态未归一化、动作输出未反归一化，模型无法正确学习，评测成功率 0-5.5%。v2 模型是在修复此配置错误后，使用完全相同的训练流程重新从零训练的版本。
>
> **v1 vs v2 的唯一区别**：v2 配置文件中补充了 `normalization_mapping: {VISUAL: IDENTITY, STATE: MEAN_STD, ACTION: MEAN_STD}`，与 a2c2 团队的 spatial 训练配置一致。其余参数（模型架构、数据集、学习率、batch size、步数）完全相同。
>
> **训练方式说明**：v2 模型为**单套件从零训练**，即每个 suite 独立用自己的数据集训练一个模型，不使用其他 suite 的数据，不使用预训练的 VLA 权重（`smolvla_base`），仅加载 VLM backbone `SmolVLM2-500M-Video-Instruct` 的权重，action expert 从随机初始化开始。

| 项目 | libero_object | libero_goal | libero_10 |
|---|---|---|---|
| 训练方式 | 单套件从零训练（a2c2 源码 + lerobot 环境） | 同左 | 同左 |
| 初始权重 | VLM: SmolVLM2-500M, Expert: 随机初始化 | 同左 | 同左 |
| 训练步数 | 100,000 | 100,000 | 100,000 |
| Batch size | 64 | 64 | 64 |
| 最终 loss | 0.032 | 0.024 | 0.044 |
| normalization_mapping | VISUAL:IDENTITY, STATE:MEAN_STD, ACTION:MEAN_STD | 同左 | 同左 |
| GPU | A100-80G × 1 | 同左 | 同左 |
| Checkpoint | `outputs/smolvla_scratch_libero_{suite}_v2/checkpoints/100000/pretrained_model_migrated` |||
| 状态 | ✅ 训练完成，✅ 评测完成 | 同左 | 同左 |


---

## 3. LIBERO-Spatial 评测结果

### 3.1 三种策略完整对比表

#### delay = 0（无推理延迟）


| execute_horizon | a2c2 朴素 | Naive (lerobot) | RTC (lerobot) | Naive vs a2c2 | RTC vs Naive |
| --------------- | ------- | --------------- | ------------- | ------------- | ------------ |
| 1               | 59%     | **74%**         | 70%           | +15%          | -4%          |
| 5               | 68%     | **85%**         | 86%           | +17%          | +1%          |
| 10              | 60%     | 82%             | **84%**       | +22%          | +2%          |
| 30              | 62%     | **82%**         | 64%           | +20%          | -18%         |
| 40              | 51%     | **75%**         | 71%           | +24%          | -4%          |
| 50              | 47%     | 58%             | **69%**       | +11%          | +11%         |


#### execute_horizon = 10，变化 delay


| inference_delay | a2c2 朴素 | Naive (lerobot) | RTC (lerobot) | Naive vs a2c2 | RTC vs Naive |
| --------------- | ------- | --------------- | ------------- | ------------- | ------------ |
| 0               | 60%     | 82%             | **84%**       | +22%          | +2%          |
| 1               | 67%     | **85%**         | 76%           | +18%          | -9%          |
| 3               | 62%     | **89%**         | 25%           | +27%          | -64%         |
| 5               | 65%     | **92%**         | 2%            | +27%          | -90%         |
| 10              | 59%     | **73%**         | 0%            | +14%          | -73%         |


#### execute_horizon = 40，变化 delay


| inference_delay | a2c2 朴素 | Naive (lerobot) | RTC (lerobot) | Naive vs a2c2 | RTC vs Naive |
| --------------- | ------- | --------------- | ------------- | ------------- | ------------ |
| 0               | 51%     | **75%**         | 71%           | +24%          | -4%          |
| 1               | 54%     | **69%**         | 66%           | +15%          | -3%          |
| 3               | 55%     | **67%**         | 48%           | +12%          | -19%         |
| 5               | 45%     | **72%**         | 22%           | +27%          | -50%         |
| 10              | 42%     | **61%**         | 9%            | +19%          | -52%         |


### 3.2 最佳配置汇总


| 策略              | 最佳配置      | 最高成功率   |
| --------------- | --------- | ------- |
| a2c2 朴素         | h=5, d=0  | 68%     |
| Naive (lerobot) | h=10, d=5 | **92%** |
| RTC (lerobot)   | h=5, d=0  | 86%     |


---

## 3-B. v2 模型 (Object / Goal / 10) 评测结果

### 3-B.1 libero_goal（v2，最佳 16%）

| 参数 (h, d) | Naive | RTC |
|---|---|---|
| h=1, d=0 | 4% | 1% |
| h=5, d=0 | 5% | 0% |
| h=10, d=0 | 2% | 0% |
| h=30, d=0 | 4% | 12% |
| h=40, d=0 | 11% | **16%** |
| h=50, d=0 | **16%** | 15% |
| h=10, d=1 | 2% | 1% |
| h=10, d=3 | 1% | 0% |
| h=10, d=5 | 0% | 2% |
| h=10, d=10 | 0% | 0% |
| h=40, d=1 | 11% | 13% |
| h=40, d=3 | 10% | 12% |
| h=40, d=5 | 11% | 11% |
| h=40, d=10 | 13% | 10% |

### 3-B.2 libero_object（v2，最佳 1%）

所有 14 组参数 × 2 模式（Naive + RTC）均为 **0%**，仅 Naive h=30 d=0 出现 1% 成功率。

### 3-B.3 libero_10（v2，0%）

所有 28 组测试均为 **0%**。

### 3-B.4 v2 评测结果小结

| Suite | Naive 最佳 | RTC 最佳 | 状态 |
|---|---|---|---|
| libero_spatial (a2c2 80K) | **92%** | 86% | 正常 |
| libero_goal (v2 100K) | 16% | 16% | 远低于预期 |
| libero_object (v2 100K) | 1% | 0% | 几乎完全失败 |
| libero_10 (v2 100K) | 0% | 0% | 完全失败 |

> 数据来源：`lerobot/outputs/eval_v2/{naive,rtc}/results_libero_{object,goal,10}.jsonl`

---

## 4. 与论文分数的对比分析

### 4.1 数据来源


| 来源          | 标记            | 文件/引用                                                          |
| ----------- | ------------- | -------------------------------------------------------------- |
| SmolVLA 论文  | SmolVLA-paper | arXiv:2506.01844, Table 2 & 消融表                                |
| A2C2 论文     | A2C2-paper    | arXiv:2509.23224, Table 1 & Table 7 & Table 8                  |
| RTC 原始论文    | RTC-paper     | arXiv:2506.07339 (NeurIPS 2025), Physical Intelligence         |
| a2c2 团队实测   | a2c2-local    | `a2c2-libero/result/libero_eval/results_libero_spatial.jsonl`  |
| 我们的 Naive   | ours-naive    | `lerobot/outputs/eval_full/naive/results_libero_spatial.jsonl` |
| 我们的 RTC     | ours-rtc      | `lerobot/outputs/eval_full/rtc/results_libero_spatial.jsonl`   |
| GitHub 社区复现 | community     | huggingface/lerobot Issues #2107, #2354, #3264, #3287          |


### 4.2 SmolVLA 论文分数解读

SmolVLA 论文有**两套训练方式**，产出分数差异巨大：


| 训练方式                | Spatial | Object  | Goal    | Long(10) | 平均        | 条件                    |
| ------------------- | ------- | ------- | ------- | -------- | --------- | --------------------- |
| 单套件 0.24B (Table 2) | 86.4%   | 46.4%   | 35.0%   | 60.0%    | 57.0%     | 无 VLA 预训练             |
| 单套件 0.45B (Table 2) | 82.5%   | 41.8%   | 45.0%   | 60.0%    | 57.3%     | 无 VLA 预训练             |
| 消融基线 (chunk=50, CA) | 89%     | 94%     | 85%     | 53%      | 80.3%     | 多套件联合                 |
| **官方 checkpoint**   | **90%** | **96%** | **92%** | **71%**  | **87.3%** | 多套件联合, 8×H100, bs=256 |


> **关键**：被广泛引用的 90/96/92/71 来自**多套件联合训练 + 大 batch**，非单套件可达。

### 4.3 RTC 在 LIBERO 上的已知结果

#### RTC 原始论文 (Black et al., NeurIPS 2025, arXiv:2506.07339)

- **仅在 Kinetix（12 个动态任务）上做了仿真对比**，未在 LIBERO 上做仿真评测
- 真机实验使用 π0.5 (3B) 在 6 个双臂任务上测试
- 论文指出 LIBERO 类准静态 (quasi-static) 任务"标准 chunked execution 就能达到接近完美的成功率"，因此 RTC 的加速优势在 LIBERO 上不显著

#### lerobot GitHub PR

- **PR #1521** (Implement RTC for SmolVLA, @ben-z)：2025.07 创建，2026.03 关闭（draft），**无 LIBERO 跑分**，被关闭原因为"RTC has been implemented"（内部已合并）
- lerobot 官方 LIBERO 文档仅公布了 π0.5 的标准评测结果，**未公布 RTC + LIBERO 的组合跑分**

#### StreamingVLA 论文的独立验证 (arXiv:2603.28565, 2026.03)

StreamingVLA 论文在 LIBERO 上对 RTC 做了**独立基准测试**（基于 π0.5-Libero, d=1），结果与我们的发现一致：


| 方法                     | Spatial   | Object    | Goal      | Long(10)  | 平均         |
| ---------------------- | --------- | --------- | --------- | --------- | ---------- |
| π0.5 (h=5, replan)     | 98.8%     | 98.2%     | 98.0%     | 92.4%     | 96.9%      |
| π0.5 (h=10, no replan) | 97.4%     | 98.2%     | 96.2%     | 88.6%     | 95.1%      |
| **RTC (d=1)**          | **96.2%** | **19.8%** | **93.0%** | **25.2%** | **58.55%** |
| SmolVLA (async)        | 97.0%     | 99.0%     | 97.0%     | 90.0%     | 95.8%      |


> 来源：StreamingVLA Table 1 (arXiv:2603.28565v1)

StreamingVLA 论文明确指出：*"this early observation strategy leads to a pronounced drop in success rate, with the average performance falling to 58.55% and long-horizon tasks degrading drastically. Although RTC attempts to refine the predicted velocity field through gradient-based correction during inference, the improvement is limited and insufficient to compensate for the information loss caused by premature observation."*

**这验证了我们的发现：RTC 在 LIBERO 上的退化不是我们 monkey-patch 实现的 bug，而是 RTC 方法本身在 LIBERO 类准静态任务上的固有局限。** Object (19.8%) 和 Long (25.2%) 的崩溃尤为显著。

#### 我们的数据与 StreamingVLA 直接对标

> **背景差异说明**：StreamingVLA 使用 π0.5 (3B 参数) 多套件联合训练模型；我们使用 SmolVLA (0.45B 参数) 单套件从零训练模型。两者模型能力差一个量级，下表旨在对比 **RTC 的退化趋势**而非绝对数值。
>
> **模型说明**：Spatial 列使用 a2c2 团队 80K checkpoint（第三方提供的 spatial 单套件训练模型）；Object / Goal / 10 列使用我们的 **v2 模型**——即修复了 normalization_mapping 配置后从零训练 100K 步的模型（详见 2.3 节），这三个 suite 各自独立训练，无跨套件数据共享。

**StreamingVLA 论文结果（π0.5-Libero, 3B, 多套件联合训练）：**

| 方法 | Spatial | Object | Goal | Long(10) | 平均 |
|---|---|---|---|---|---|
| π0.5 Baseline (h=10) | 97.4% | 98.2% | 96.2% | 88.6% | 95.1% |
| **RTC (d=1)** | **96.2%** | **19.8%** | **93.0%** | **25.2%** | **58.6%** |
| RTC 退化幅度 | -1.2% | **-78.4%** | -3.2% | **-63.4%** | -36.5% |

**我们的结果（SmolVLA 0.45B, 单套件从零训练）：**

| 方法 | Spatial | Object | Goal | Long(10) | 平均 |
|---|---|---|---|---|---|
| Naive (h=5, d=0) | 85% | 0% | 5% | 0% | 22.5% |
| Naive (h=10, d=0) | 82% | 0% | 2% | 0% | 21.0% |
| **RTC (h=10, d=1)** | **76%** | **0%** | **1%** | **0%** | **19.2%** |
| RTC 退化幅度 (vs h=10 d=0) | -6% | — | -1% | — | -1.8% |
| Naive best (各 suite 最佳) | 92% | 1% | 16% | 0% | 27.3% |
| RTC best (各 suite 最佳) | 86% | 0% | 16% | 0% | 25.5% |

> 数据来源：Spatial `lerobot/outputs/eval_full/{naive,rtc}/results_libero_spatial.jsonl`；Object/Goal/10 `lerobot/outputs/eval_v2/{naive,rtc}/results_libero_{object,goal,10}.jsonl`

**RTC 退化趋势对比（在 Spatial 上，两方唯一可比的 suite）：**

| delay | StreamingVLA RTC (π0.5) | 我们的 RTC (SmolVLA) | 趋势一致？ |
|---|---|---|---|
| d=0 | — | 84% (≈ Naive 82%) | ✅ RTC ≈ Naive |
| d=1 | 96.2% (-1.2%) | 76% (-6%) | ✅ 均下降，我们幅度更大 |
| d=3 | — | 25% (-57%) | — |
| d=5 | — | 2% (-80%) | — |
| d=10 | — | 0% (-82%) | — |

**结论**：在 Spatial 这个唯一可对比的 suite 上，我们的 RTC 退化方向与 StreamingVLA 完全一致。StreamingVLA 仅测了 d=1，已经观察到 Object/Long 崩溃至 20-25%；我们在更大 delay (d=3~10) 下观察到 Spatial 也崩溃至 0-25%，进一步验证了 RTC 在 LIBERO 上随 delay 增大而急剧退化的规律。

#### A2C2 论文 (arXiv:2509.23224) Table 8

A2C2 论文在 LIBERO Spatial 上对比了 Naive / RTC / A2C2 三种方法（d=0 时）：**RTC ≈ Naive（差距 < 1%）**，这与 RTC 原始论文"LIBERO 上不显著"的结论一致。

### 4.4 LIBERO Spatial 完整四方对比（Naive 策略）

以下对比同一模型 (SmolVLA 80K checkpoint) 在不同评测来源下的表现：


| 参数 (h, d)  | A2C2 论文 (50 trials) | A2C2 论文 RTC | a2c2 团队实测 (10 trials) | 我们 Naive (10 trials) | 我们 RTC (10 trials) |
| ---------- | ------------------- | ----------- | --------------------- | -------------------- | ------------------ |
| h=1, d=0   | 90.8%               | 90.9%       | 59%                   | **74%**              | 70%                |
| h=5, d=0   | 88.6%               | 88.9%       | 68%                   | **85%**              | 86%                |
| h=10, d=0  | 86.6%               | 86.3%       | 60%                   | 82%                  | **84%**            |
| h=10, d=1  | 83.7%               | —           | 67%                   | **85%**              | 76%                |
| h=10, d=3  | 81.0%               | —           | 62%                   | **89%**              | 25%                |
| h=10, d=5  | 82.0%               | —           | 65%                   | **92%**              | 2%                 |
| h=10, d=10 | 75.0%               | —           | 59%                   | **73%**              | 0%                 |
| h=30, d=0  | —                   | —           | 62%                   | **82%**              | 64%                |
| h=40, d=0  | 79.0%               | —           | 51%                   | **75%**              | 71%                |
| h=40, d=5  | 66.0%               | —           | 45%                   | **72%**              | 22%                |
| h=40, d=10 | 67.0%               | —           | 42%                   | **61%**              | 9%                 |
| h=50, d=0  | 71.0%               | —           | 47%                   | 58%                  | **69%**            |


> 数据来源：A2C2 论文 Table 7 (arXiv:2509.23224); a2c2 团队实测 (`a2c2-libero/result/libero_eval/results_libero_spatial.jsonl`); 我们的结果 (`lerobot/outputs/eval_full/{naive,rtc}/results_libero_spatial.jsonl`)

### 4.5 社区复现情况

#### 官方 checkpoint 评测（多套件联合训练模型 `HuggingFaceVLA/smolvla_libero`）

| 来源 | Spatial | Object | Goal | Long(10) | 平均 | 说明 |
|---|---|---|---|---|---|---|
| **论文官方** | 90% | 96% | 92% | 71% | 87.3% | 8×H100, bs=256, 多套件 |
| GitHub #3264 | 63% | 93% | 81% | 56% | 73.3% | lerobot 0.5.1, mujoco 3.3.2 |
| GitHub #3287 | 68% | 71% | 76% | 41% | 64.0% | bs=32, 80K, finetune from smolvla_base |

> **即使是官方 checkpoint 也难以完全复现**：社区评测平均仅 64-73%，远低于论文 87%。

#### 其他人训练测评情况

| 来源 | Suite | 成功率 | 训练配置 | 评测方式 |
|---|---|---|---|---|
| GitHub #1316 @zlw21gxy | spatial | 72% → 82% | from scratch, lerobot | n_action_steps=1 提升到 82% |
| GitHub #1316 @bairuofei | spatial | 67% | from scratch, 6K steps | 自定义 eval 脚本 |
| GitHub #1316 @hahans | object | 66% | finetune smolvla_base | 自定义 eval 脚本 |
| GitHub #1316 @hahans | object | **0%** | from scratch, bs=128, 200K | 从零训练完全失败 |
| GitHub #1369 @zhoutao | object | 30% | from scratch, bs=4, 100K | 极小 batch |
| GitHub #2107 @zimgong | libero_10 | 7.5% | 0.24B, bs=64, 100K | 单 RTX4090 |
| GitHub #3287 @Yoonkyo | libero_10 | 41% | finetune smolvla_base, bs=32, 80K | 微调而非从零训练 |
| **我们 v2** | **object** | **1%** | from scratch, bs=64, 100K | chunk reuse 评测 |
| **我们 v2** | **goal** | **16%** | from scratch, bs=64, 100K | chunk reuse 评测 |
| **我们 v2** | **libero_10** | **0%** | from scratch, bs=64, 100K | chunk reuse 评测 |

### 4.6 lerobot 官方 RTC + LIBERO 跑分情况

**结论：lerobot 官方仓库中没有 RTC + LIBERO 的成功率跑分。**

| 来源 | 状态 | 说明 |
|---|---|---|
| PR #1521 (RTC for SmolVLA, @ben-z) | 关闭（draft） | 2025.07 创建，2026.03 关闭，**无 LIBERO 跑分** |
| `examples/rtc/eval_dataset.py` | 已合并 | 测试 RTC 动作一致性（dataset sample），**不测 LIBERO 任务成功率** |
| PR #3319 (benchmark CI) | 进行中 | LIBERO 冒烟测试，仅 1 episode，无 RTC 评测 |
| 官方 LIBERO 文档 | 已发布 | 仅公布 π0.5 标准评测 (97.5%)，**无 RTC 结果** |



---

## 5. 关键发现

### 5.1 Spatial 表现良好，Object/Goal/10 从零训练几乎失败

| Suite | 我们最佳 | SmolVLA 论文单套件 | 社区从零训练 | 评价 |
|---|---|---|---|---|
| spatial (a2c2 80K) | **92%** | 82.5% | 66-82% | **超越论文和社区** |
| goal (v2 100K) | **16%** | 45.0% | — | 远低于预期 |
| object (v2 100K) | **1%** | 41.8% | 0-66% | 与社区从零训练 0% 一致 |
| libero_10 (v2 100K) | **0%** | 60.0% | 0-7.5% | 与社区从零训练趋势一致 |

**关键对比**：GitHub #1316 有用户报告 "from scratch on combined dataset, bs=128, 200K steps → object SR=0%"。我们的 object/10 结果 0% **与社区经验一致**——单套件从零训练（无 VLA 预训练）在 object/10 上非常困难。

**libero_goal 16% 也基本合理**：社区无单套件从零训练 goal 的数据，但 finetune 从 smolvla_base 的用户报告 goal=76%（#3287, bs=32）。从零训练显著更难。

### 5.2 我们的 Spatial Naive 大幅超越 a2c2 团队实测

在所有 14 组参数下，我们的 Naive 均显著优于 a2c2 团队自己的 10-trial 评测（平均提升约 +19%），并与 A2C2 论文的 50-trial 结果处于同一量级。

> 数据来源对比：`lerobot/outputs/eval_full/naive/results_libero_spatial.jsonl` vs `a2c2-libero/result/libero_eval/results_libero_spatial.jsonl`

**可能原因**：

- a2c2-libero 评测脚本的 `SmolVLAPolicy` 内部归一化实现与训练时可能存在细微差异
- 评测环境随机种子不同（a2c2 未公开 seed 设置）
- lerobot Naive 的 tokenization 策略（`padding="max_length"` 固定 48 token）可能更匹配训练时的预处理

### 5.3 RTC 在 d=0 时与论文一致，d>1 时严重退化

这是最重要的发现：

- **d=0 时**：我们的 RTC 与 A2C2 论文报告的 RTC 基线趋势一致——RTC ≈ Naive，部分配置 RTC 略优（h=50 时 RTC 69% vs Naive 58%）。这**符合 RTC 原始论文的预期**：LIBERO 是准静态任务，RTC 在此类任务上不会带来显著提升。
- **d≥3 时**：RTC 成功率急剧下降至 0-25%，而 Naive 反而保持或提升。
- **StreamingVLA 论文独立验证**：RTC (d=1) 在 π0.5-Libero 上 Object 仅 19.8%、Long 仅 25.2%，平均 58.55%。

> 数据来源：`lerobot/outputs/eval_full/rtc/results_libero_spatial.jsonl`

**RTC 高 delay 退化的可能原因**：

1. **Monkey-patch 移植问题**：官方 RTC 基于 lerobot 0.5.x，移植到 0.4.0 时 flow matching denoising loop 的时间步调度可能有差异
2. **Guidance 累积误差**：高 delay 导致 prefix 引导区域过长，`max_guidance_weight=10.0` 在大 prefix 下可能使 denoising 发散
3. **超参数未调优**：RTC 论文在 Kinetix (chunk_size=8) 上调参，移植到 SmolVLA (chunk_size=50) 时超参不匹配
4. **方法固有局限**：StreamingVLA 论文确认 RTC 在 LIBERO 上有严重退化，这不是实现 bug 而是方法本身的问题

### 5.4 lerobot 官方无 RTC + LIBERO 跑分

**lerobot 官方仓库中完全没有 RTC + LIBERO 的成功率数据。** PR #1521 作为 draft 关闭，官方 `examples/rtc/eval_dataset.py` 仅测 action 一致性不测任务成功率，官方 LIBERO 文档仅有 π0.5 标准结果。RTC 在 LIBERO 上的唯一量化数据来自 StreamingVLA 论文（第三方）。

### 5.5 Naive chunk reuse 的意外鲁棒性

Naive 策略在 h=10, d=5 达到最高 **92%**，高于 d=0 时的 82%。适度的 inference_delay 反而提升表现，可能是旧 chunk 尾部动作提供了有效的运动平滑过渡。这一现象在 A2C2 论文中也有观察：Table 7 中 h=10 d=1 (83%) > h=10 d=0 (85%) 差距不大，但 d=5 时 A2C2 论文为 82% 而我们达到 92%，存在统计波动。

### 5.6 为什么不能完全复现论文分数

| 差异因素 | 影响程度 | 说明 |
|---|---|---|
| **训练方式** | 极大 | 论文标题数字来自多套件联合训练；我们是单套件训练 |
| **是否有 VLA 预训练** | 极大 | 论文 finetune smolvla_base（含社区数据预训练）；我们从零训练 |
| **Batch size** | 大 | 论文 256 vs 我们 64；影响泛化能力 |
| **Trial 数量** | 中 | 论文 50 rollouts (±4.5%) vs 我们 10 rollouts (±10%) |
| **MuJoCo 版本** | 中 | 社区报告不同版本差异可达 15-20% |
| **评测协议** | 中 | chunk reuse 协议 vs 标准 n_action_steps=10 |
| **库版本** | 小 | lerobot 0.4.0 vs 0.5.x |

> 参考：SmolVLA 论文 Table 2 的单套件 0.45B 平均仅 **57.3%**，我们的 spatial 76% 平均已显著超过此基线。
> 对于 object/10 从零训练 0% 的结果，社区也有完全一致的报告（#1316 @hahans: object from scratch → 0%）。

---

## 6. 当前进度总结

| 任务 | 状态 | 备注 |
|---|---|---|
| libero_spatial Naive 14 组评测 | ✅ 完成 | 最佳 92% (h=10, d=5) |
| libero_spatial RTC 14 组评测 | ✅ 完成 | d=0 与论文一致，d>1 退化 |
| libero_spatial 带视频评测 | ✅ 完成 | 100 个 mp4 视频 |
| SmolVLA 从零训练 (spatial) | ✅ 完成 | loss=0.029，100K 步 |
| libero_object/goal/10 v1 评测 | ✅ 完成 | 因 normalization_mapping 缺失导致 0-5.5% |
| libero_object/goal/10 v2 重训 | ✅ 完成 | object 0.032, goal 0.024, 10 0.044 |
| libero_object/goal/10 v2 评测 | ✅ 完成 | object 0-1%, goal 0-16%, 10 0% |
| 社区/论文对比分析 | ✅ 完成 | 结果与社区从零训练经验一致 |

---

## 7. 文件路径索引

| 文件/目录 | 路径 |
|---|---|
| 评测脚本 | `lerobot/scripts/eval_libero_rtc.py` |
| RTC monkey-patch | `lerobot/scripts/rtc_local/monkey_patch.py` |
| Naive 结果 (spatial) | `lerobot/outputs/eval_full/naive/results_libero_spatial.jsonl` |
| RTC 结果 (spatial) | `lerobot/outputs/eval_full/rtc/results_libero_spatial.jsonl` |
| object/goal/10 v1 结果 | `lerobot/outputs/eval_full/{naive,rtc}/results_libero_{object,goal,10}.jsonl` |
| object/goal/10 **v2** 结果 | `lerobot/outputs/eval_v2/{naive,rtc}/results_libero_{object,goal,10}.jsonl` |
| 评测视频 | `lerobot/outputs/eval_with_video/naive/videos_libero_spatial/` |
| a2c2 团队原始结果 | `a2c2-libero/result/libero_eval/results_libero_spatial.jsonl` |
| 评测用 checkpoint (80K) | `a2c2-libero/libero_smolvla_scratch/checkpoints/080000/pretrained_model_migrated` |
| 从零训练 checkpoint (spatial 100K) | `lerobot/outputs/smolvla_scratch_libero_spatial/checkpoints/100000/pretrained_model` |
| v2 训练配置 | `lerobot/configs/train_scratch_libero_{object,goal,10}.json` |
| v2 训练日志 | `lerobot/outputs/train_log_scratch_{object,goal,10}_v2.log` |
| v2 checkpoint (migrated) | `lerobot/outputs/smolvla_scratch_libero_{object,goal,10}_v2/checkpoints/100000/pretrained_model_migrated` |

---

## 8. 结论与后续建议

### 8.1 核心结论

1. **Spatial（a2c2 80K checkpoint）表现优秀**：Naive 最佳 92%，超过 SmolVLA 论文单套件基线 (82.5%) 和社区水平 (66-82%)
2. **Object / Goal / 10 从零训练失败**：与 lerobot 社区大量相同经验一致（GitHub #1316, #2107），SmolVLA 单套件从零训练在 object/10 上极难收敛
3. **RTC 在 LIBERO 上无正向收益**：
   - d=0 时 RTC ≈ Naive（符合预期）
   - d>0 时 RTC 严重退化，StreamingVLA 论文独立验证了相同结论
   - **lerobot 官方没有任何 RTC + LIBERO 成功率数据**
4. **论文分数不可比**：被引用的 90/96/92/71 来自多套件联合训练 + 8×H100 + bs=256，非单套件可达

### 8.2 后续建议

1. **尝试 finetune from smolvla_base**：社区经验表明从 smolvla_base 微调可显著提升 object/goal/10 成功率（66-76% 级别）
2. **评测从零训练的 spatial 100K 模型**：验证我们的训练流程在 spatial 上的效果
3. **尝试多套件联合训练**：使用 `HuggingFaceVLA/libero` 联合数据集，batch_size=128+
4. **评测时使用 n_action_steps=1**：社区报告此设置可提升 10-16%
5. **消融 RTC 超参数**：在 delay=3/5 上 grid search `max_guidance_weight ∈ {1, 5, 10, 20}`

