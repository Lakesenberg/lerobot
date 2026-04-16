"""
评测 SmolVLA on LIBERO benchmark 的 chunk 管理消融实验。

【三种评测策略】
  1. 朴素 chunk 重用（naive chunk reuse，默认）：
     - 每 execute_horizon 步请求新 chunk
     - 旧 chunk 的前 inference_delay 步先执行（模拟推理延迟期间的动作缓冲）
     - 新 chunk 从第 inference_delay 步开始取 execute_horizon 步执行
     - 这与 a2c2-libero/eval_libero/evaluation_libero.py 的逻辑一致

  2. baseline（全 chunk 顺序执行，--no_rtc）：
     - 每 execute_horizon 步请求新 chunk
     - 直接从 chunk 开头取 execute_horizon 步执行，没有 delay/overlap

  3. 官方 RTC guidance（--use_rtc）：
     - 通过 monkey-patch 在 flow matching denoising loop 中注入
       RTCProcessor.denoise_step，使新 chunk 的 prefix 与上一个
       chunk 的剩余部分对齐
     - 这是 lerobot 0.5.x 的官方实现，移植到 0.4.0

【运行环境】
  conda activate lerobot                      # lerobot 0.4.0
  pip install -e .../third_party/libero       # 安装 libero

【用法示例】
  # 朴素 chunk 重用
  MUJOCO_GL=egl python scripts/eval_libero_rtc.py \\
      --checkpoint /path/to/pretrained_model \\
      --suite libero_spatial \\
      --execute_horizon 5 --inference_delay 0 --trials 10

  # 官方 RTC guidance
  MUJOCO_GL=egl python scripts/eval_libero_rtc.py \\
      --checkpoint /path/to/pretrained_model \\
      --suite libero_spatial --use_rtc \\
      --execute_horizon 10 --inference_delay 5 --trials 10

  # 14 组标准消融 + 官方 RTC
  MUJOCO_GL=egl python scripts/eval_libero_rtc.py \\
      --checkpoint /path/to/pretrained_model \\
      --suite libero_spatial --mode rtc_ablation --use_rtc --trials 10

  # baseline（无 chunk 管理）
  MUJOCO_GL=egl python scripts/eval_libero_rtc.py \\
      --checkpoint /path/to/pretrained_model \\
      --suite libero_spatial --no_rtc --execute_horizon 50 --trials 10

  # 断点续跑
  MUJOCO_GL=egl python scripts/eval_libero_rtc.py \\
      --checkpoint /path/to/pretrained_model \\
      --suite libero_spatial --mode rtc_ablation --use_rtc --trials 10 --resume
"""

import argparse
import collections
import json
import logging
import math
import os
import pathlib
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# 将 scripts/ 加入搜索路径，以便 import rtc_local
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
CHUNK_SIZE = 50
NUM_STEPS_WAIT = 10

MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}

# 14 组标准消融参数 (inference_delay, execute_horizon)
RTC_ABLATION_PAIRS = [
    (0,  1), (0,  5), (0, 10), (0, 30), (0, 40), (0, 50),
    (1, 10), (3, 10), (5, 10), (10, 10),
    (1, 40), (3, 40), (5, 40), (10, 40),
]


# ─────────────────────────── 环境工具 ────────────────────────────────────────

def _quat2axisangle(quat):
    q3 = max(-1.0, min(1.0, quat[3]))
    den = np.sqrt(1.0 - q3 ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(q3)) / den


def _get_libero_env(task, resolution, seed):
    task_bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


_TOKENIZER = None
_TOKENIZER_MAX_LEN = 48

# 归一化/反归一化参数（从训练 checkpoint 提取）
_STATE_MEAN = None  # (8,)
_STATE_STD = None   # (8,)
_ACTION_MEAN = None  # (7,)
_ACTION_STD = None   # (7,)


def init_tokenizer(policy):
    """从已加载的 SmolVLA policy 中提取 tokenizer。"""
    global _TOKENIZER, _TOKENIZER_MAX_LEN
    _TOKENIZER = policy.model.vlm_with_expert.processor.tokenizer
    _TOKENIZER_MAX_LEN = policy.config.tokenizer_max_length
    logging.info(
        f"Tokenizer 已初始化: {type(_TOKENIZER).__name__}, "
        f"max_length={_TOKENIZER_MAX_LEN}"
    )


def init_normalization(norm_checkpoint_path, device):
    """从原始 checkpoint 加载归一化参数（state mean/std, action mean/std）。

    lerobot 0.4.0 的 SmolVLAPolicy 不内置 normalize/unnormalize，需要手动完成。
    """
    global _STATE_MEAN, _STATE_STD, _ACTION_MEAN, _ACTION_STD
    import safetensors.torch as stt

    sd = stt.load_file(norm_checkpoint_path)
    _STATE_MEAN = sd["normalize_inputs.buffer_observation_state.mean"].to(device)
    _STATE_STD = sd["normalize_inputs.buffer_observation_state.std"].to(device)
    _ACTION_MEAN = sd["unnormalize_outputs.buffer_action.mean"].to(device)
    _ACTION_STD = sd["unnormalize_outputs.buffer_action.std"].to(device)
    logging.info(
        f"归一化参数已加载: state dim={_STATE_MEAN.shape[0]}, "
        f"action dim={_ACTION_MEAN.shape[0]}"
    )


def normalize_state(state_tensor):
    """归一化 state: (x - mean) / std"""
    if _STATE_MEAN is None:
        return state_tensor
    return (state_tensor - _STATE_MEAN) / _STATE_STD.clamp(min=1e-8)


def unnormalize_action(action_tensor):
    """反归一化 action: x * std + mean"""
    if _ACTION_MEAN is None:
        return action_tensor
    return action_tensor * _ACTION_STD + _ACTION_MEAN


def _obs_to_tensor(obs, task_description, device):
    """把 LIBERO obs dict 转为 SmolVLA 输入格式（含 language tokenization）。"""
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    front = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    state = np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))

    # tokenize 任务描述
    task_text = task_description if task_description.endswith("\n") else task_description + "\n"
    tok_out = _TOKENIZER(
        [task_text],
        max_length=_TOKENIZER_MAX_LEN,
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors="pt",
    )

    batch = {
        "observation.images.image": (
            torch.from_numpy(front / 255.0).permute(2, 0, 1)
            .float().unsqueeze(0).to(device)
        ),
        "observation.images.wrist_image": (
            torch.from_numpy(wrist / 255.0).permute(2, 0, 1)
            .float().unsqueeze(0).to(device)
        ),
        "observation.state": normalize_state(
            torch.from_numpy(state).float().unsqueeze(0).to(device)
        ),
        "observation.language.tokens": tok_out["input_ids"].to(device),
        "observation.language.attention_mask": tok_out["attention_mask"].to(dtype=torch.bool, device=device),
        "task": task_description,
    }
    return batch, front


# ─────────────────────────── 单 episode 推理 ─────────────────────────────────

def run_episode(
    policy,
    env,
    initial_state,
    task_description,
    max_steps,
    execute_horizon,
    inference_delay,
    chunk_reuse,
    device,
):
    """
    运行单个 episode。

    chunk_reuse=True（朴素 chunk 重用，与 a2c2-libero 一致）：
      - 每 execute_horizon 步请求新 chunk
      - 旧 chunk 的前 inference_delay 步 → 继续执行（模拟推理延迟缓冲）
      - 新 chunk 从 [inference_delay : inference_delay+execute_horizon] 替换

    chunk_reuse=False（baseline）：
      - 每 execute_horizon 步请求新 chunk
      - 顺序执行 chunk 前 execute_horizon 步，无 overlap
    """
    env.reset()
    policy.reset()
    obs = env.set_init_state(initial_state)

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    frames = [np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])]
    done = False
    t = 0

    if chunk_reuse and inference_delay > 0:
        # ── 朴素 chunk 重用模式 ─────────────────────────────────────────
        # 与 a2c2-libero evaluation_libero.py 第 250-262 行逻辑一致
        action_plan = collections.deque()
        pending_actions = collections.deque()
        first_chunk = True

        while t < max_steps:
            if len(action_plan) == 0:
                batch, frame = _obs_to_tensor(obs, task_description, device)
                frames.append(frame)

                with torch.no_grad():
                    new_chunk = policy.predict_action_chunk(batch)
                new_chunk = unnormalize_action(new_chunk.squeeze(0)).cpu().numpy()
                chunk_np = new_chunk

                if first_chunk:
                    first_chunk = False
                    exec_actions = chunk_np[:execute_horizon]
                    action_plan = collections.deque(exec_actions)
                    # 缓冲 delay 对应的后续动作
                    for a in chunk_np[execute_horizon:execute_horizon + inference_delay]:
                        pending_actions.append(a)
                else:
                    exec_actions = []
                    for _ in range(min(inference_delay, len(pending_actions))):
                        exec_actions.append(pending_actions.popleft())
                    remaining = execute_horizon - len(exec_actions)
                    exec_actions.extend(chunk_np[inference_delay:inference_delay + remaining])
                    action_plan = collections.deque(exec_actions)
                    for a in chunk_np[execute_horizon:execute_horizon + inference_delay]:
                        pending_actions.append(a)

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action)
            if done:
                break
            t += 1

    else:
        # ── baseline / delay=0 的简单模式 ────────────────────────────────
        remaining_actions = []

        while t < max_steps:
            if len(remaining_actions) == 0:
                batch, frame = _obs_to_tensor(obs, task_description, device)
                frames.append(frame)

                with torch.no_grad():
                    new_chunk = policy.predict_action_chunk(batch)
                chunk_np = unnormalize_action(new_chunk.squeeze(0)).cpu().numpy()
                remaining_actions = list(chunk_np[:execute_horizon])

            action = remaining_actions.pop(0)
            obs, _, done, _ = env.step(action)
            if done:
                break
            t += 1

    return done, frames


# ─────────────────────────── 官方 RTC episode 推理 ────────────────────────────

def run_episode_rtc(
    policy,
    env,
    initial_state,
    task_description,
    max_steps,
    execute_horizon,
    inference_delay,
    device,
):
    """
    使用官方 RTC guidance 运行单个 episode。

    与 run_episode(chunk_reuse=True) 的区别：
      - 朴素版在动作层面拼接旧/新 chunk（action-level splicing）
      - RTC 版在 flow matching denoising loop 中注入梯度 guidance，
        让新 chunk 的 prefix 自然与旧 chunk 剩余部分对齐

    流程：
      1. 首次预测：无 prev_chunk_left_over → 普通推理
      2. 后续预测：将上一个 raw chunk 从 execute_horizon 开始的剩余部分
         作为 prev_chunk_left_over 传入 predict_action_chunk，
         RTCProcessor 在 denoising 中施加 guidance
      3. 新 chunk 的前 inference_delay 步对应模拟推理延迟期间
         （旧 chunk 仍在执行），跳过这些步后取 execute_horizon 步执行
    """
    env.reset()
    policy.reset()
    obs = env.set_init_state(initial_state)

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    frames = [np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])]
    done = False
    t = 0

    prev_raw_chunk = None   # 上一次预测的原始 chunk tensor (chunk_size, action_dim)
    action_buffer = []

    while t < max_steps:
        if len(action_buffer) == 0:
            batch, frame = _obs_to_tensor(obs, task_description, device)
            frames.append(frame)

            # 构建 prev_chunk_left_over
            if prev_raw_chunk is not None:
                left_over = prev_raw_chunk[execute_horizon:]
                if left_over.shape[0] == 0:
                    left_over = None
            else:
                left_over = None

            # 带 RTC kwargs 的推理
            rtc_kwargs = {}
            if left_over is not None:
                rtc_kwargs["inference_delay"] = inference_delay
                rtc_kwargs["prev_chunk_left_over"] = left_over
                rtc_kwargs["execution_horizon"] = execute_horizon

            with torch.no_grad():
                new_chunk = policy.predict_action_chunk(batch, **rtc_kwargs)

            # new_chunk 是归一化空间的输出 (1, chunk_size, action_dim)
            norm_chunk = new_chunk.squeeze(0)  # (chunk_size, action_dim)

            if left_over is not None and inference_delay > 0:
                start_idx = inference_delay
            else:
                start_idx = 0

            end_idx = start_idx + execute_horizon
            # 只对要执行的 action 做反归一化
            actions_np = unnormalize_action(norm_chunk[start_idx:end_idx]).cpu().numpy()
            action_buffer = list(actions_np)

            # prev_raw_chunk 保留归一化空间的值，供 RTC guidance 使用
            prev_raw_chunk = norm_chunk.detach()

        action = action_buffer.pop(0)
        obs, _, done, _ = env.step(action)
        if done:
            break
        t += 1

    return done, frames


# ─────────────────────────── 整套 suite 评测 ─────────────────────────────────

def eval_suite(
    policy,
    task_suite_name,
    num_trials_per_task,
    seed,
    execute_horizon,
    inference_delay,
    chunk_reuse,
    video_out_path,
    device,
    skip_video=False,
    use_rtc=False,
):
    """对 (suite, execute_horizon, inference_delay) 组合完整评测。

    use_rtc=True 时使用官方 RTC guidance（需先 apply_rtc_patch）；
    否则使用朴素 chunk reuse 或 baseline。
    """

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"未知 suite: {task_suite_name}，可选: {list(benchmark_dict.keys())}")

    task_suite = benchmark_dict[task_suite_name]()
    max_steps = MAX_STEPS_MAP.get(task_suite_name, 520)
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm(range(task_suite.n_tasks), desc=task_suite_name):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        task_successes = 0

        n_trials = min(num_trials_per_task, len(initial_states))
        for ep_idx in tqdm(range(n_trials), desc=f"Task {task_id}", leave=False):
            try:
                if use_rtc:
                    done, frames = run_episode_rtc(
                        policy=policy,
                        env=env,
                        initial_state=initial_states[ep_idx],
                        task_description=task_description,
                        max_steps=max_steps,
                        execute_horizon=execute_horizon,
                        inference_delay=inference_delay,
                        device=device,
                    )
                else:
                    done, frames = run_episode(
                        policy=policy,
                        env=env,
                        initial_state=initial_states[ep_idx],
                        task_description=task_description,
                        max_steps=max_steps,
                        execute_horizon=execute_horizon,
                        inference_delay=inference_delay,
                        chunk_reuse=chunk_reuse,
                        device=device,
                    )
            except Exception as exc:
                logging.error(f"Episode 失败: {exc}")
                done = False
                frames = []

            if done:
                task_successes += 1
                total_successes += 1
            total_episodes += 1

            if not skip_video and frames:
                suffix = "success" if done else "failure"
                if use_rtc:
                    method_tag = "rtc"
                elif chunk_reuse:
                    method_tag = "chunk_reuse"
                else:
                    method_tag = "baseline"
                task_seg = task_description.replace(" ", "_").replace("/", "_")[:60]
                video_path = (
                    pathlib.Path(video_out_path)
                    / f"rollout_task_{task_id}_ep{ep_idx}_{task_seg}_{suffix}_{method_tag}.mp4"
                )
                try:
                    writer = imageio.get_writer(str(video_path), fps=30)
                    for f in frames:
                        writer.append_data(f)
                    writer.close()
                except Exception as exc:
                    logging.warning(f"视频保存失败: {exc}")

        logging.info(
            f"Task {task_id}: {task_successes}/{n_trials}  [{task_description[:50]}]"
        )

    sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    if use_rtc:
        method = "rtc"
    elif chunk_reuse:
        method = "chunk_reuse"
    else:
        method = "baseline"
    logging.info(
        f"[{task_suite_name}][{method}] horizon={execute_horizon} delay={inference_delay} "
        f"=> {total_successes}/{total_episodes} ({sr:.1%})"
    )
    return {
        "method":              method,
        "task_suite_name":     task_suite_name,
        "execute_horizon":     execute_horizon,
        "inference_delay":     inference_delay,
        "chunk_reuse":         chunk_reuse,
        "use_rtc":             use_rtc,
        "num_trials_per_task": num_trials_per_task,
        "total_episodes":      total_episodes,
        "total_successes":     total_successes,
        "success_rate":        round(sr, 4),
    }


# ─────────────────────────── 断点续跑 / 汇总 ────────────────────────────────

def load_completed(jsonl_path):
    completed = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                completed.add((
                    d["task_suite_name"],
                    d["execute_horizon"],
                    d["inference_delay"],
                    d.get("method", "chunk_reuse"),
                ))
            except Exception:
                pass
    return completed


def print_summary(results):
    if not results:
        return
    width = 75
    print("\n" + "=" * width)
    print(f"{'suite':<16}{'method':<14}{'horizon':>8}{'delay':>7}{'succ/total':>12}{'rate':>9}")
    print("-" * width)
    for r in sorted(
        results,
        key=lambda x: (x["task_suite_name"], x["method"], x["execute_horizon"], x["inference_delay"]),
    ):
        print(
            f"{r['task_suite_name']:<16}"
            f"{r['method']:<14}"
            f"{r['execute_horizon']:>8}"
            f"{r['inference_delay']:>7}"
            f"  {r['total_successes']:>4}/{r['total_episodes']:<5}"
            f"{r['success_rate']:>8.1%}"
        )
    print("=" * width)


def _find_norm_checkpoint(checkpoint_dir):
    """在 checkpoint 目录及上级目录中搜索包含归一化参数的 model.safetensors。

    migrated checkpoint 不含归一化参数，需要找到原始 checkpoint。
    搜索策略：如果 checkpoint_dir 名为 pretrained_model_migrated，
    则检查同级的 pretrained_model/model.safetensors。
    """
    import safetensors.torch as stt

    ckpt_path = pathlib.Path(checkpoint_dir)

    candidates = [
        ckpt_path / "model.safetensors",
    ]
    if ckpt_path.name == "pretrained_model_migrated":
        candidates.insert(0, ckpt_path.parent / "pretrained_model" / "model.safetensors")
    candidates.append(ckpt_path.parent / "pretrained_model" / "model.safetensors")

    for cand in candidates:
        if not cand.exists():
            continue
        try:
            sd = stt.load_file(str(cand))
            if "unnormalize_outputs.buffer_action.mean" in sd:
                logging.info(f"自动发现归一化参数: {cand}")
                return str(cand)
        except Exception:
            continue

    return None


# ─────────────────────────── CLI ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Chunk 管理消融评测 SmolVLA on LIBERO")
    p.add_argument(
        "--checkpoint", required=True,
        help="SmolVLA pretrained_model 目录路径（含 config.json + model.safetensors）",
    )
    p.add_argument(
        "--suite", default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    p.add_argument(
        "--mode", default="single", choices=["single", "rtc_ablation"],
        help="single: 单组参数; rtc_ablation: 标准 14 组消融实验",
    )
    p.add_argument("--execute_horizon", type=int, default=5,
                   help="每次推理后执行的步数（mode=single 时生效）")
    p.add_argument("--inference_delay",  type=int, default=0,
                   help="模拟推理延迟步数（mode=single 时生效）")
    p.add_argument("--trials",  type=int, default=10, help="每个 task 的 episode 数")
    p.add_argument("--seed",    type=int, default=7)
    p.add_argument("--out_dir", default="outputs/eval_libero_rtc",
                   help="结果输出根目录")
    p.add_argument("--resume",   action="store_true", help="跳过已完成的参数组，断点续跑")
    p.add_argument("--no_video", action="store_true", help="不保存视频（加速评测）")
    p.add_argument("--no_rtc",   action="store_true",
                   help="关闭 chunk reuse，作为 baseline 使用")
    p.add_argument("--use_rtc",  action="store_true",
                   help="启用官方 RTC guidance（monkey-patch flow matching denoising loop）")
    # RTC 超参
    p.add_argument("--rtc_schedule", default="LINEAR",
                   choices=["ZEROS", "ONES", "LINEAR", "EXP"],
                   help="RTC prefix attention schedule")
    p.add_argument("--rtc_max_gw", type=float, default=10.0,
                   help="RTC 最大 guidance weight")
    p.add_argument("--norm_checkpoint", default=None,
                   help="包含归一化参数的原始 checkpoint model.safetensors 路径。"
                        "若不指定，自动在 checkpoint 同级或上级目录搜索。")
    p.add_argument("--device", default="auto")
    return p.parse_args()


# ─────────────────────────── 主函数 ──────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logging.info(f"设备: {device}")

    use_rtc = args.use_rtc
    chunk_reuse = not args.no_rtc

    if args.mode == "rtc_ablation":
        pairs = [
            (d, h) for (d, h) in RTC_ABLATION_PAIRS
            if h >= d and (h + d) <= CHUNK_SIZE
        ]
        logging.info(f"消融模式：{len(pairs)} 组参数")
    else:
        pairs = [(args.inference_delay, args.execute_horizon)]
        logging.info(
            f"单组模式: execute_horizon={args.execute_horizon}, "
            f"inference_delay={args.inference_delay}"
        )

    logging.info(f"加载模型: {args.checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    init_tokenizer(policy)

    # 加载归一化参数
    norm_ckpt = args.norm_checkpoint
    if norm_ckpt is None:
        norm_ckpt = _find_norm_checkpoint(args.checkpoint)
    if norm_ckpt:
        init_normalization(norm_ckpt, device)
    else:
        logging.warning("未找到归一化参数，跳过 normalize/unnormalize（结果可能异常）")

    # 如果启用官方 RTC，施加 monkey-patch
    if use_rtc:
        from rtc_local.rtc_config import RTCAttentionSchedule, RTCConfig
        from rtc_local.monkey_patch import apply_rtc_patch

        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule(args.rtc_schedule),
            max_guidance_weight=args.rtc_max_gw,
            execution_horizon=args.execute_horizon,
        )
        apply_rtc_patch(policy, rtc_config)
        logging.info(
            f"官方 RTC guidance 已启用: schedule={args.rtc_schedule}, "
            f"max_gw={args.rtc_max_gw}"
        )
    elif chunk_reuse:
        logging.info("朴素 chunk 重用模式（action-level splicing，与 a2c2-libero 一致）")
    else:
        logging.info("Baseline 模式：顺序执行，无 overlap")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 确定 method 标签
    if use_rtc:
        method_label = "rtc"
    elif chunk_reuse:
        method_label = "chunk_reuse"
    else:
        method_label = "baseline"

    jsonl_path = out_dir / f"results_{args.suite}.jsonl"
    completed = load_completed(jsonl_path) if args.resume else set()
    if completed:
        logging.info(f"断点续跑：已跳过 {len(completed)} 组")

    all_results = []

    for (inf_delay, exec_horizon) in pairs:
        key = (args.suite, exec_horizon, inf_delay, method_label)
        if key in completed:
            logging.info(f"跳过 horizon={exec_horizon} delay={inf_delay}（已完成）")
            continue

        video_dir = (
            out_dir
            / f"videos_{args.suite}"
            / f"{method_label}_horizon_{exec_horizon}_delay_{inf_delay}"
        )

        result = eval_suite(
            policy=policy,
            task_suite_name=args.suite,
            num_trials_per_task=args.trials,
            seed=args.seed,
            execute_horizon=exec_horizon,
            inference_delay=inf_delay,
            chunk_reuse=chunk_reuse,
            video_out_path=str(video_dir),
            device=device,
            skip_video=args.no_video,
            use_rtc=use_rtc,
        )
        all_results.append(result)

        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print_summary(all_results)


if __name__ == "__main__":
    main()
