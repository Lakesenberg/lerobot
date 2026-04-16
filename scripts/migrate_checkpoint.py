"""将 a2c2 格式的 checkpoint 迁移为 lerobot 0.4.0 兼容格式。

主要变更：
1. 从 model.safetensors 提取归一化参数，写入 preprocessor/postprocessor JSON + safetensors
2. 从 model.safetensors 移除归一化 key
3. 修复 config.json 中 image shape 为 CHW 格式
"""

import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch as stt
import torch


def migrate(src_dir: str, dst_dir: str | None = None):
    src = Path(src_dir)
    if dst_dir is None:
        dst = src.parent / (src.name + "_migrated")
    else:
        dst = Path(dst_dir)

    if dst.exists():
        print(f"目标目录已存在，跳过: {dst}")
        return str(dst)

    dst.mkdir(parents=True)

    # 加载 model.safetensors
    sd = stt.load_file(str(src / "model.safetensors"))

    norm_keys = [k for k in sd if k.startswith(("normalize_inputs.", "unnormalize_outputs."))]
    print(f"找到 {len(norm_keys)} 个归一化 key")

    # 提取归一化参数
    state_mean = sd.get("normalize_inputs.buffer_observation_state.mean")
    state_std = sd.get("normalize_inputs.buffer_observation_state.std")
    action_mean = sd.get("unnormalize_outputs.buffer_action.mean")
    action_std = sd.get("unnormalize_outputs.buffer_action.std")

    if state_mean is not None:
        # 写 preprocessor
        pre_json = {
            "steps": [
                {"type": "Identity", "key": "observation.images.image"},
                {"type": "Identity", "key": "observation.images.wrist_image"},
                {"type": "Identity", "key": "observation.language.tokens"},
                {"type": "Identity", "key": "observation.language.attention_mask"},
                {"type": "Identity", "key": "task"},
                {
                    "type": "Normalize",
                    "key": "observation.state",
                    "stats": {"mean": state_mean.tolist(), "std": state_std.tolist()},
                },
            ]
        }
        with open(dst / "policy_preprocessor.json", "w") as f:
            json.dump(pre_json, f, indent=2)

        pre_sd = {
            "policy_preprocessor_step_5_normalizer_processor.mean": state_mean,
            "policy_preprocessor_step_5_normalizer_processor.std": state_std,
        }
        stt.save_file(pre_sd, str(dst / "policy_preprocessor_step_5_normalizer_processor.safetensors"))

    if action_mean is not None:
        # 写 postprocessor
        post_json = {
            "steps": [
                {
                    "type": "Unnormalize",
                    "key": "action",
                    "stats": {"mean": action_mean.tolist(), "std": action_std.tolist()},
                }
            ]
        }
        with open(dst / "policy_postprocessor.json", "w") as f:
            json.dump(post_json, f, indent=2)

        post_sd = {
            "policy_postprocessor_step_0_unnormalizer_processor.mean": action_mean,
            "policy_postprocessor_step_0_unnormalizer_processor.std": action_std,
        }
        stt.save_file(post_sd, str(dst / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))

    # 写不含归一化 key 的 model.safetensors
    clean_sd = {k: v for k, v in sd.items() if k not in norm_keys}
    stt.save_file(clean_sd, str(dst / "model.safetensors"))
    print(f"model.safetensors: {len(sd)} -> {len(clean_sd)} keys")

    # 修复 config.json: HWC -> CHW
    with open(src / "config.json") as f:
        config = json.load(f)

    for feat_name, feat_info in config.get("input_features", {}).items():
        shape = feat_info.get("shape", [])
        if feat_info.get("type") == "VISUAL" and len(shape) == 3 and shape[2] == 3:
            feat_info["shape"] = [3, shape[0], shape[1]]
            print(f"修复 {feat_name} shape: {shape} -> {feat_info['shape']}")

    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 复制 train_config.json
    if (src / "train_config.json").exists():
        shutil.copy2(src / "train_config.json", dst / "train_config.json")

    print(f"迁移完成: {dst}")
    return str(dst)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", help="原始 checkpoint 目录")
    p.add_argument("--dst", default=None, help="目标目录（默认 src_migrated）")
    args = p.parse_args()
    migrate(args.src, args.dst)
