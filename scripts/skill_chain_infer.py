#!/usr/bin/env python
"""
Multi-skill chaining inference script for LeRobot.

Sequentially runs multiple trained policies on the same robot, using the official
record_loop() for each skill. Based on the pattern from examples/lekiwi/evaluate.py.

Example (single arm SO101):
    python scripts/skill_chain_infer.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5AB01799231 \
        --robot.id=my_follower \
        --robot.cameras='{"front_cam": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
        --fps=30 \
        --display_data=true \
        --skills='[
            {"path": "user/act_pour_oil",     "task": "Pour oil into the pan",     "episode_time_s": 10},
            {"path": "user/act_crack_egg",    "task": "Crack egg into the pan",    "episode_time_s": 15},
            {"path": "user/act_pour_noodles", "task": "Pour noodles into the pan", "episode_time_s": 12}
        ]'

Example (bimanual SO101):
    python scripts/skill_chain_infer.py \
        --robot.type=bi_so101_follower \
        --robot.left_arm_port=/dev/tty.usbmodemXXXX \
        --robot.right_arm_port=/dev/tty.usbmodemYYYY \
        --robot.id=bimanual_follower \
        --robot.cameras='...' \
        --fps=30 \
        --skills='[...]'

Keyboard controls during inference:
    Right arrow: skip current skill, move to next
    ESC: stop entire chain
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    bi_so100_follower,
    bi_so101_follower,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun


@dataclass
class SkillChainConfig:
    robot: RobotConfig
    # JSON string: list of {"path": "...", "task": "...", "episode_time_s": N}
    skills: str = "[]"
    fps: int = 30
    # Seconds to pause between skills for the robot to stabilize
    transition_pause_s: float = 1.0
    display_data: bool = False
    play_sounds: bool = True


def load_skill(
    skill_dict: dict,
    robot: Robot,
    fps: int,
) -> dict[str, Any]:
    """
    Load a single skill's policy and processors.
    Follows the same pattern as examples/lekiwi/evaluate.py.
    """
    model_path = skill_dict["path"]
    task = skill_dict["task"]
    episode_time_s = skill_dict.get("episode_time_s", 30)

    logging.info(f"Loading policy from: {model_path}")

    # Load policy (auto-detects type: ACT, SmolVLA, Pi0, etc.)
    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_cfg.pretrained_path = model_path

    # Create a temporary dataset just to get the features/stats for the processors
    # This follows the same approach as the official evaluate examples
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=f"local/eval_{task.replace(' ', '_')[:30]}",
        fps=fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=False,
        image_writer_threads=4,
    )

    # Load the policy model
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    # Build pre/post processors (loads normalization stats from the pretrained checkpoint)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
        },
    )

    logging.info(f"  Policy type: {policy_cfg.type}, device: {policy.config.device}")

    return {
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "dataset": dataset,
        "task": task,
        "episode_time_s": episode_time_s,
    }


@parser.wrap()
def skill_chain(cfg: SkillChainConfig):
    """Main entry point: load all skills and run them sequentially via record_loop()."""
    init_logging()

    if cfg.display_data:
        init_rerun(session_name="skill_chain")

    # Parse skills JSON
    skills_list = json.loads(cfg.skills) if isinstance(cfg.skills, str) else cfg.skills
    if not skills_list:
        raise ValueError("No skills provided. Use --skills='[{\"path\":..., \"task\":..., \"episode_time_s\":...}]'")

    logging.info(f"Skill chain with {len(skills_list)} skills:")
    for i, s in enumerate(skills_list):
        logging.info(f"  {i+1}. {s['task']} (model: {s['path']}, time: {s.get('episode_time_s', 30)}s)")

    # Setup robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    # Default processors (identity pipelines for robot-level processing)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Keyboard listener
    listener, events = init_keyboard_listener()

    # Load all skills upfront
    logging.info("Loading all policies...")
    skills = []
    for s in skills_list:
        skill = load_skill(s, robot, cfg.fps)
        skills.append(skill)
    logging.info("All policies loaded.\n")

    # Execute skills sequentially
    for i, skill in enumerate(skills):
        if events["stop_recording"]:
            break

        task = skill["task"]
        log_say(f"Skill {i+1} of {len(skills)}: {task}", cfg.play_sounds)
        logging.info(f"{'='*60}")
        logging.info(f"Running skill {i+1}/{len(skills)}: {task}")
        logging.info(f"  Model: {skills_list[i]['path']}")
        logging.info(f"  Max time: {skill['episode_time_s']}s")
        logging.info(f"{'='*60}")

        # Reset the exit_early flag for each skill
        events["exit_early"] = False

        # Run inference using the official record_loop
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.fps,
            policy=skill["policy"],
            preprocessor=skill["preprocessor"],
            postprocessor=skill["postprocessor"],
            dataset=skill["dataset"],
            control_time_s=skill["episode_time_s"],
            single_task=task,
            display_data=cfg.display_data,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        logging.info(f"Skill '{task}' finished.")

        # Pause between skills to stabilize
        if i < len(skills) - 1 and not events["stop_recording"]:
            log_say("Transitioning to next skill", cfg.play_sounds)
            time.sleep(cfg.transition_pause_s)

    # Cleanup
    log_say("Skill chain complete", cfg.play_sounds, blocking=True)
    robot.disconnect()
    if listener is not None:
        listener.stop()


def main():
    skill_chain()


if __name__ == "__main__":
    main()
