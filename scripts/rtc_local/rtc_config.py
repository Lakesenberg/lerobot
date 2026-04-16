"""
从 lerobot 0.5.x 提取的 RTCConfig 和 RTCAttentionSchedule。
自包含，不依赖 lerobot.configs.types。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RTCAttentionSchedule(str, Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclass
class RTCConfig:
    """Real-Time Chunking 推理配置。"""

    enabled: bool = False
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight 必须为正数，当前值: {self.max_guidance_weight}")
