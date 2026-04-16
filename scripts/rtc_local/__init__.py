"""
从 lerobot 0.5.x 提取的 RTC (Real-Time Chunking) 本地模块。

用于在 lerobot 0.4.0 (Python 3.10) 环境下通过 monkey-patch
实现官方 RTC guidance 评测，无需升级 lerobot 版本。
"""

from rtc_local.rtc_config import RTCAttentionSchedule, RTCConfig
from rtc_local.rtc_processor import RTCProcessor
from rtc_local.monkey_patch import apply_rtc_patch

__all__ = [
    "RTCConfig",
    "RTCAttentionSchedule",
    "RTCProcessor",
    "apply_rtc_patch",
]
