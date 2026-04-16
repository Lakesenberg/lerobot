"""
从 lerobot 0.5.x 提取的 RTCProcessor 核心逻辑。
实现 flow matching denoising loop 中的 RTC guidance。

参考：
  https://github.com/Physical-Intelligence/real-time-chunking-kinetix
  https://www.physicalintelligence.company/download/real_time_chunking.pdf
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import torch
from torch import Tensor

from rtc_local.rtc_config import RTCAttentionSchedule, RTCConfig

logger = logging.getLogger(__name__)


class RTCProcessor:
    """Real-Time Chunking 处理器。

    在 flow matching denoising 的每一步对 velocity 施加 prefix guidance，
    使新生成的 action chunk 与上一个 chunk 的剩余部分保持一致。
    """

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Optional[Tensor],
        inference_delay: int,
        time: float,
        original_denoise_step_partial: Callable[[Tensor], Tensor],
        execution_horizon: Optional[int] = None,
    ) -> Tensor:
        """在单步 denoising 中注入 RTC guidance。

        Args:
            x_t: 当前噪声/状态 (B, T, A) 或 (T, A)。
            prev_chunk_left_over: 上一个 chunk 未执行的剩余动作，
                (B, T_prev, A) 或 (T_prev, A)。为 None 时不施加 guidance。
            inference_delay: 用于构建 prefix weights 的起始索引。
            time: 归一化时间 [0, 1]，lerobot 中从 1 递减到 0。
            original_denoise_step_partial: 原始 denoiser，
                签名 (x_t) -> v_t。
            execution_horizon: 用于构建 prefix weights 的结束索引。
                为 None 时使用 rtc_config.execution_horizon。

        Returns:
            带 RTC guidance 的 velocity，形状与 v_t 相同。
        """
        # lerobot 实现中 time 从 1→0，原始 RTC 论文从 0→1
        tau = 1 - time

        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)

        x_t = x_t.clone().detach()

        squeezed = False
        if len(x_t.shape) < 3:
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if len(prev_chunk_left_over.shape) < 3:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        if execution_horizon > prev_chunk_left_over.shape[1]:
            execution_horizon = prev_chunk_left_over.shape[1]

        batch_size = x_t.shape[0]
        action_chunk_size = x_t.shape[1]
        action_dim = x_t.shape[2]

        # 对齐形状：如果 prev_chunk_left_over 短于当前 chunk，用零填充
        if (prev_chunk_left_over.shape[1] < action_chunk_size
                or prev_chunk_left_over.shape[2] < action_dim):
            padded = torch.zeros(
                batch_size, action_chunk_size, action_dim, device=x_t.device
            )
            padded[
                :,
                : prev_chunk_left_over.shape[1],
                : prev_chunk_left_over.shape[2],
            ] = prev_chunk_left_over
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_t.shape, (
            f"填充后形状不匹配: prev={prev_chunk_left_over.shape} vs x_t={x_t.shape}"
        )

        # 构建 prefix weights: inference_delay 之前为 1，execution_horizon 之后为 0，中间渐变
        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
            .to(x_t.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        # 计算 guidance correction
        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)

            x1_t = x_t - time * v_t
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(
                x1_t, x_t, grad_outputs, retain_graph=False
            )[0]

        # guidance weight 计算
        max_gw = torch.as_tensor(self.rtc_config.max_guidance_weight)
        tau_t = torch.as_tensor(tau)
        sq_one_minus_tau = (1 - tau_t) ** 2
        inv_r2 = (sq_one_minus_tau + tau_t ** 2) / sq_one_minus_tau
        c = torch.nan_to_num((1 - tau_t) / tau_t, posinf=max_gw)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_gw)
        guidance_weight = torch.minimum(guidance_weight, max_gw)

        result = v_t - guidance_weight * correction

        if squeezed:
            result = result.squeeze(0)

        return result

    # ─────────────── prefix weights 构建 ───────────────

    def get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = min(start, end)
        schedule = self.rtc_config.prefix_attention_schedule

        if schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif schedule == RTCAttentionSchedule.LINEAR:
            lin = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif schedule == RTCAttentionSchedule.EXP:
            lin = self._linweights(start, end, total)
            lin = lin * torch.expm1(lin).div(math.e - 1)
            weights = self._add_trailing_zeros(lin, total, end)
            weights = self._add_leading_ones(weights, start, total)
        else:
            raise ValueError(f"未知 schedule: {schedule}")

        return weights

    def _linweights(self, start: int, end: int, total: int) -> Tensor:
        skip = max(total - end, 0)
        n = total - skip - start
        if end <= start or n <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, n + 2)[1:-1]

    def _add_trailing_zeros(self, w: Tensor, total: int, end: int) -> Tensor:
        z = total - end
        return torch.cat([w, torch.zeros(z)]) if z > 0 else w

    def _add_leading_ones(self, w: Tensor, start: int, total: int) -> Tensor:
        n = min(start, total)
        return torch.cat([torch.ones(n), w]) if n > 0 else w
