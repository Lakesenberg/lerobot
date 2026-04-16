"""
Monkey-patch lerobot 0.4.0 SmolVLAPolicy，注入 RTC guidance 到 denoising loop。

核心思路：
  1. 替换 policy.model.sample_actions → 新版本接受 **kwargs 并在 Euler 积分中
     调用 RTCProcessor.denoise_step
  2. 替换 policy._get_action_chunk → 透传 **kwargs
  3. 替换 policy.predict_action_chunk → 透传 **kwargs

使用方式：
  from rtc_local.monkey_patch import apply_rtc_patch
  apply_rtc_patch(policy, rtc_config)
"""

from __future__ import annotations

import logging
import types
from typing import Optional

import torch
from torch import Tensor

from rtc_local.rtc_config import RTCConfig
from rtc_local.rtc_processor import RTCProcessor

logger = logging.getLogger(__name__)

# 0.4.0 的模块级工具函数和常量
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


def _make_patched_sample_actions(rtc_processor: RTCProcessor):
    """构造带 RTC guidance 的 sample_actions 替换函数。

    闭包捕获 rtc_processor，返回一个可绑定为方法的函数。
    """

    def patched_sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs,
    ) -> Tensor:
        """带 RTC guidance 的 sample_actions（替换 0.4.0 原版）。

        额外 kwargs:
            inference_delay (int): 推理延迟步数。
            prev_chunk_left_over (Tensor | None): 上一个 chunk 的剩余动作。
            execution_horizon (int): 执行窗口大小。
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 前缀编码和 KV cache（与 0.4.0 原版完全一致）
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise

        # 提取 RTC 参数
        prev_chunk_left_over = kwargs.get("prev_chunk_left_over", None)
        inference_delay = kwargs.get("inference_delay", 0)
        execution_horizon = kwargs.get("execution_horizon", None)

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(
                time_val, dtype=torch.float32, device=device
            ).expand(bsize)

            # 0.4.0 的 denoise_step 参数顺序：
            #   (prefix_pad_masks, past_key_values, x_t, timestep)
            def _denoise_partial(input_x_t, _ts=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks, past_key_values, input_x_t, _ts
                )

            if prev_chunk_left_over is not None:
                v_t = rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time_val,
                    original_denoise_step_partial=_denoise_partial,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = _denoise_partial(x_t)

            x_t = x_t + dt * v_t

        return x_t

    return patched_sample_actions


def _patched_get_action_chunk(self, batch, noise=None, **kwargs) -> Tensor:
    """替换 0.4.0 _get_action_chunk，透传 **kwargs 到 sample_actions。"""
    for k in batch:
        if k in self._queues and k != ACTION:
            batch[k] = torch.stack(list(self._queues[k]), dim=1)

    images, img_masks = self.prepare_images(batch)
    state = self.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    actions = self.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
    )

    original_action_dim = self.config.action_feature.shape[0]
    actions = actions[:, :, :original_action_dim]

    if self.config.adapt_to_pi_aloha:
        actions = self._pi_aloha_encode_actions(actions)

    return actions


@torch.no_grad()
def _patched_predict_action_chunk(self, batch, noise=None, **kwargs) -> Tensor:
    """替换 0.4.0 predict_action_chunk，透传 **kwargs。"""
    self.eval()
    batch = self._prepare_batch(batch)
    self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
    actions = self._get_action_chunk(batch, noise, **kwargs)
    return actions


def apply_rtc_patch(
    policy,
    rtc_config: RTCConfig,
) -> RTCProcessor:
    """对 lerobot 0.4.0 SmolVLAPolicy 施加 RTC monkey-patch。

    Args:
        policy: 已加载的 SmolVLAPolicy 实例。
        rtc_config: RTC 配置。

    Returns:
        创建的 RTCProcessor 实例（可用于后续检查）。
    """
    rtc_processor = RTCProcessor(rtc_config)

    # 1) 替换 model.sample_actions
    patched_sa = _make_patched_sample_actions(rtc_processor)
    policy.model.sample_actions = types.MethodType(patched_sa, policy.model)

    # 2) 替换 policy._get_action_chunk
    policy._get_action_chunk = types.MethodType(_patched_get_action_chunk, policy)

    # 3) 替换 policy.predict_action_chunk
    policy.predict_action_chunk = types.MethodType(
        _patched_predict_action_chunk, policy
    )

    logger.info(
        f"[RTC patch] 已应用 monkey-patch: "
        f"schedule={rtc_config.prefix_attention_schedule.value}, "
        f"max_gw={rtc_config.max_guidance_weight}, "
        f"exec_horizon={rtc_config.execution_horizon}"
    )

    return rtc_processor
