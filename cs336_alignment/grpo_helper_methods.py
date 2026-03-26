from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from typing import List, Callable, Dict, Tuple
from vllm import LLM, SamplingParams


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".

    rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.

    repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.

    group_size: int Number of responses per question (group).

    advantage_eps: float Small constant to avoid division by zero in normalization.

    normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise subtract only the group mean.

    Returns:

    tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    # 1. 计算所有响应的原始奖励值
    raw_reward_list = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(resp, gt)
        # 提取总奖励字段 "reward"
        raw_reward_list.append(reward_dict.get("reward", 0.0))
    
    # 转换为 Tensor 以便进行向量化运算
    raw_rewards = torch.tensor(raw_reward_list, dtype=torch.float32)
    
    # 2. 将奖励值重塑为 (num_groups, group_size) 
    # rollout_batch_size = n_prompts_per_rollout_batch * group_size
    grouped_rewards = raw_rewards.view(-1, group_size)
    
    # 计算组内平均值 
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    
    # 3. 根据参数选择归一化策略
    if normalize_by_std:
        # 使用组内标准差进行归一化 
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        # 仅减去均值 (Dr. GRPO 建议的简化方式)
        advantages = grouped_rewards - group_means
    
    # 4. 展平回原始形状 (rollout_batch_size,)
    advantages = advantages.view(-1)
    
    # 5. 整理元数据用于日志记录
    metadata = {
        "reward/mean": raw_rewards.mean().item(),
        "reward/std": raw_rewards.std().item(),
        "reward/max": raw_rewards.max().item(),
        "reward/min": raw_rewards.min().item(),
    }
    
    return advantages, raw_rewards, metadata

