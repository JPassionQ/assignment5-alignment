from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from typing import List, Callable, Dict, Tuple, Literal
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

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor
)->torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either the raw reward or an already-normalized advantage.

    Args:
    raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar reward/advantage for each rollout response.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for each token.

    Returns:
    torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in the training loop).
    """
    pre_token_pg_loss = - policy_log_probs * raw_rewards_or_advantages
    return pre_token_pg_loss

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float
)->tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:

    advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
    
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the policy being trained.

    old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the old policy.

    cliprange: float Clip parameter ϵ (e.g. 0.2).

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.

        metadata dict containing whatever you want to log. We suggest logging whether each token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of the min was lower than the LHS.
    """
    # 1. 计算当前策略与旧策略的概率比率
    # 使用 exp 来保证数值稳定性
    ratio = torch.exp(policy_log_probs - old_log_probs)  # Shape: (batch_size, sequence_length)

    # 2.将优势值广播到序列维度

    # 3. 计算GRPO-Clip的两个代理项
    surrogate1 = ratio * advantages  # Shape: (batch_size, sequence_length)
    surrogate2 = torch.clamp(
        ratio,
        1.0 - cliprange,
        1.0 + cliprange
    ) * advantages  # Shape: (batch_size, sequence_length)
    # 4.取最小值并取符号 作为 loss
    per_token_loss = -torch.min(surrogate1, surrogate2)  # Shape: (batch_size, sequence_length)

    # 5.计算元数据
    # 记录有多少token的surrogate2 < surrogate1 (当 A > 0 时)
    # 或者说记录有多少比例的token实际上被clip限制了
    with torch.no_grad():
        # 当被裁剪后的项小于未裁剪项时，说明发生了裁剪动作
        clipped = (surrogate2 < surrogate1).float()  # Shape: (batch_size, sequence_length)
        clip_fraction = clipped # 后续可以通过 masked_mean 计算具体的比例
    
    metadata = {
        "loss/clip_fraction": clip_fraction
    }

    return per_token_loss, metadata


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    根据不同的损失类型计算损失
    """
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    return loss, metadata

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
)->torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.

    Args:
    tensor: torch.Tensor The data to be averaged.

    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
    
    dim: int | None Dimension over which to average. If None, compute the mean over all masked elements.

    Returns:
    torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    masked_tensor = tensor * mask
    sum_masked = masked_tensor.sum(dim=dim)
    count = mask.sum(dim=dim)
    mean = sum_masked / count
    return mean

def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """
    # 1. 根据 loss_type 计算每个 token 的损失
    pg_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 2. 对损失进行掩码处理，确保只计算响应部分的损失
    masked_pg_loss = masked_mean(pg_loss, response_mask, dim=1)  # Shape: (batch_size,)
    # 3. 平均损失并调整以适应梯度累积
    mean_loss = masked_pg_loss.mean()  # Shape: scalar
    loss = mean_loss / gradient_accumulation_steps # Shape: scalar
    loss.backward()
    return loss.detach(), metadata
