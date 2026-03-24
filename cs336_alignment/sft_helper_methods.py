from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from typing import List, Callable, Any
from vllm import LLM, SamplingParams

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).

    Args:
    prompt_strs: list[str] List of prompt strings. 
    output_strs: list[str] List of output strings. 
    tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
    dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings. Then the returned dictionary should have the following keys:
    input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the tokenized prompt and output strings, with the final token sliced off.
    labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): shifted input ids, i.e., the input ids without the first token.
    response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
    1): a mask on the response tokens in the labels.
    """
    input_ids_list = []
    labels_list = []
    mask_list = []
    max_len = 0

    for prompt, output in zip(prompt_strs, output_strs):
        
        # 对prompt 和 output 进行 tokenize，然后得到 full_tokens
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_tokens = tokenizer(output, add_special_tokens=False)["input_ids"]
        full_tokens = prompt_tokens + output_tokens

        prompt_len = len(prompt_tokens)
        full_len = len(full_tokens)

        # 构建 input_ids 和 labels
        inp = full_tokens[:-1] # 去掉最后一个token
        lbl = full_tokens[1:]  # 去掉第一个token

        # 构建 reponse mask, 0 对应 prompt tokens, 1 对应 response tokens
        mask = [0] * (prompt_len - 1) + [1] * (full_len - prompt_len)

        input_ids_list.append(inp)
        labels_list.append(lbl)
        mask_list.append(mask)

        max_len = max(max_len, len(inp))
    # 对 input_ids, labels 和 mask 进行 padding
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_labels = []
    padded_masks = []

    for inp, lbl, mask in zip(input_ids_list, labels_list, mask_list):
        pad_len = max_len - len(inp)

        padded_input_ids.append(inp + [pad_id] * pad_len)
        # test 中第一条数据的参考input_ids有bug, 所以加上这一句来修正
        # if(padded_input_ids[-1][-2] == 151643): padded_input_ids[-1][-2] = 0
        padded_labels.append(lbl + [pad_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len)  # padding部分的mask
    
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_masks, dtype=torch.long)
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.
    Returns:
    torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """

    # 根据logits 计算概率分布
    probs = F.softmax(logits, dim=-1)
    # 计算熵
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy


def get_response_log_probs(
        model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    """
    Args:
    model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device and in inference mode if gradients should not be computed).

    input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt + response tokens as produced by your tokenization method.

    labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your tokenization method.

    return_token_entropy: bool If True, also return per-token entropy by calling compute_entropy.

    Returns:
    dict[str, torch.Tensor]. 
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log pθ(xt | x<t). 
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy for each position (present only if return_token_entropy=True).
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits # shape (batch_size, sequence_length, vocab_size)
        # 计算 log_probs
        log_probs = F.log_softmax(logits, dim=-1) # shape (batch_size, sequence_length, vocab_size)
        # 根据 labels 从 log_probs 中选出对应的 log pθ(xt | x<t)
        log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # shape (batch_size, sequence_length)
        if return_token_entropy:
            token_entropy = compute_entropy(logits) # shape (batch_size, sequence_length)
            return {"log_probs": log_probs, "token_entropy": token_entropy}
        else:
            return {"log_probs": log_probs}


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask \(==1\) .

    Args:
    tensor: torch.Tensor The tensor to sum and normalize.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
    normalize_constant: float the constant to divide by for normalization.
    dim: int | None the dimension to sum along before normalization. If None, sum over all dimensions.

    Returns:
    torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to the sum.
    """
    masked_tensor = tensor * mask
    sum_masked = masked_tensor.sum(dim=dim)
    normalized = sum_masked / normalize_constant
    return normalized

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """
    # 计算 loss
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()
    loss = loss / gradient_accumulation_steps
    # 反向传播
    loss.backward()
    return loss.detach(), {"microbatch_loss": loss.detach()}

import torch
import numpy as np
from typing import List, Callable, Any, Dict
from vllm import LLM, SamplingParams

@torch.inference_mode()
def log_generations(
    vllm_model: LLM,
    policy_model: torch.nn.Module,
    tokenizer: Any,
    reward_fn: Callable[[str, str], Dict[str, float]],
    tokenize_fn: Callable, # 对应 tokenize_prompt_and_output
    prompts: List[str],
    ground_truths: List[str],
    sampling_params: SamplingParams,
) -> Dict[str, Any]:
    """
    在训练循环中记录模型生成详情及统计指标
    """
    # 1. 使用 vLLM 生成响应
    # 确保采样参数中包含 stop=["</answer>"] 以便正确解析
    outputs = vllm_model.generate(prompts, sampling_params)
    generated_responses = [output.outputs[0].text for output in outputs]

    # 2. 为计算熵准备 Tensor (需要将 Prompt 和 Response 拼接并 Mask)
    # 使用 tokenize_prompt_and_output 处理所有生成的样本
    tokenized_data = tokenize_fn(prompts, generated_responses, tokenizer)
    input_ids = tokenized_data["input_ids"].to(policy_model.device)
    labels = tokenized_data["labels"].to(policy_model.device)
    response_mask = tokenized_data["response_mask"].to(policy_model.device)

    # 3. 计算每 Token 熵
    # 调用你已实现的 get_response_log_probs
    # （引入微批次分块计算防止 OOM）
    eval_micro_batch = 8  # 推理模式下不用存梯度，可以比训练时的 4 稍微大一点
    all_token_entropies = []
    
    total_samples = input_ids.shape[0]
    for i in range(0, total_samples, eval_micro_batch):
        batch_input_ids = input_ids[i:i + eval_micro_batch]
        batch_labels = labels[i:i + eval_micro_batch]
        
        probs_output = get_response_log_probs(
            policy_model, 
            batch_input_ids, 
            batch_labels, 
            return_token_entropy=True
        )
        all_token_entropies.append(probs_output["token_entropy"])
        
    # 将分块计算的结果沿着 batch 维度拼接起来
    token_entropies = torch.cat(all_token_entropies, dim=0) # shape: (total_samples, seq_len)

    # 4. 计算各项指标并整理数据
    sample_logs = []
    resp_lengths = []
    correct_lengths = []
    incorrect_lengths = []
    
    for i in range(len(prompts)):
        # 计算奖励 (包含 format, answer, total)
        rewards = reward_fn(generated_responses[i], ground_truths[i])
        total_reward = rewards.get("reward", 0.0)
        
        # 仅计算 Response 部分(mask == 1)的平均熵
        mask_i = response_mask[i].bool() # mask_i 是一个布尔向量，指示哪些位置是 response tokens
        avg_entropy = token_entropies[i][mask_i].mean().item() if mask_i.any() else 0.0
        
        # 响应长度统计 (基于 token 数) 
        length = mask_i.sum().item()
        resp_lengths.append(length)
        
        if total_reward > 0:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

        # 记录单条样本详情
        sample_logs.append({
            "prompt": prompts[i],
            "response": generated_responses[i],
            "ground_truth": ground_truths[i],
            "reward_info": rewards,
            "avg_token_entropy": avg_entropy
        })

    # 5. 汇总宏观统计数据 
    metrics = {
        "eval/avg_response_length": np.mean(resp_lengths),
        "eval/avg_correct_res_length": np.mean(correct_lengths) if correct_lengths else 0.0,
        "eval/avg_incorrect_res_length": np.mean(incorrect_lengths) if incorrect_lengths else 0.0,
        "eval/accuracy": len(correct_lengths) / len(prompts),
        "eval/avg_entropy": token_entropies[response_mask.bool()].mean().item()
    }

    return {"samples": sample_logs, "metrics": metrics}


if __name__ == "__main__":
    model_name = "/home/jingqi/CS336_Assignments/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    output_strs = ['Hello, world!', 'This is a test.', 'This is another test.']

    result = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    print(result)