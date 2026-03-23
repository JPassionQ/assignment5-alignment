from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from torch.nn import functional as F

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

if __name__ == "__main__":
    model_name = "/home/jingqi/CS336_Assignments/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    output_strs = ['Hello, world!', 'This is a test.', 'This is another test.']

    result = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    print(result)