import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["VLLM_USE_V1"] = "0"
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import typer

from sft_helper_methods import (
    tokenize_prompt_and_output,
    log_generations
)
from drgrpo_grader import r1_zero_reward_fn

# ==========================================
# 1. vLLM Initialization 
# ==========================================
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# ==========================================
# 2. Main Evaluation Logic
# ==========================================
def run_eval(
    model_path: str = "/home/ubuntu/cs336_assignments/assignment5-alignment/grpo_final_model/grpo_final_grpo_clip_epoch2_train_batch256",
    val_data_path: str = "/home/ubuntu/cs336_assignments/assignment5-alignment/data/MATH/sft_validation.jsonl",
    BASE_MODEL_PATH = "/home/ubuntu/cs336_assignments/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B",
    gpu_memory_utilization: float = 0.85,
    eval_samples: int = 1024,
):
    policy_device = "cuda:0"
    vllm_device = "cuda:1"

    print(f"Loading Tokenizer and Policy Model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 评测时仍然需要 policy_model 来计算 token entropy
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(policy_device)
    policy_model.eval()
    
    print("Loading vLLM Engine with base config...")
    vllm_engine = init_vllm(BASE_MODEL_PATH, device=vllm_device, seed=42, gpu_memory_utilization=gpu_memory_utilization)

    print("Injecting finetuned weights into vLLM...")
    load_policy_into_vllm_instance(policy_model, vllm_engine)

    # Data Loading
    print(f"Loading {eval_samples} validation samples from {val_data_path}...")
    val_data = [json.loads(l) for l in open(val_data_path)][:eval_samples]
    val_prompts = [d["prompt"] for d in val_data]
    val_gts = [d["answer"] for d in val_data]

    # Evaluation Sampling Params
    eval_sampling_params = SamplingParams(
        temperature=1.0, 
        max_tokens=1024, 
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )
    
    print("Starting Evaluation (Generation & Entropy Calculation)...")
    eval_logs = log_generations(
        vllm_model=vllm_engine,
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_fn=r1_zero_reward_fn, 
        tokenize_fn=tokenize_prompt_and_output,
        prompts=val_prompts,
        ground_truths=val_gts,
        sampling_params=eval_sampling_params
    )

    # 在终端打印最终的评测指标
    print("\n" + "="*50)
    print("🎯 Validation Performance Metrics:")
    print("="*50)
    for metric_name, value in eval_logs["metrics"].items():
        print(f"{metric_name:<30}: {value:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_eval()