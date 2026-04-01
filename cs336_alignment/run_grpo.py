import os
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["VLLM_USE_V1"] = "0"
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb
import typer
from tqdm import tqdm
from typing import Literal

from sft_helper_methods import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    log_generations,
    masked_normalize
)
from grpo_helper_methods import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean
)
from drgrpo_grader import r1_zero_reward_fn

# ==========================================
# 1. vLLM Initialization & Weight Sync
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
# 2. Main GRPO Training Loop
# ==========================================
def run_grpo(
    model_path: str = "/home/ubuntu/cs336_assignments/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B",
    train_data_path: str = "/home/ubuntu/cs336_assignments/assignment5-alignment/data/MATH/sft_train.jsonl",
    val_data_path: str = "/home/ubuntu/cs336_assignments/assignment5-alignment/data/MATH/sft_validation.jsonl",
    n_grpo_steps: int = 200,
    learning_rate: float = 2e-5, # [1e-6, 5e-6, 1e-5]
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 2,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "grpo_clip", 
    # Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = False,
    cliprange: float = 0.2,
    eval_freq: int = 10,
    eval_samples: int = 1024,
):
    # Sanity checks
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    
    policy_device = "cuda:0"
    vllm_device = "cuda:1"

    # Initialize WandB
    wandb.init(project="cs336_a5_grpo", mode="offline", name=f"grpo_{loss_type}_lr{learning_rate}_epoch{epochs_per_rollout_batch}_train_batch{train_batch_size}")
    wandb.define_metric("grpo_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="grpo_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Models & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(policy_device)
    
    vllm_engine = init_vllm(model_path, device=vllm_device, seed=42, gpu_memory_utilization=gpu_memory_utilization)
    
    optimizer = torch.optim.AdamW(
        policy_model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.0, 
        betas=(0.9, 0.95)
    )

    # Data Loading
    all_train_data = [json.loads(l) for l in open(train_data_path)]
    val_data = [json.loads(l) for l in open(val_data_path)][:eval_samples]
    val_prompts = [d["prompt"] for d in val_data]
    val_gts = [d["answer"] for d in val_data]

    sampling_params = SamplingParams(
        n=group_size, 
        temperature=sampling_temperature, 
        top_p=1.0, 
        max_tokens=sampling_max_tokens, 
        min_tokens=sampling_min_tokens, 
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )

    # ==========================================
    # Main GRPO Loop
    # ==========================================
    step_pbar = tqdm(range(1, n_grpo_steps + 1), desc="GRPO Steps")
    for step in step_pbar:
        step_pbar.set_description(f"Step {step} - Sampling & Scoring")
        
        # 1. Sample Questions
        sampled_questions = random.sample(all_train_data, n_prompts_per_rollout_batch)
        db_prompts = [q["prompt"] for q in sampled_questions]
        db_gts = [q["answer"] for q in sampled_questions]

        # Note: vLLM naturally handles the `n=group_size` generation if passed a single prompt.
        # So we pass n_prompts_per_rollout_batch prompts, and it returns group_size outputs for each.
        repeated_gts = [gt for gt in db_gts for _ in range(group_size)]
        repeated_prompts = [p for p in db_prompts for _ in range(group_size)]

        # 2. Sync weights and Generate
        policy_model.eval()
        load_policy_into_vllm_instance(policy_model, vllm_engine)
        outputs = vllm_engine.generate(db_prompts, sampling_params)
        
        # Flatten outputs
        rollout_responses = []
        for out in outputs:
            for i in range(group_size):
                rollout_responses.append(out.outputs[i].text)

        # 3. Compute Advantages & Rewards
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )
        
        advantages = advantages.to(policy_device).unsqueeze(1) # (rollout_batch_size, 1)
        raw_rewards = raw_rewards.to(policy_device).unsqueeze(1) # (rollout_batch_size, 1)

        # Log training rewards
        wandb.log({
            "train/reward_mean": reward_metadata["reward/mean"],
            "train/reward_max": reward_metadata["reward/max"],
            "train/reward_min": reward_metadata["reward/min"],
            "grpo_step": step
        })

        # 4. Tokenize
        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(policy_device)
        labels = tokenized["labels"].to(policy_device)
        response_mask = tokenized["response_mask"].to(policy_device)

        # 计算最大长度
        response_lengths = response_mask.sum(dim=1)
        current_max_res_len = response_lengths.max().item()
        # 5. Compute Old Log Probs (Once per rollout batch for off-policy or grpo_clip)
        old_log_probs = None
        if loss_type == "grpo_clip" or epochs_per_rollout_batch > 1:
            old_log_probs_list = []
            with torch.inference_mode():
                for i in range(0, rollout_batch_size, micro_train_batch_size):
                    chunk_ids = input_ids[i:i + micro_train_batch_size]
                    chunk_labels = labels[i:i + micro_train_batch_size]
                    chunk_out = get_response_log_probs(policy_model, chunk_ids, chunk_labels)
                    old_log_probs_list.append(chunk_out["log_probs"])
            old_log_probs = torch.cat(old_log_probs_list, dim=0)

        # 6. Policy Training Loop
        step_pbar.set_description(f"Step {step} - Optimizing Policy")
        policy_model.train()

        epoch_pbar = tqdm(range(epochs_per_rollout_batch), desc="Training Epochs", leave=False)
        for epoch in epoch_pbar:
            indices = torch.randperm(rollout_batch_size)
            
            for i in range(0, rollout_batch_size, train_batch_size): # 这一步是在模拟 dataloader
                batch_indices = indices[i:i + train_batch_size]
                
                total_loss = 0.0
                step_entropies = []
                step_clip_fractions = []

                # Microbatch accumulation
                for j in range(0, len(batch_indices), micro_train_batch_size):
                    mb_idx = batch_indices[j:j + micro_train_batch_size]
                    
                    mb_input_ids = input_ids[mb_idx]
                    mb_labels = labels[mb_idx]
                    mb_mask = response_mask[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_raw = raw_rewards[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx] if old_log_probs is not None else None

                    # Forward pass
                    policy_out = get_response_log_probs(
                        policy_model, mb_input_ids, mb_labels, return_token_entropy=True
                    )
                    
                    # Backward pass
                    mb_loss, mb_metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_out["log_probs"],
                        response_mask=mb_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        normalize_constant=float(current_max_res_len),
                        raw_rewards=mb_raw,
                        advantages=mb_adv,
                        old_log_probs=mb_old_log_probs,
                        cliprange=cliprange
                    )
                    
                    total_loss += mb_loss.item()
                    
                    # Compute mean entropy for response tokens
                    mb_ent = masked_mean(policy_out["token_entropy"], mb_mask).mean().item()
                    step_entropies.append(mb_ent)
                    
                    if "loss/clip_fraction" in mb_metadata:
                        cf = masked_mean(mb_metadata["loss/clip_fraction"], mb_mask).mean().item()
                        step_clip_fractions.append(cf)

                # Optimizer step
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                log_dict = {
                    "train/loss": total_loss, # Scaled back up for logging
                    "train/grad_norm": grad_norm,
                    "train/entropy": np.mean(step_entropies),
                    "grpo_step": step
                }
                if step_clip_fractions:
                    log_dict["train/clip_fraction"] = np.mean(step_clip_fractions)
                
                wandb.log(log_dict)
                # 在子进度条末尾显示实时 Loss
                epoch_pbar.set_postfix({"loss": f"{total_loss:.4f}"})

        # 7. Validation Evaluation
        if step % eval_freq == 0 or step == 1 or step == n_grpo_steps:
            step_pbar.set_description(f"Step {step} - Validating")
            policy_model.eval()
            load_policy_into_vllm_instance(policy_model, vllm_engine)
            
            # Use generation settings suitable for evaluation
            eval_sampling_params = SamplingParams(
                temperature=1.0, 
                max_tokens=1024, 
                stop=["</answer>"], 
                include_stop_str_in_output=True
            )
            
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
            wandb.log({**eval_logs["metrics"], "eval_step": step})

    # Save Final Model
    output_dir = f"/home/ubuntu/cs336_assignments/assignment5-alignment/grpo_final_model/grpo_final_{loss_type}_epoch{epochs_per_rollout_batch}_train_batch{train_batch_size}"
    print(f"Saving full model to {output_dir}...")
    
    # 1. 保存模型权重与架构配置 (强制使用 safetensors，vLLM 加载更快且不会出错)
    policy_model.save_pretrained(
        save_directory=output_dir, 
        safe_serialization=True
    )
    # 2. 保存 Tokenizer (包含词表、special_tokens_map 和 chat_template)
    tokenizer.save_pretrained(save_directory=output_dir)
    # 3. 保存 Generation Config (极其关键！)
    # 确保 vLLM 在生成时知道正确的 eos_token, pad_token 和 bos_token
    if policy_model.generation_config is not None:
        policy_model.generation_config.save_pretrained(output_dir)
    else:
        print("Warning: generation_config not found, vLLM will use default HF configs.")
    # 4. 保存基础 Config (双保险，确保 config.json 完整)
    policy_model.config.save_pretrained(output_dir)
    print("Model saved successfully!")
    wandb.finish()

if __name__ == "__main__":
    typer.run(run_grpo)
    # run_grpo()