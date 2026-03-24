import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
# 强制禁用 vLLM 的 V1 引擎，回退到兼容作业代码的 V0 引擎
os.environ["VLLM_USE_V1"] = "0"
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb
import torch.nn.functional as F
from sft_helper_methods import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    log_generations
)
from tqdm import tqdm
from drgrpo_grader import r1_zero_reward_fn

# ==========================================
# 1. vLLM 初始化与权重同步 (直接来自文档)
# ==========================================
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    启动推理进程，使用 vLLM 在与策略模型不同的 GPU 上加载模型
    """
    vllm_set_random_seed(seed)
    
    # 猴子补丁 (Monkeypatch) 修复 TRL/vLLM 的已知设置问题
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
    """将策略模型的最新权重加载到 vLLM 实例中"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ==========================================
# 2. 核心训练函数
# ==========================================
def run_sft_experiment():
    # 超参数设置 (文档建议调优学习率和 batch_size 以达到 15% 准确率)
    model_path = "/home/ubuntu/cs336_assignments/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B"
    data_path = "/home/ubuntu/cs336_assignments/assignment5-alignment/data/MATH/sft_train.jsonl" 
    val_data_path = "/home/ubuntu/cs336_assignments/assignment5-alignment/data/MATH/sft_validation.jsonl"
    
    epochs = 1
    batch_size = 32
    micro_batch_size = 4
    gradient_accumulation_steps = batch_size // micro_batch_size
    learning_rate = 2e-5
    eval_interval = 10  # 每多少个 train_step 评估一次
    
    # 设置设备：GPU 0 用于策略模型，GPU 1 用于 vLLM
    policy_device = "cuda:0"
    vllm_device = "cuda:1"

    # ==========================================
    # 离线 wandb 初始化
    # ==========================================
    wandb.init(project="cs336_a5_sft", mode="offline", name="sft_run_1_train_set_full")
    
    # 设置 wandb 指标关联 
    wandb.define_metric("train_step") 
    wandb.define_metric("eval_step") 
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # ==========================================
    # 加载模型与 Tokenizer
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" 
    ).to(policy_device)
    policy_model.train()

    # 初始化 vLLM
    vllm_engine = init_vllm(model_path, device=vllm_device, seed=42)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    # ==========================================
    # 数据加载 (伪代码：你需要实现一个简单的 Dataset 类)
    # ==========================================
    # 假设 load_jsonl 返回 [{"prompt": ..., "response": ...}, ...]
    train_data = [json.loads(l) for l in open(data_path)]
    val_data = [json.loads(l) for l in open(val_data_path)]
    val_prompts = [d["prompt"] for d in val_data]
    val_gts = [d["answer"] for d in val_data]
    
    train_loader = DataLoader(train_data, batch_size=micro_batch_size, shuffle=True)

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, 
        stop=["</answer>"], include_stop_str_in_output=True
    )

    # ==========================================
    # 训练循环 (Algorithm 1) 
    # ==========================================
    train_step = 0
    eval_step = 0

    for epoch in range(epochs):
        # 使用 tqdm 包装 train_loader，显式显示当前 Epoch 的进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for idx, batch in enumerate(pbar):
            
            # 定期评估：仅在达到 eval_interval 且是当前 train_step 的第一个 micro_batch 时执行
            if train_step % eval_interval == 0 and train_step > 0 and idx % gradient_accumulation_steps == 0:
                # 暂停进度条打印，避免评估日志打断进度条显示
                pbar.write(f"\n[Step {train_step}] 开始执行验证集评估...")

                policy_model.eval()
                # 1. 同步权重到 vLLM
                load_policy_into_vllm_instance(policy_model, vllm_engine)
                
                # 2. 生成并记录 (调用上一轮完善的 log_generations)
                eval_logs = log_generations(
                    vllm_model=vllm_engine,
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    reward_fn=r1_zero_reward_fn, 
                    tokenize_fn=tokenize_prompt_and_output,
                    prompts=val_prompts,
                    ground_truths=val_gts,
                    sampling_params=sampling_params
                )
                
                # 3. 记录到 wandb 且终端打印核心指标
                wandb.log({**eval_logs["metrics"], "eval_step": eval_step})
                pbar.write(f"[Step {train_step}] 评估完成! 准确率: {eval_logs['metrics']['eval/accuracy']:.2%} | 平均长度: {eval_logs['metrics']['eval/avg_response_length']:.1f}")

                eval_step += 1
                policy_model.train()

            # --- 核心微批次训练步骤 ---
            prompts = batch["prompt"]
            responses = batch["response"]
            
            # 1. 标记化并构建 mask
            tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized["input_ids"].to(policy_device)
            labels = tokenized["labels"].to(policy_device)
            response_mask = tokenized["response_mask"].to(policy_device)

            # 2. 前向传播
            logits = policy_model(input_ids).logits
            log_probs = F.log_softmax(logits, dim=-1)
            # 提取目标 token 的 log prob
            policy_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            # 计算当前 batch 的 response 的平均长度
            avg_response_len = response_mask.sum(dim=-1).float().mean().item()
            
            # 3. 计算损失并反向传播 
            # (注意：sft_microbatch_train_step 内部已经执行了 loss / gradient_accumulation_steps 以及 loss.backward())
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=avg_response_len
            )

            # 4. 梯度累积与更新
            if (idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪：截断值为 1.0
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                real_loss = loss.item() * gradient_accumulation_steps
                # 记录训练指标
                wandb.log({
                    "train/loss": real_loss / gradient_accumulation_steps,
                    "train_step": train_step
                })

                # 实时更新进度条后缀，显示当前 step 和 loss
                pbar.set_postfix({
                    "step": train_step,
                    "loss": f"{real_loss / gradient_accumulation_steps:.4f}"
                })

                train_step += 1
    # 在训练结束的时候评测一次
    policy_model.eval()
    # 1. 同步权重到 vLLM
    load_policy_into_vllm_instance(policy_model, vllm_engine)
    
    # 2. 生成并记录
    eval_logs = log_generations(
        vllm_model=vllm_engine,
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_fn=r1_zero_reward_fn, 
        tokenize_fn=tokenize_prompt_and_output,
        prompts=val_prompts,
        ground_truths=val_gts,
        sampling_params=sampling_params
    )
    
    # 3. 记录到 wandb 且终端打印核心指标
    wandb.log({**eval_logs["metrics"], "eval_step": eval_step})
    # 保存最终模型
    output_dir = "/home/ubuntu/cs336_assignments/assignment5-alignment/sft_final_model"
    # policy_model.save_pretrained(save_directory=output_dir)
    # tokenizer.save_pretrained(save_directory=output_dir)
    wandb.finish()

if __name__ == "__main__":
    run_sft_experiment()