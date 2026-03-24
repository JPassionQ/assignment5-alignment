import os
import json
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["VLLM_USE_V1"] = "0"
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb
import torch.nn.functional as F
from tqdm import tqdm

from sft_helper_methods import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    log_generations
)
from drgrpo_grader import r1_zero_reward_fn

# ==========================================
# 1. vLLM 初始化与权重同步 
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
# 3. 核心专家迭代算法
# ==========================================
def run_expert_iteration():
    # --- 超参数设置 ---
    model_path = "/home/ubuntu/cs336_assignments/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B"
    train_data_path = "/home/ubuntu/cs336_assignments/assignment5-alignment/data/MATH/sft_train.jsonl" 
    val_data_path = "/home/ubuntu/cs336_assignments/assignment5-alignment/data/MATH/sft_validation.jsonl"
    
    n_ei_steps = 5             # 专家迭代循环总次数
    db_size = 1024             # 每次迭代采样的题目数量 Db，可在 {512, 1024, 2048} 中调整 
    G = 4                      # 每道题目生成 G 个候选答案 (Rollouts)
    sft_epochs = 1             # 每次 SFT 的 Epoch 数量
    
    sft_batch_size = 64
    sft_micro_batch_size = 2
    gradient_accumulation_steps = sft_batch_size // sft_micro_batch_size
    learning_rate = 1e-5
    
    policy_device = "cuda:0"
    vllm_device = "cuda:1"

    wandb.init(project="cs336_a5_expert_iteration", mode="offline", name=f"ei_db{db_size}_G{G}_ep{sft_epochs}")
    wandb.define_metric("ei_step") 
    wandb.define_metric("sft_step") 
    wandb.define_metric("train/*", step_metric="sft_step")
    wandb.define_metric("eval/*", step_metric="ei_step")

    # --- 模型与环境初始化 ---
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(policy_device)
    vllm_engine = init_vllm(model_path, device=vllm_device, seed=42)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    # --- 数据加载 ---
    all_train_data = [json.loads(l) for l in open(train_data_path)]
    val_data = [json.loads(l) for l in open(val_data_path)][:200]
    val_prompts = [d["prompt"] for d in val_data]
    val_gts = [d["answer"] for d in val_data]

    # vLLM 生成参数：注意添加了 sampling_min_tokens 
    sampling_params = SamplingParams(
        n=G, # 为每个 prompt 采样 G 个回复
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        min_tokens=4, 
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )

    global_sft_step = 0

    # ==========================================
    # 开始专家迭代大循环 (Algorithm 2) 
    # ==========================================
    for ei_step in range(1, n_ei_steps + 1):
        print(f"\n========== Expert Iteration Step {ei_step}/{n_ei_steps} ==========")
        
        # 1. 采样题目批次 Db
        sampled_questions = random.sample(all_train_data, db_size)
        db_prompts = [d["prompt"] for d in sampled_questions]
        db_gts = [d["answer"] for d in sampled_questions]

        # 2. 同步模型权重到 vLLM
        policy_model.eval()
        load_policy_into_vllm_instance(policy_model, vllm_engine)

        # 3. 使用旧策略采样 G 个输出
        print(f"[{ei_step}] 生成推理轨迹 (Rollouts)...")
        outputs = vllm_engine.generate(db_prompts, sampling_params)
        
        # 4. 评分并过滤出正确答案以构建 D_sft 
        print(f"[{ei_step}] 评分并过滤数据集...")
        sft_dataset = []
        for i, output in enumerate(outputs):
            prompt = db_prompts[i]
            ground_truth = db_gts[i]
            
            # 由于 n=G，每个 output 会包含 G 个生成的序列
            for gen_idx in range(G):
                generated_text = output.outputs[gen_idx].text
                rewards = r1_zero_reward_fn(generated_text, ground_truth)
                
                # 保留答对的数据
                if rewards.get("answer_reward", 0.0) > 0.0 or rewards.get("reward", 0.0) > 0.0:
                    sft_dataset.append({"prompt": prompt, "response": generated_text})

        print(f"[{ei_step}] 过滤完成! 共生成 {db_size * G} 条，答对 {len(sft_dataset)} 条 (合格率: {len(sft_dataset)/(db_size*G):.2%})")

        # 5. 在验证集上评估当前模型
        print(f"[{ei_step}] 运行验证集评估...")
        eval_logs = log_generations(
            vllm_model=vllm_engine,
            policy_model=policy_model,
            tokenizer=tokenizer,
            reward_fn=r1_zero_reward_fn, 
            tokenize_fn=tokenize_prompt_and_output,
            prompts=val_prompts,
            ground_truths=val_gts,
            sampling_params=SamplingParams(temperature=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)
        )
        wandb.log({**eval_logs["metrics"], "ei_step": ei_step})

        # 如果没有正确答案，跳过本轮 SFT
        if len(sft_dataset) == 0:
            print(f"[{ei_step}] 警告：本轮未产生任何正确解答，跳过 SFT。")
            continue

        # 6. SFT 模型更新 
        print(f"[{ei_step}] 开始 SFT 训练 ({sft_epochs} Epochs)...")
        policy_model.train()
        train_loader = DataLoader(sft_dataset, batch_size=sft_micro_batch_size, shuffle=True)
        
        for epoch in range(sft_epochs):
            pbar = tqdm(train_loader, desc=f"EI {ei_step} - SFT Epoch {epoch+1}/{sft_epochs}")
            for idx, batch in enumerate(pbar):
                
                tokenized = tokenize_prompt_and_output(batch["prompt"], batch["response"], tokenizer)
                input_ids = tokenized["input_ids"].to(policy_device)
                labels = tokenized["labels"].to(policy_device)
                response_mask = tokenized["response_mask"].to(policy_device)

                logits = policy_model(input_ids).logits
                log_probs = F.log_softmax(logits, dim=-1)
                policy_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    normalize_constant=1.0
                )

                if (idx + 1) % gradient_accumulation_steps == 0 or (idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    real_loss = loss.item() * gradient_accumulation_steps
                    wandb.log({"train/ei_sft_loss": real_loss, "sft_step": global_sft_step, "ei_step": ei_step})
                    pbar.set_postfix({"loss": f"{real_loss:.4f}"})
                    global_sft_step += 1

    # 训练结束，保存模型
    output_dir = "/home/ubuntu/cs336_assignments/assignment5-alignment/ei_final_model"
    policy_model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    wandb.finish()

if __name__ == "__main__":
    run_expert_iteration()