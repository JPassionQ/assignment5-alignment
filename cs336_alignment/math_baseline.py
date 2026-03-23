from typing import Callable, List
import json
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn

prompts = [
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question} \n Assistant: <think>",
]

sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024
)
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True

model_path = "/home/jingqi/CS336_Assignments/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B"
validation_data_path = "/home/jingqi/CS336_Assignments/assignment5-alignment/data/MATH/origin_validation.jsonl"
def evaluate_vllm(vllm_model: LLM,
                  reward_fn: Callable[[str, str], dict[str, float]],
                  prompts: List[str],
                  eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts, compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)

    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    with open(validation_data_path, 'r') as f:
        validation_data = [json.loads(line) for line in f]
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = validation_data[i]['answer']

        full_response = "<think>" + generated_text

        rewards = reward_fn(full_response, ground_truth)

        total_reward += rewards['reward']
        total_format_reward += rewards['format_reward']
        total_answer_reward += rewards['answer_reward']
    
    num_prompts = len(prompts)
    print(f"Average Reward: {total_reward / num_prompts:.4f}")
    print(f"Average Format Reward: {total_format_reward / num_prompts:.4f}")
    print(f"Average Answer Reward: {total_answer_reward / num_prompts:.4f}")

if __name__ == "__main__":
    model = LLM(model=model_path)

    with open(validation_data_path, 'r') as f:
        validation_data = [json.loads(line) for line in f]
    
    eval_prompts = [prompts[0].format(question=item['problem']) for item in validation_data]

    evaluate_vllm(model, r1_zero_reward_fn, eval_prompts, sampling_params)

"""
Average Reward: 0.0396
Average Format Reward: 0.2316
Average Answer Reward: 0.0396
"""