"""
将原始的数据集按照文档要求的SFT数据格式进行转换
"""

import json
import os

def convert_math_to_sft(input_path, output_path):
    # 文档定义的 r1_zero prompt 模板
    prompt_template = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it.\n"
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.\n"
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
        "User: {question}\n"
        "Assistant: <think>"
    )

    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            
            # 提取原始字段
            question = data.get("problem", "")
            solution = data.get("solution", "")
            answer = data.get("answer", "")
            
            # 1. 构造符合文档要求的 prompt 
            full_prompt = prompt_template.format(question=question)
            
            # 2. 构造符合文档要求的 response
            # 注意：由于 prompt 以 <think> 结尾，response 直接接推理内容
            full_response = f"{solution} </think> <answer> {answer} </answer>"
            
            # 3. 组合成目标格式 
            sft_item = {
                "prompt": full_prompt,
                "response": full_response
            }
            
            f_out.write(json.dumps(sft_item, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"转换完成！共处理 {processed_count} 条数据。")
    print(f"输出文件路径: {output_path}")

# 执行转换
input_file = "/home/jingqi/CS336_Assignments/assignment5-alignment/data/MATH/origin_validation.jsonl" 
output_file = "/home/jingqi/CS336_Assignments/assignment5-alignment/data/MATH/sft_validation.jsonl"
convert_math_to_sft(input_file, output_file)