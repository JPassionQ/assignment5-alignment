import json
import os

def split_dataset(input_file, train_output_file, test_output_file):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
    
    train_count = 0
    test_count = 0
    
    # 同时打开一个读文件和两个写文件
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(train_output_file, 'w', encoding='utf-8') as f_train, \
         open(test_output_file, 'w', encoding='utf-8') as f_test:
        
        for line in f_in:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            try:
                # 解析JSON以检查 unique_id
                data = json.loads(line_stripped)
                unique_id = data.get('unique_id', '')
                
                # 根据 unique_id 的前缀分别写入不同的文件
                # 直接写入原始的 line，以保留原始格式并提高效率
                if unique_id.startswith('train'):
                    f_train.write(line)
                    train_count += 1
                elif unique_id.startswith('test'):
                    f_test.write(line)
                    test_count += 1
                else:
                    print(f"数据跳过：不以 train 或 test 开头 -> {unique_id}")
                    
            except json.JSONDecodeError:
                print("解析JSON行时出错，跳过该行。")
                
    print(f"数据分离完成！")
    print(f"已保存 {train_count} 条数据到: {train_output_file}")
    print(f"已保存 {test_count} 条数据到: {test_output_file}")

if __name__ == '__main__':
    # 文件路径配置
    input_path = '/home/jingqi/CS336_Assignments/assignment5-alignment/data/MATH/train.jsonl'
    train_out_path = '/home/jingqi/CS336_Assignments/assignment5-alignment/data/MATH/origin_train.jsonl'
    test_out_path = '/home/jingqi/CS336_Assignments/assignment5-alignment/data/MATH/origin_test.jsonl'
    
    split_dataset(input_path, train_out_path, test_out_path)