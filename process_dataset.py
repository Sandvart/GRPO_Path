import json
import os

def process_dataset():
    input_file = '/Users/sandvart/Desktop/MyCode/GRPO_Path/datasets/train.json'
    output_file = '/Users/sandvart/Desktop/MyCode/GRPO_Path/datasets/train_instruction.json'

    print(f"正在读取文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取 {len(data)} 条数据")
        
        formatted_data = []
        for item in data:
            question = item.get('question', '')
            a_entities = item.get('answer', [])
            
            # 处理 a_entity，将其转换为字符串作为 output
            if isinstance(a_entities, list):
                output_text = ", ".join(a_entities)
            else:
                output_text = str(a_entities)

            # 构造符合 Instruction Tuning 格式的数据
            new_item = {
                "instruction": question,
                "input": "",
                "output": output_text
            }
            formatted_data.append(new_item)

        # 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成！已保存 {len(formatted_data)} 条问答对到: {output_file}")
        
        # 打印前几条作为示例
        print("\n示例数据:")
        for i in range(min(3, len(formatted_data))):
            print(json.dumps(formatted_data[i], ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    process_dataset()
