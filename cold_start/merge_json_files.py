"""
合并cold_start目录下所有分片的JSON文件为一个完整文件
"""

import json
import os
import glob
from tqdm import tqdm

def merge_json_files(input_pattern: str, output_file: str):
    """
    合并多个JSON文件为一个文件
    
    Args:
        input_pattern: 输入文件的通配符模式，例如 "rag_cold_start_dataset_cwq_part*.json"
        output_file: 输出文件路径
    """
    # 获取所有匹配的文件
    input_dir = os.path.dirname(input_pattern) or "."
    pattern = os.path.basename(input_pattern)
    
    # 查找所有匹配的文件并排序
    file_pattern = os.path.join(input_dir, pattern)
    json_files = sorted(glob.glob(file_pattern))
    
    if not json_files:
        print(f"错误: 未找到匹配的文件 {file_pattern}")
        return
    
    print(f"找到 {len(json_files)} 个文件待合并:")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    
    # 合并所有数据
    merged_data = []
    total_count = 0
    
    print("\n开始合并...")
    for json_file in tqdm(json_files, desc="合并文件", unit="个"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                merged_data.extend(data)
                total_count += len(data)
                print(f"  已加载 {os.path.basename(json_file)}: {len(data)} 条数据")
            else:
                print(f"  警告: {json_file} 不是列表格式，跳过")
                
        except Exception as e:
            print(f"  错误: 读取 {json_file} 时出错: {e}")
            continue
    
    # 保存合并后的数据
    print(f"\n正在保存合并后的数据到: {output_file}")
    print(f"总计 {total_count} 条数据")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成！已保存到: {output_file}")
    
    # 显示文件大小
    file_size = os.path.getsize(output_file)
    if file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.2f} KB"
    elif file_size < 1024 * 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"
    else:
        size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"
    
    print(f"文件大小: {size_str}")


if __name__ == "__main__":
    # 设置输入输出路径
    input_pattern = "cold_start/rag_cold_start_dataset_cwq/rag_cold_start_dataset_cwq_part*.json"
    output_file = "cold_start/rag_cold_start_dataset_cwq/rag_cold_start_dataset_cwq_merged.json"
    
    merge_json_files(input_pattern, output_file)
