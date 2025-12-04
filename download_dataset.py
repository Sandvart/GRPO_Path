import os
import json
from datasets import load_dataset

def download_and_save_dataset(batch_size=10000):
    """
    下载并保存数据集
    
    Args:
        batch_size: 每个JSON文件包含的数据条数，默认10000条
    """
    dataset_name = "rmanluo/RoG-cwq"
    base_output_dir = "datasets/cwq"
    
    # 确保基础输出目录存在
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"创建目录: {base_output_dir}")

    print(f"正在从 Hugging Face 下载数据集: {dataset_name} ...")
    try:
        # 下载并加载数据集
        dataset = load_dataset(dataset_name)
        print("数据集下载成功！")
        
        # 打印数据集信息
        print(f"数据集包含以下 splits: {list(dataset.keys())}")

        # 1. 保存为 Hugging Face 原生格式 (Arrow/Parquet)
        # 这将保存数据集的完整结构，可以视为“原文件”的一种形式
        raw_save_path = os.path.join(base_output_dir, "raw_hf")
        print(f"正在保存原始 Hugging Face 格式到: {raw_save_path}")
        dataset.save_to_disk(raw_save_path)

        # 2. 遍历每个 split 并保存为 JSON 文件（支持大文件拆分）
        print("正在转换并保存为 JSON 文件...")
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            total_count = len(split_data)
            
            print(f"\n处理 {split_name} split (共 {total_count} 条数据)")
            
            # 如果数据量小于batch_size，直接保存为单个文件
            if total_count <= batch_size:
                json_filename = f"{split_name}.json"
                json_file_path = os.path.join(base_output_dir, json_filename)
                
                split_data.to_pandas().to_json(json_file_path, orient="records", lines=False, force_ascii=False, indent=2)
                print(f"  - {split_name} split 已保存到: {json_file_path}")
            else:
                # 数据量大，需要拆分保存
                num_batches = (total_count + batch_size - 1) // batch_size
                print(f"  数据量较大，将拆分为 {num_batches} 个文件（每个 {batch_size} 条）")
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, total_count)
                    
                    # 获取当前批次的数据
                    batch_data = split_data.select(range(start_idx, end_idx))
                    
                    # 生成批次文件名
                    json_filename = f"{split_name}_part{batch_idx + 1:03d}.json"
                    json_file_path = os.path.join(base_output_dir, json_filename)
                    
                    # 保存为 JSON
                    batch_data.to_pandas().to_json(json_file_path, orient="records", lines=False, force_ascii=False, indent=2)
                    print(f"  - 批次 {batch_idx + 1}/{num_batches} 已保存到: {json_filename} (索引 {start_idx}-{end_idx-1})")
            
        print("\n所有操作完成！")
        print(f"文件保存在: {base_output_dir}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 每10000条数据保存为一个JSON文件
    # 可以根据需要调整batch_size参数
    download_and_save_dataset(batch_size=2000)
