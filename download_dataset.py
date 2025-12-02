import os
from datasets import load_dataset

def download_and_save_dataset():
    dataset_name = "rmanluo/RoG-webqsp"
    base_output_dir = "datasets"
    
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

        # 2. 遍历每个 split 并保存为 JSON 文件
        print("正在转换并保存为 JSON 文件...")
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            
            # 构建 JSON 文件路径
            json_filename = f"{split_name}.json"
            json_file_path = os.path.join(base_output_dir, json_filename)
            
            # 保存为 JSON
            # indent=2 使得 JSON 文件易于阅读
            # 使用 to_pandas() 避免 datasets.to_json 的格式化问题
            split_data.to_pandas().to_json(json_file_path, orient="records", lines=False, force_ascii=False, indent=2)
            print(f"  - {split_name} split 已保存到: {json_file_path}")
            
        print("\n所有操作完成！")
        print(f"文件保存在: {base_output_dir}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    download_and_save_dataset()
