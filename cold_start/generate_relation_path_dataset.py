import json
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple
from tqdm import tqdm

def find_shortest_paths(graph: List[List[str]], q_entities: List[str], a_entities: List[str]) -> Dict[str, List[List[str]]]:
    """
    从图中找到从问题实体到答案实体的所有最短路径
    优化版：使用分层BFS + 路径剪枝
    
    Args:
        graph: 三元组列表 [主体, 关系, 客体]
        q_entities: 问题实体列表
        a_entities: 答案实体列表
    
    Returns:
        字典，键为问题实体，值为该实体到答案实体的所有最短路径（关系序列）
    """
    # 构建图的邻接表
    adjacency = defaultdict(list)  # entity -> [(relation, target_entity), ...]
    
    for triple in graph:
        if len(triple) >= 3:
            subject, relation, obj = triple[0], triple[1], triple[2]
            adjacency[subject].append((relation, obj))
    
    # 预先转换为set，加速查找
    a_entity_set = set(a_entities)
    result = {}
    
    for q_entity in q_entities:
        # 如果问题实体不在图中，跳过
        if q_entity not in adjacency:
            result[q_entity] = []
            continue
        
        # 如果问题实体本身就是答案实体，返回空路径
        if q_entity in a_entity_set:
            result[q_entity] = [[]]
            continue
        
        # 分层BFS：使用字典存储每个节点的所有路径
        current_level = {q_entity: [[]]}  # entity -> list of paths (as relation lists)
        visited = {q_entity}
        found_paths = []
        max_depth = 5  # 限制最大搜索深度，避免过深搜索
        
        for depth in range(max_depth):
            if found_paths:  # 已找到最短路径，停止
                break
            
            next_level = defaultdict(list)
            
            for entity, paths in current_level.items():
                if entity not in adjacency:
                    continue
                
                for relation, neighbor in adjacency[entity]:
                    # 检查是否到达答案实体
                    if neighbor in a_entity_set:
                        for path in paths:
                            found_paths.append(path + [relation])
                    # 未访问过的节点才继续扩展
                    elif neighbor not in visited:
                        for path in paths:
                            next_level[neighbor].append(path + [relation])
            
            # 更新visited集合（包含下一层的所有节点）
            visited.update(next_level.keys())
            
            # 如果没有找到路径且没有下一层，停止搜索
            if not found_paths and not next_level:
                break
            
            current_level = next_level
        
        # 去重路径
        unique_paths = []
        seen = set()
        for path in found_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path)
        
        result[q_entity] = unique_paths
    
    return result

def generate_output(q_entities: List[str], relation_paths: Dict[str, List[List[str]]]) -> str:
    """
    生成输出JSON字符串
    """
    output_dict = {
        "问题实体": q_entities,
        "关系路径": {}
    }
    
    for entity in q_entities:
        if entity in relation_paths and relation_paths[entity]:
            output_dict["关系路径"][entity] = relation_paths[entity]
    
    # 转换为JSON字符串，保持格式
    return json.dumps(output_dict, ensure_ascii=False, indent=2)

def process_dataset(input_file: str, output_file: str):
    """
    处理数据集，生成新的任务数据
    """
    instruction = "请从给定的问题中识别出所有涉及的实体，并为每个实体给出其到答案实体的所有关系路径集合。关系路径指的是从问题实体出发，经过一系列关系，最终到达答案实体的路径，每个问题实体可能对应多条不同的关系路径。\n\n注意：仅输出JSON内容，不要添加多余说明。"
    
    print(f"正在读取数据文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共读取 {len(data)} 条数据")
    
    output_data = []
    
    for idx, item in enumerate(tqdm(data, desc="处理数据", unit="条")):
        question = item.get('question', '')
        q_entities = item.get('q_entity', [])
        a_entities = item.get('a_entity', [])
        graph = item.get('graph', [])
        
        # 找到最短路径
        relation_paths = find_shortest_paths(graph, q_entities, a_entities)
        
        # 生成输出
        output_str = generate_output(q_entities, relation_paths)
        
        # 构造新的数据项
        new_item = {
            "instruction": instruction,
            "input": question,
            "output": output_str
        }
        
        output_data.append(new_item)
    
    # 保存结果
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"完成！共生成 {len(output_data)} 条数据")

if __name__ == "__main__":
    input_file = "/Users/sandvart/Desktop/MyCode/GRPO_Path/datasets/webqsp/train.json"
    output_file = "/Users/sandvart/Desktop/MyCode/GRPO_Path/cold_start/relation_path_dataset.json"
    
    process_dataset(input_file, output_file)
