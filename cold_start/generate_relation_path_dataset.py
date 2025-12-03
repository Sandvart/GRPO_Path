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
        "question_entities": q_entities,
        "relation_paths": {}
    }
    
    for entity in q_entities:
        if entity in relation_paths and relation_paths[entity]:
            output_dict["relation_paths"][entity] = relation_paths[entity]
    
    # 转换为JSON字符串，保持格式
    return json.dumps(output_dict, ensure_ascii=False, indent=2)

def process_dataset(input_file: str, output_file: str):
    """
    处理数据集，生成新的任务数据
    """
    instruction = """Given a question, identify the key entities that can serve as reasoning starting points to answer the question, and provide all relation paths from each question entity to the answer entities.

Task Requirements:
1. Identify question entities: Extract entities from the question that are helpful for answering it and can serve as reasoning starting points. These are NOT all entities mentioned in the question, but only those relevant to finding the answer.
2. Use canonical entity names: Output the standardized/canonical names of entities, not their surface forms as they appear in the question.
3. Find relation paths: For each question entity, find ALL shortest relation paths that lead to any answer entity.
4. Path structure: A relation path is a sequence of relations connecting a question entity to an answer entity through the knowledge graph.
5. Multiple paths: Each question entity may have multiple different relation paths, and there can be multiple question entities.

Output Format:
- Return ONLY a JSON object with no additional explanation
- JSON structure: {"question_entities": [list of canonical entity names], "relation_paths": {entity: [[path1], [path2], ...]}}
- Each path is a list of relation names in sequential order from question entity to answer entity
- Use canonical entity names (standardized forms) in the output, not surface forms from the question

Example Output:
{
  "question_entities": ["Entity1", "Entity2"],
  "relation_paths": {
    "Entity1": [["relation1", "relation2"], ["relation3"]],
    "Entity2": [["relation4", "relation5", "relation6"]]
  }
}

Note: Output JSON only, no additional text."""
    
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
    output_file = "/Users/sandvart/Desktop/MyCode/GRPO_Path/cold_start/rag_cold_start_dataset.json"
    
    process_dataset(input_file, output_file)
