import json
import re
from openai import OpenAI
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Dict

SPARQLPATH = "http://10.201.173.146:3001/sparql"

class FreebaseRetriever:
    """Freebase知识图谱检索器"""
    
    def __init__(self, sparql_endpoint: str):
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
    
    def get_entity_uris(self, entity_name: str, limit: int = 10) -> List[str]:
        """根据实体名称获取所有可能的URI"""
        query = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?entity
            WHERE {{
                ?entity ns:type.object.name "{entity_name}"@en .
            }}
            LIMIT {limit}
        """
        
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
            uris = []
            if "results" in results and "bindings" in results["results"]:
                for binding in results["results"]["bindings"]:
                    uri = binding.get("entity", {}).get("value", "")
                    if uri:
                        uris.append(uri)
            return uris
        except Exception as e:
            print(f"获取实体URI时出错: {e}")
        return []
    
    def normalize_relation(self, relation: str) -> str:
        """标准化关系URI"""
        # 如果关系已经是完整URI，直接返回
        if relation.startswith("http://"):
            return relation
        # 如果关系以ns:开头，替换为完整前缀
        if relation.startswith("ns:"):
            return relation.replace("ns:", "http://rdf.freebase.com/ns/")
        # 否则添加Freebase前缀
        return f"http://rdf.freebase.com/ns/{relation}"
    
    def simplify_relation(self, relation_uri: str) -> str:
        """简化关系URI"""
        if "rdf.freebase.com/ns/" in relation_uri:
            return relation_uri.split("rdf.freebase.com/ns/")[-1]
        return relation_uri
    
    def follow_relation_path_from_uri(self, start_entity_uri: str, start_entity_name: str, 
                                     relation_path: List[str], current_path: List[Dict] = None) -> List[List[Dict]]:
        """
        从指定URI的起始实体出发，沿着关系路径检索，返回所有可能的完整路径
        使用递归来处理一个关系对应多个目标实体的情况
        
        Args:
            start_entity_uri: 当前实体URI
            start_entity_name: 当前实体名称
            relation_path: 剩余的关系序列
            current_path: 当前已构建的路径
        
        Returns:
            所有可能的完整推理路径列表
        """
        if current_path is None:
            current_path = []
        
        # 如果关系路径已经走完，返回当前路径
        if not relation_path:
            return [current_path] if current_path else []
        
        # 取出第一个关系
        relation = relation_path[0]
        remaining_relations = relation_path[1:]
        
        # 标准化关系URI
        normalized_relation = self.normalize_relation(relation)
        
        # 尝试正向查询，获取所有可能的下一个实体
        query_forward = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?nextEntity ?nextEntityName
            WHERE {{
                <{start_entity_uri}> <{normalized_relation}> ?nextEntity .
                ?nextEntity ns:type.object.name ?nextEntityName .
                FILTER (LANG(?nextEntityName) = "en")
            }}
            LIMIT 20
        """
        
        # 尝试反向查询
        query_backward = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?nextEntity ?nextEntityName
            WHERE {{
                ?nextEntity <{normalized_relation}> <{start_entity_uri}> .
                ?nextEntity ns:type.object.name ?nextEntityName .
                FILTER (LANG(?nextEntityName) = "en")
            }}
            LIMIT 20
        """
        
        next_entities = []  # [(uri, name), ...]
        
        # 先尝试正向查询
        self.sparql.setQuery(query_forward)
        try:
            results = self.sparql.query().convert()
            if "results" in results and "bindings" in results["results"]:
                bindings = results["results"]["bindings"]
                for binding in bindings:
                    uri = binding.get("nextEntity", {}).get("value", "")
                    name = binding.get("nextEntityName", {}).get("value", "")
                    if uri and name:
                        next_entities.append((uri, name))
                if next_entities:
                    print(f"      正向查询: {start_entity_name} --[{self.simplify_relation(normalized_relation)}]--> 找到 {len(next_entities)} 个实体")
        except Exception as e:
            print(f"      正向查询出错: {e}")
        
        # 如果正向没找到，尝试反向查询
        if not next_entities:
            self.sparql.setQuery(query_backward)
            try:
                results = self.sparql.query().convert()
                if "results" in results and "bindings" in results["results"]:
                    bindings = results["results"]["bindings"]
                    for binding in bindings:
                        uri = binding.get("nextEntity", {}).get("value", "")
                        name = binding.get("nextEntityName", {}).get("value", "")
                        if uri and name:
                            next_entities.append((uri, name))
                    if next_entities:
                        print(f"      反向查询: 找到 {len(next_entities)} 个实体 --[{self.simplify_relation(normalized_relation)}]--> {start_entity_name}")
            except Exception as e:
                print(f"      反向查询出错: {e}")
        
        # 如果没有找到任何下一个实体，路径断裂
        if not next_entities:
            print(f"      路径断裂: 无法从 {start_entity_name} 通过关系 {self.simplify_relation(normalized_relation)} 找到下一个实体")
            return []
        
        # 对每个可能的下一个实体，递归构建后续路径
        all_complete_paths = []
        for next_uri, next_name in next_entities:
            # 添加当前步骤到路径
            new_step = {
                "subject": start_entity_name,
                "relation": self.simplify_relation(normalized_relation),
                "object": next_name
            }
            new_path = current_path + [new_step]
            
            # 如果还有剩余关系，继续递归
            if remaining_relations:
                sub_paths = self.follow_relation_path_from_uri(
                    next_uri, next_name, remaining_relations, new_path
                )
                all_complete_paths.extend(sub_paths)
            else:
                # 没有剩余关系了，这是一条完整路径
                all_complete_paths.append(new_path)
        
        return all_complete_paths
    
    def follow_relation_path(self, start_entity: str, relation_path: List[str]) -> List[List[Dict]]:
        """
        从起始实体出发，沿着关系路径检索，返回所有可能的完整推理路径
        """
        # 获取起始实体的所有可能URI
        entity_uris = self.get_entity_uris(start_entity)
        if not entity_uris:
            print(f"  ❌ 未找到实体 '{start_entity}' 的URI")
            return []
        
        print(f"  ✓ 找到 {len(entity_uris)} 个可能的URI for '{start_entity}'")
        for i, uri in enumerate(entity_uris[:3]):  # 只显示前3个
            print(f"    URI {i+1}: {uri}")
        
        all_paths = []
        for idx, uri in enumerate(entity_uris):
            print(f"    尝试URI {idx+1}/{len(entity_uris)}: {uri}")
            paths = self.follow_relation_path_from_uri(uri, start_entity, relation_path)
            if paths:
                all_paths.extend(paths)
                print(f"    ✓ 从此URI检索到 {len(paths)} 条完整路径")
        
        return all_paths


def test_single_sample():
    """测试单个样例"""
    
    # 测试样例
    test_sample = {
        "question": "what is the name of justin bieber parents",
        "answer": ["Jeremy Bieber", "Pattie Mallette"],
        "q_entity": ["Justin Bieber"],
        "a_entity": ["Jeremy Bieber", "Pattie Mallette"]
    }
    
    # API配置
    REASONING_API_KEY = ""  # 推理模型的API密钥
    REASONING_BASE_URL = "http://10.201.173.146:8003/v1"
    REASONING_MODEL = "/data/shahy/models/SFT/GRPO_Path_cold_start_3.0"
    
    print("="*80)
    print("测试样例信息:")
    print(f"问题: {test_sample['question']}")
    print(f"标准答案: {test_sample['answer']}")
    print(f"问题实体: {test_sample['q_entity']}")
    print(f"答案实体: {test_sample['a_entity']}")
    print("="*80)
    
    # 初始化检索器
    retriever = FreebaseRetriever(SPARQLPATH)
    
    # 第一步：调用模型生成推理指示器
    print("\n[步骤1] 调用训练后的模型生成推理指示器...")
    
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
    
    try:
        client = OpenAI(api_key=REASONING_API_KEY, base_url=REASONING_BASE_URL)
        completion = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in knowledge graph reasoning."},
                {"role": "user", "content": f"{instruction}\n\nQuestion: {test_sample['question']}"}
            ],
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        
        response = completion.choices[0].message.content
        print(f"\n模型原始响应:\n{response}\n")
        
        # 解析JSON响应
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            reasoning_indicator = json.loads(json_str)
        else:
            print("无法从响应中提取JSON，使用默认值")
            reasoning_indicator = {"question_entities": [], "relation_paths": {}}
    except Exception as e:
        print(f"调用模型出错: {e}")
        print("使用默认推理指示器")
        reasoning_indicator = {
            "question_entities": ["Justin Bieber"],
            "relation_paths": {
                "Justin Bieber": [
                    ["people.person.parents", "people.person.children"],
                    ["people.person.sibling_s", "people.siblings.sibling"],
                ]
            }
        }
    
    print("\n[步骤2] 推理指示器解析结果:")
    print(f"问题实体: {reasoning_indicator['question_entities']}")
    print(f"关系路径: {json.dumps(reasoning_indicator['relation_paths'], indent=2, ensure_ascii=False)}")
    
    # 第三步：检索推理路径
    print("\n" + "="*80)
    print("[步骤3] 从Freebase检索推理路径...")
    
    all_reasoning_paths = []
    
    for entity in reasoning_indicator['question_entities']:
        if entity not in reasoning_indicator['relation_paths']:
            continue
        
        entity_rel_paths = reasoning_indicator['relation_paths'][entity]
        print(f"\n处理实体 '{entity}', 共有 {len(entity_rel_paths)} 条关系路径")
        
        for idx, rel_path in enumerate(entity_rel_paths):
            print(f"\n  关系路径 {idx+1}: {rel_path}")
            
            # 检索完整路径
            full_paths = retriever.follow_relation_path(entity, rel_path)
            
            if full_paths:
                print(f"  ✓ 成功检索到 {len(full_paths)} 条完整推理路径")
                for path in full_paths:
                    all_reasoning_paths.append({
                        "start_entity": entity,
                        "relation_sequence": rel_path,
                        "reasoning_chain": path
                    })
            else:
                print(f"  ❌ 未能检索到完整路径")
    
    # 第四步：显示结果
    print("\n" + "="*80)
    print("[步骤4] 检索结果汇总")
    print(f"总共检索到 {len(all_reasoning_paths)} 条完整推理路径\n")
    
    if all_reasoning_paths:
        for i, path_info in enumerate(all_reasoning_paths, 1):
            print(f"推理路径 {i}:")
            print(f"  起始实体: {path_info['start_entity']}")
            print(f"  关系序列: {path_info['relation_sequence']}")
            print(f"  推理链:")
            for step in path_info['reasoning_chain']:
                print(f"    {step['subject']} --[{step['relation']}]--> {step['object']}")
            
            # 检查是否到达答案实体
            final_entity = path_info['reasoning_chain'][-1]['object']
            if final_entity in test_sample['a_entity']:
                print(f"  ✓ 成功到达答案实体: {final_entity}")
            else:
                print(f"  ⚠ 到达实体 {final_entity}，但不在答案列表中")
            print()
    else:
        print("未检索到任何推理路径")
    
    print("="*80)
    
    # 保存结果
    output_file = "test_single_sample_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_sample": test_sample,
            "reasoning_indicator": reasoning_indicator,
            "retrieved_paths": all_reasoning_paths
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    test_single_sample()
