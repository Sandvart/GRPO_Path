import json
import random
import re
import string
from openai import OpenAI
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

SPARQLPATH = "http://10.201.173.146:3001/sparql"

class FreebaseRetriever:
    """Freebase知识图谱检索器"""
    
    def __init__(self, sparql_endpoint: str):
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
    
    def get_entity_uris(self, entity_name: str, limit: int = 50) -> List[str]:
        """根据实体名称获取所有可能的URI
        
        Args:
            entity_name: 实体的规范名称
            limit: 返回URI的最大数量，默认50（增加以提高召回率）
        
        Returns:
            实体URI列表
        """
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
            LIMIT 100
        """
        
        # 尝试反向查询
        query_backward = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?nextEntity ?nextEntityName
            WHERE {{
                ?nextEntity <{normalized_relation}> <{start_entity_uri}> .
                ?nextEntity ns:type.object.name ?nextEntityName .
                FILTER (LANG(?nextEntityName) = "en")
            }}
            LIMIT 100
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
        
        Args:
            start_entity: 起始实体名称
            relation_path: 关系序列
        
        Returns:
            所有可能的完整推理路径列表
        """
        # 获取起始实体的所有可能URI
        entity_uris = self.get_entity_uris(start_entity)
        if not entity_uris:
            print(f"  未找到实体 '{start_entity}' 的URI")
            return []
        
        print(f"  ✓ 找到 {len(entity_uris)} 个可能的URI for '{start_entity}'")
        for i, uri in enumerate(entity_uris[:50]):  # 只显示前50个
            print(f"    URI {i+1}: {uri}")
        
        all_paths = []
        for idx, uri in enumerate(entity_uris):
            print(f"    尝试URI {idx+1}/{len(entity_uris)}")
            paths = self.follow_relation_path_from_uri(uri, start_entity, relation_path)
            if paths:
                all_paths.extend(paths)
                print(f"    ✓ 从此URI检索到 {len(paths)} 条完整路径")
        
        return all_paths
    
    def simplify_relation(self, relation_uri: str) -> str:
        """简化关系URI"""
        if "rdf.freebase.com/ns/" in relation_uri:
            return relation_uri.split("rdf.freebase.com/ns/")[-1]
        return relation_uri
    
    def retrieve_reasoning_paths(self, question_entities: List[str], 
                                 relation_paths: Dict[str, List[List[str]]]) -> List[Dict]:
        """
        为所有问题实体检索完整的推理路径
        
        Args:
            question_entities: 问题实体列表
            relation_paths: 每个实体对应的关系路径列表
        
        Returns:
            所有完整的推理路径
        """
        all_reasoning_paths = []
        
        for entity in question_entities:
            if entity not in relation_paths:
                print(f"实体 '{entity}' 不在关系路径字典中")
                continue
            
            entity_rel_paths = relation_paths[entity]
            if not entity_rel_paths:
                print(f"实体 '{entity}' 没有关系路径")
                continue
            
            print(f"\n处理实体 '{entity}', 共有 {len(entity_rel_paths)} 条关系路径")
            
            for idx, rel_path in enumerate(entity_rel_paths):
                print(f"  关系路径 {idx+1}: {rel_path}")
                
                # 检索完整路径（可能返回多条）
                full_paths = self.follow_relation_path(entity, rel_path)
                
                if full_paths:
                    print(f"    成功检索到 {len(full_paths)} 条完整推理路径")
                    # full_paths 现在是 List[List[Dict]]，每个元素本身就是一条完整路径
                    for path in full_paths:
                        all_reasoning_paths.append({
                            "start_entity": entity,
                            "relation_sequence": rel_path,
                            "reasoning_chain": path
                        })
                else:
                    print(f"    未能检索到完整路径")
        
        return all_reasoning_paths


class ModelEvaluator:
    """模型评测器"""
    
    def __init__(self, reasoning_api_key: str, reasoning_base_url: str,
                 answer_api_key: str, answer_base_url: str,
                 reasoning_model: str = "qwen-turbo", 
                 answer_model: str = "qwen-turbo"):
        # 推理指示器生成的API客户端
        self.reasoning_client = OpenAI(api_key=reasoning_api_key, base_url=reasoning_base_url)
        # 答案生成的API客户端
        self.answer_client = OpenAI(api_key=answer_api_key, base_url=answer_base_url)
        
        self.reasoning_model = reasoning_model  # 用于生成推理指示器的训练后模型
        self.answer_model = answer_model  # 用于生成最终答案的模型
        self.retriever = FreebaseRetriever(SPARQLPATH)
    
    def generate_reasoning_indicator(self, question: str, instruction: str) -> Dict:
        """
        第一阶段：调用训练后的模型生成推理指示器
        
        Args:
            question: 输入问题
            instruction: 任务指令
        
        Returns:
            包含问题实体和关系路径的字典
        """
        try:
            completion = self.reasoning_client.chat.completions.create(
                model=self.reasoning_model,  # 使用训练后的推理模型
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in knowledge graph reasoning."},
                    {"role": "user", "content": f"{instruction}\n\nQuestion: {question}"}
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
            
            # 解析JSON响应
            # 尝试提取JSON内容
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            else:
                print(f"无法从响应中提取JSON: {response}")
                return {"question_entities": [], "relation_paths": {}}
        
        except Exception as e:
            print(f"生成推理指示器时出错: {e}")
            return {"question_entities": [], "relation_paths": {}}
    
    def generate_final_answer(self, question: str, reasoning_paths: List[Dict]) -> str:
        """
        第二阶段：基于检索到的推理路径生成最终答案
        
        Args:
            question: 输入问题
            reasoning_paths: 从Freebase检索到的完整推理路径
        
        Returns:
            最终答案
        """
        # 构建包含推理路径的提示词
        paths_text = self.format_reasoning_paths(reasoning_paths)
        
        prompt = f"""Please answer the following question by integrating the reasoning paths from the knowledge graph with your own knowledge.

Question: {question}

Reasoning Paths from Knowledge Graph:
{paths_text}

Instructions:
1. If reasoning paths are available, carefully analyze them and extract relevant information
2. Combine the information from reasoning paths with your own knowledge to provide a comprehensive answer
3. Even if no reasoning paths are found or the paths are incomplete, still attempt to answer the question using your knowledge
4. Provide a concise and accurate answer based on all available information
5. If multiple entities satisfy the answer, list them all

Answer:"""
        
        try:
            completion = self.answer_client.chat.completions.create(
                model=self.answer_model,  # 使用答案生成模型
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided knowledge graph reasoning paths."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            
            return completion.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"生成最终答案时出错: {e}")
            return ""
    
    def format_reasoning_paths(self, reasoning_paths: List[Dict]) -> str:
        """格式化推理路径为可读文本"""
        if not reasoning_paths:
            return "No reasoning paths found."
        
        formatted = []
        for i, path_info in enumerate(reasoning_paths, 1):
            start_entity = path_info["start_entity"]
            chain = path_info["reasoning_chain"]
            
            formatted.append(f"\nPath {i} (starting from '{start_entity}'):")
            for step in chain:
                formatted.append(f"  {step['subject']} --[{step['relation']}]--> {step['object']}")
        
        return "\n".join(formatted)
    
    def evaluate(self, test_file: str, num_samples: int = 10, 
                reasoning_instruction: str = None) -> Dict:
        """
        完整的评测流程
        
        Args:
            test_file: 测试数据文件路径
            num_samples: 评测样本数量
            reasoning_instruction: 推理指示器生成的指令
        
        Returns:
            评测结果
        """
        # 加载测试数据
        print(f"正在加载测试数据: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 随机采样
        if len(data) > num_samples:
            samples = random.sample(data, num_samples)
        else:
            samples = data
        
        print(f"已加载 {len(samples)} 个测试样本\n")
        
        # 默认指令
        if reasoning_instruction is None:
            reasoning_instruction =  """Given a question, identify the key entities that can serve as reasoning starting points to answer the question, and provide all relation paths from each question entity to the answer entities.

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
        
        results = []
        hits = 0
        total_f1 = 0.0
        questions_with_paths = 0  # 记录检索到推理路径的问题数量
        questions_reach_answer = 0  # 记录到达答案实体的问题数量
        
        for idx, sample in enumerate(tqdm(samples, desc="评测进度")):
            question = sample['question']
            ground_truth = sample['answer']
            sample_id = sample.get('id', f'sample_{idx}')
            
            print(f"\n{'='*80}")
            print(f"[{idx+1}/{len(samples)}] ID: {sample_id}")
            print(f"问题: {question}")
            print(f"标准答案: {ground_truth}")
            
            # 第一阶段：生成推理指示器
            print("\n[阶段1] 生成推理指示器...")
            reasoning_indicator = self.generate_reasoning_indicator(question, reasoning_instruction)
            print(f"问题实体: {reasoning_indicator.get('question_entities', [])}")
            print(f"关系路径: {json.dumps(reasoning_indicator.get('relation_paths', {}), ensure_ascii=False, indent=2)}")
            
            # 第二阶段：检索推理路径
            print("\n[阶段2] 从Freebase检索推理路径...")
            question_entities = reasoning_indicator.get('question_entities', [])
            relation_paths_dict = reasoning_indicator.get('relation_paths', {})
            
            print(f"问题实体数量: {len(question_entities)}")
            print(f"关系路径字典: {list(relation_paths_dict.keys())}")
            
            retrieved_paths = self.retriever.retrieve_reasoning_paths(
                question_entities, 
                relation_paths_dict
            )
            print(f"\n总共检索到 {len(retrieved_paths)} 条完整推理路径")
            
            # 统计是否检索到推理路径
            if len(retrieved_paths) > 0:
                questions_with_paths += 1
            
            # 检查是否有路径到达答案实体
            reach_answer = False
            if retrieved_paths:
                for path_info in retrieved_paths:
                    if path_info['reasoning_chain']:
                        final_entity = path_info['reasoning_chain'][-1]['object']
                        if final_entity in ground_truth:
                            reach_answer = True
                            break
            
            if reach_answer:
                questions_reach_answer += 1
            
            # 显示检索到的推理路径详情
            if retrieved_paths:
                print("\n检索到的推理路径详情:")
                for i, path_info in enumerate(retrieved_paths[:10], 1):  # 最多显示前10条
                    print(f"\n  推理路径 {i}:")
                    print(f"    起始实体: {path_info['start_entity']}")
                    print(f"    关系序列: {path_info['relation_sequence']}")
                    print(f"    推理链:")
                    for step in path_info['reasoning_chain']:
                        print(f"      {step['subject']} --[{step['relation']}]--> {step['object']}")
                    # 检查最终实体是否在答案中
                    final_entity = path_info['reasoning_chain'][-1]['object']
                    if final_entity in ground_truth:
                        print(f"    ✓ 到达答案实体: {final_entity}")
                    else:
                        print(f"    ⚠ 到达实体 {final_entity}，但不在答案列表中")
                if len(retrieved_paths) > 10:
                    print(f"\n  ... 还有 {len(retrieved_paths) - 10} 条路径未显示")
            
            # 第三阶段：生成最终答案
            print("\n[阶段3] 生成最终答案...")
            final_answer = self.generate_final_answer(question, retrieved_paths)
            print(f"模型答案: {final_answer}")
            
            # 计算评测指标
            hit = self.eval_hit(final_answer, ground_truth)
            f1, precision, recall = self.eval_f1([final_answer], ground_truth)
            
            if hit:
                hits += 1
            total_f1 += f1
            
            print(f"\n评测指标 - 命中: {hit}, F1: {f1:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")
            
            # 保存结果
            results.append({
                'id': sample_id,
                'question': question,
                'ground_truth': ground_truth,
                'reasoning_indicator': reasoning_indicator,
                'retrieved_paths': retrieved_paths,
                'final_answer': final_answer,
                'hit': hit,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
        
        # 计算总体指标
        hit_rate = hits / len(samples) if samples else 0
        avg_f1 = total_f1 / len(samples) if samples else 0
        path_retrieval_rate = questions_with_paths / len(samples) if samples else 0
        answer_reach_rate = questions_reach_answer / len(samples) if samples else 0
        
        print(f"\n{'='*80}")
        print("评测结果汇总:")
        print(f"总样本数: {len(samples)}")
        print(f"检索到推理路径的问题数: {questions_with_paths}")
        print(f"推理路径检索率: {path_retrieval_rate:.4f} ({path_retrieval_rate*100:.2f}%)")
        print(f"到达答案实体的问题数: {questions_reach_answer}")
        print(f"答案实体到达率: {answer_reach_rate:.4f} ({answer_reach_rate*100:.2f}%)")
        print(f"命中数: {hits}")
        print(f"命中率: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
        print(f"平均F1分数: {avg_f1:.4f}")
        print(f"{'='*80}\n")
        
        return {
            'summary': {
                'total_samples': len(samples),
                'questions_with_paths': questions_with_paths,
                'path_retrieval_rate': path_retrieval_rate,
                'questions_reach_answer': questions_reach_answer,
                'answer_reach_rate': answer_reach_rate,
                'hits': hits,
                'hit_rate': hit_rate,
                'average_f1': avg_f1
            },
            'details': results
        }
    
    @staticmethod
    def normalize(s: str) -> str:
        """标准化字符串"""
        s = s.lower()
        exclude = set(string.punctuation)
        s = "".join(char for char in s if char not in exclude)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = " ".join(s.split())
        return s
    
    @staticmethod
    def match(s1: str, s2: str) -> bool:
        """判断s2是否在s1中"""
        s1 = ModelEvaluator.normalize(s1)
        s2 = ModelEvaluator.normalize(s2)
        return s2 in s1
    
    @staticmethod
    def eval_hit(prediction: str, answer: List[str]) -> int:
        """计算命中率"""
        for a in answer:
            if ModelEvaluator.match(prediction, a):
                return 1
        return 0
    
    @staticmethod
    def eval_f1(prediction: List[str], answer: List[str]) -> Tuple[float, float, float]:
        """计算F1分数
        
        Args:
            prediction: 预测答案列表（每个元素是一个预测的答案字符串）
            answer: 标准答案列表
        
        Returns:
            (f1, precision, recall)
        """
        if len(prediction) == 0:
            return 0.0, 0.0, 0.0
        
        # 计算预测中有多少个答案被匹配到
        matched_predictions = 0
        prediction_str = ' '.join(prediction)
        for pred in prediction:
            for a in answer:
                if ModelEvaluator.match(pred, a):
                    matched_predictions += 1
                    break  # 一个预测只计数一次
        
        # 计算标准答案中有多少个被预测覆盖
        matched_answers = 0
        for a in answer:
            if ModelEvaluator.match(prediction_str, a):
                matched_answers += 1
        
        # 精确率 = 预测中正确的数量 / 预测总数
        precision = matched_predictions / len(prediction) if len(prediction) > 0 else 0.0
        # 召回率 = 标准答案中被覆盖的数量 / 标准答案总数
        recall = matched_answers / len(answer) if len(answer) > 0 else 0.0
        
        # F1分数
        if precision + recall == 0:
            return 0.0, precision, recall
        else:
            f1 = 2 * precision * recall / (precision + recall)
            return f1, precision, recall


def main():
    """主函数"""
    # 配置参数 - 推理指示器生成模型的API
    REASONING_API_KEY = ""  # 替换为推理模型的API密钥
    REASONING_BASE_URL = "http://10.201.173.146:8003/v1"  # 替换为推理模型的API地址
    REASONING_MODEL = "/data/shahy/models/SFT/GRPO_Path_cold_start_3.0_1.7b"  # 替换为训练后的推理指示器生成模型名称
    
    # 配置参数 - 答案生成模型的API
    ANSWER_API_KEY = "sk-96f7de02e5644f29a4e2192f59ff3e47"
    ANSWER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ANSWER_MODEL = "qwen-turbo"  # 用于生成最终答案的模型
    
    TEST_FILE = "datasets/webqsp/test.json"
    NUM_SAMPLES = 100
    OUTPUT_FILE = "evaluation_results_with_reasoning.json"
    
    # 设置随机种子
    random.seed(42)
    
    # 创建评测器
    evaluator = ModelEvaluator(
        reasoning_api_key=REASONING_API_KEY,
        reasoning_base_url=REASONING_BASE_URL,
        answer_api_key=ANSWER_API_KEY,
        answer_base_url=ANSWER_BASE_URL,
        reasoning_model=REASONING_MODEL,  # 推理指示器生成模型
        answer_model=ANSWER_MODEL  # 答案生成模型
    )
    
    # 执行评测
    results = evaluator.evaluate(
        test_file=TEST_FILE,
        num_samples=NUM_SAMPLES
    )
    
    # 保存结果
    print(f"正在保存评测结果到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"评测完成！结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
