"""
GRPO训练代码 - 关系路径推理指示器任务
基于WebQSP数据集，训练模型生成高质量的推理指示器（问题实体+关系路径）

奖励函数:
1. 格式奖励: 输出符合JSON格式要求
2. 关系存在奖励: 关系在Freebase中存在的比例
3. 路径有效性奖励: 能沿着关系路径检索到任意实体
4. 答案检索奖励: 能检索到答案实体
"""

# watch -n 1 nvidia-smi

# wandb sync 

# nohup python train_grpo.py &

import os
import json
import re
from typing import List, Dict, Tuple
from datasets import Dataset, load_dataset, load_from_disk
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
from importlib.util import find_spec

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

SPARQLPATH = "http://10.201.173.146:3001/sparql"

# ==================== Freebase检索器 ====================

class FreebaseRetriever:
    """简化的Freebase检索器，用于验证关系和检索路径"""
    
    def __init__(self, sparql_endpoint: str):
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(SPARQL_JSON)
        self.relation_cache = {}  # 缓存关系查询结果
    
    def check_relation_exists(self, relation: str) -> bool:
        """检查关系是否在Freebase中存在（使用缓存）"""
        if relation in self.relation_cache:
            return self.relation_cache[relation]
        
        # 标准化关系
        if not relation.startswith("http://"):
            if relation.startswith("ns:"):
                relation = relation.replace("ns:", "http://rdf.freebase.com/ns/")
            else:
                relation = f"http://rdf.freebase.com/ns/{relation}"
        
        # 查询关系是否存在（限制1条即可）
        query = f"""
            SELECT ?s ?o
            WHERE {{
                ?s <{relation}> ?o .
            }}
            LIMIT 1
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            exists = len(results.get("results", {}).get("bindings", [])) > 0
            self.relation_cache[relation] = exists
            return exists
        except Exception:
            self.relation_cache[relation] = False
            return False
    
    def get_entity_uris(self, entity_name: str, limit: int = 10) -> List[str]:
        """根据实体名称获取URI"""
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
        except Exception:
            return []
    
    def normalize_relation(self, relation: str) -> str:
        """标准化关系URI"""
        if relation.startswith("http://"):
            return relation
        if relation.startswith("ns:"):
            return relation.replace("ns:", "http://rdf.freebase.com/ns/")
        return f"http://rdf.freebase.com/ns/{relation}"
    
    def follow_relation_from_uri(self, uri: str, relation: str, limit: int = 20) -> List[str]:
        """从指定URI沿着关系查询下一跳实体"""
        normalized_relation = self.normalize_relation(relation)
        
        # 正向查询
        query_forward = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?nextEntity ?nextEntityName
            WHERE {{
                <{uri}> <{normalized_relation}> ?nextEntity .
                ?nextEntity ns:type.object.name ?nextEntityName .
                FILTER (LANG(?nextEntityName) = "en")
            }}
            LIMIT {limit}
        """
        
        next_entity_names = []
        self.sparql.setQuery(query_forward)
        try:
            results = self.sparql.query().convert()
            if "results" in results and "bindings" in results["results"]:
                for binding in results["results"]["bindings"]:
                    name = binding.get("nextEntityName", {}).get("value", "")
                    if name:
                        next_entity_names.append(name)
        except Exception:
            pass
        
        # 如果正向没找到，尝试反向
        if not next_entity_names:
            query_backward = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?nextEntity ?nextEntityName
                WHERE {{
                    ?nextEntity <{normalized_relation}> <{uri}> .
                    ?nextEntity ns:type.object.name ?nextEntityName .
                    FILTER (LANG(?nextEntityName) = "en")
                }}
                LIMIT {limit}
            """
            
            self.sparql.setQuery(query_backward)
            try:
                results = self.sparql.query().convert()
                if "results" in results and "bindings" in results["results"]:
                    for binding in results["results"]["bindings"]:
                        name = binding.get("nextEntityName", {}).get("value", "")
                        if name:
                            next_entity_names.append(name)
            except Exception:
                pass
        
        return next_entity_names
    
    def can_retrieve_path(self, start_entity: str, relation_path: List[str], 
                         answer_entities: List[str]) -> Tuple[bool, bool]:
        """
        检查是否能沿着关系路径检索到实体
        
        Returns:
            (can_retrieve_any, can_retrieve_answer)
            - can_retrieve_any: 是否能检索到任意实体
            - can_retrieve_answer: 是否能检索到答案实体
        """
        # 标准化答案实体列表（处理可能的嵌套列表）
        normalized_answers = []
        for item in answer_entities:
            if isinstance(item, list):
                normalized_answers.extend(item)
            elif isinstance(item, str):
                normalized_answers.append(item)
        
        # 获取起始实体URI
        entity_uris = self.get_entity_uris(start_entity, limit=5)
        if not entity_uris:
            return False, False
        
        # 对每个URI尝试检索
        for uri in entity_uris:
            current_uris = [uri]
            
            # 沿着关系路径前进
            for relation in relation_path:
                next_entities = []
                # 限制分支数避免组合爆炸，同时保证足够的覆盖率
                # 每个URI最多扩展5条路径，每跳最多返回15个实体
                for current_uri in current_uris[:5]:  # 增加到5个分支
                    entities = self.follow_relation_from_uri(current_uri, relation, limit=15)
                    next_entities.extend(entities)
                
                if not next_entities:
                    break  # 路径断裂
                
                current_uris = next_entities
                
                # 如果是最后一个关系，检查是否到达答案
                if relation == relation_path[-1]:
                    can_retrieve_any = len(current_uris) > 0
                    can_retrieve_answer = any(
                        entity.lower() in [a.lower() for a in normalized_answers]
                        for entity in current_uris
                    )
                    return can_retrieve_any, can_retrieve_answer
        
        return False, False


# ==================== 数据处理 ====================

INSTRUCTION = """Given a question, identify the key entities that can serve as reasoning starting points to answer the question, and provide all relation paths from each question entity to the answer entities.

Task Requirements:
1. Identify question entities: Extract entities from the question that are helpful for answering it and can serve as reasoning starting points.
2. Use canonical entity names: Output the standardized/canonical names of entities, not their surface forms as they appear in the question.
3. Find relation paths: For each question entity, find ALL shortest relation paths that lead to any answer entity.
4. Path structure: A relation path is a sequence of relations connecting a question entity to an answer entity through the knowledge graph.

Output Format:
- Return ONLY a JSON object with no additional explanation
- JSON structure: {"question_entities": [list of canonical entity names], "relation_paths": {entity: [[path1], [path2], ...]}}
- Each path is a list of relation names in sequential order from question entity to answer entity

Example Output:
{
  "question_entities": ["Entity1", "Entity2"],
  "relation_paths": {
    "Entity1": [["relation1", "relation2"], ["relation3"]],
    "Entity2": [["relation4", "relation5"]]
  }
}

Note: Output JSON only, no additional text."""


def prepare_dataset(data_path: str, split: str = "train", max_samples: int = None) -> Dataset:
    """准备训练数据集"""
    print(f"正在加载数据: {data_path}")
    
    # 尝试多种方式加载数据
    data = None
    
    # 方法1: 标准JSON加载
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ 成功使用标准JSON加载")
    except json.JSONDecodeError as e:
        print(f"✗ 标准JSON加载失败: {e}")
        print("尝试逐行加载JSONL格式...")
        
        # 方法2: 尝试JSONL格式（每行一个JSON对象）
        try:
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            if line_num <= 10:  # 只报告前10个错误
                                print(f"警告: 第{line_num}行解析失败")
                            continue
                    
                    # 如果指定了max_samples且已经加载足够，提前停止
                    if max_samples and len(data) >= max_samples:
                        break
            
            if data:
                print(f"✓ 成功逐行加载 {len(data)} 条数据")
        except Exception as e:
            print(f"✗ 逐行加载也失败: {e}")
            
            # 方法3: 尝试分块加载（只加载部分数据）
            print("尝试分块加载...")
            try:
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    content = f.read(10 * 1024 * 1024)  # 读取前10MB
                    # 尝试找到完整的JSON数组
                    if content.startswith('['):
                        # 找到最后一个完整的对象
                        last_brace = content.rfind('}')
                        if last_brace > 0:
                            content = content[:last_brace+1] + ']'
                            data = json.loads(content)
                            print(f"✓ 成功分块加载 {len(data)} 条数据")
            except Exception as e:
                print(f"✗ 分块加载失败: {e}")
                raise RuntimeError(f"无法加载数据文件 {data_path}，请检查文件格式")
    
    if not data:
        raise RuntimeError(f"数据加载失败: {data_path}")
    
    print(f"原始数据条数: {len(data)}")
    
    if max_samples and len(data) > max_samples:
        print(f"采样 {max_samples} 条数据进行训练")
        data = data[:max_samples]
    
    # 转换为Dataset格式
    processed_data = []
    for idx, item in enumerate(data):
        try:
            question = item.get('question', '')
            answer = item.get('answer', [])
            
            if not question:
                continue
            
            processed_data.append({
                'prompt': [
                    {'role': 'system', 'content': 'You are a helpful assistant specialized in knowledge graph reasoning.'},
                    {'role': 'user', 'content': f"{INSTRUCTION}\n\nQuestion: {question}"}
                ],
                'question': question,
                'answer': answer
            })
        except Exception as e:
            print(f"警告: 处理第{idx}条数据时出错: {e}")
            continue
    
    if not processed_data:
        raise RuntimeError("没有有效的训练数据")
    
    dataset = Dataset.from_list(processed_data)
    print(f"最终数据集大小: {len(dataset)}")
    return dataset


# ==================== 奖励函数 ====================

retriever = FreebaseRetriever(SPARQLPATH)


def extract_json_from_response(text: str) -> Dict | None:
    """从响应中提取JSON"""
    try:
        # 尝试直接解析
        return json.loads(text)
    except:
        pass
    
    try:
        # 尝试提取JSON块
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except:
        pass
    
    return None


def format_reward_func(completions, **kwargs) -> List[float]:
    """
    奖励函数1: 格式奖励
    检查输出是否符合JSON格式，且包含必需的字段
    """
    rewards = []
    for completion in completions:
        content = completion[0]['content']
        parsed = extract_json_from_response(content)
        
        if parsed is None:
            rewards.append(0.0)
            continue
        
        # 检查必需字段
        has_entities = 'question_entities' in parsed
        has_paths = 'relation_paths' in parsed
        
        if not (has_entities and has_paths):
            rewards.append(0.3)  # 有JSON但缺少字段
            continue
        
        # 检查字段类型
        entities_valid = isinstance(parsed['question_entities'], list)
        paths_valid = isinstance(parsed['relation_paths'], dict)
        
        if entities_valid and paths_valid:
            rewards.append(1.0)  # 完美格式
        else:
            rewards.append(0.5)  # 格式基本正确
    
    return rewards


def relation_existence_reward_func(completions, **kwargs) -> List[float]:
    """
    奖励函数2: 关系存在奖励
    检查输出的关系在Freebase中是否存在
    返回存在的关系比例
    """
    rewards = []
    for completion in completions:
        content = completion[0]['content']
        parsed = extract_json_from_response(content)
        
        if parsed is None or 'relation_paths' not in parsed:
            rewards.append(0.0)
            continue
        
        # 收集所有关系
        all_relations = []
        relation_paths = parsed['relation_paths']
        if isinstance(relation_paths, dict):
            for entity, paths in relation_paths.items():
                if isinstance(paths, list):
                    for path in paths:
                        if isinstance(path, list):
                            all_relations.extend(path)
        
        if not all_relations:
            rewards.append(0.0)
            continue
        
        # 检查关系是否存在
        existing_count = sum(1 for rel in all_relations if retriever.check_relation_exists(rel))
        ratio = existing_count / len(all_relations)
        
        rewards.append(ratio * 2.0)  # 最高2分
    
    return rewards


def path_validity_reward_func(completions, answer, **kwargs) -> List[float]:
    """
    奖励函数3: 路径有效性奖励
    检查推理指示器能否沿着关系路径检索到任意实体
    - 能检索到任意实体: +2分
    """
    rewards = []
    for completion in completions:
        content = completion[0]['content']
        parsed = extract_json_from_response(content)
        
        if parsed is None:
            rewards.append(0.0)
            continue
        
        question_entities = parsed.get('question_entities', [])
        relation_paths = parsed.get('relation_paths', {})
        
        if not question_entities or not relation_paths:
            rewards.append(0.0)
            continue
        
        # 尝试检索
        can_retrieve_any = False
        
        for entity in question_entities:
            if entity not in relation_paths:
                continue
            
            entity_paths = relation_paths[entity]
            if not isinstance(entity_paths, list):
                continue
            
            for path in entity_paths:
                if not isinstance(path, list) or len(path) == 0:
                    continue
                
                # 检查能否检索到任意实体
                answer_entities = answer if isinstance(answer, list) else [answer]
                retrieve_any, _ = retriever.can_retrieve_path(
                    entity, path, answer_entities
                )
                
                if retrieve_any:
                    can_retrieve_any = True
                    break
            
            if can_retrieve_any:
                break
        
        # 计算奖励
        reward = 2.0 if can_retrieve_any else 0.0
        rewards.append(reward)
    
    return rewards


def answer_retrieval_reward_func(completions, answer, **kwargs) -> List[float]:
    """
    奖励函数4: 答案检索奖励
    检查推理指示器能否检索到答案实体
    - 能检索到答案实体: +3分
    """
    rewards = []
    for completion in completions:
        content = completion[0]['content']
        parsed = extract_json_from_response(content)
        
        if parsed is None:
            rewards.append(0.0)
            continue
        
        question_entities = parsed.get('question_entities', [])
        relation_paths = parsed.get('relation_paths', {})
        
        if not question_entities or not relation_paths:
            rewards.append(0.0)
            continue
        
        # 尝试检索
        can_retrieve_answer = False
        
        for entity in question_entities:
            if entity not in relation_paths:
                continue
            
            entity_paths = relation_paths[entity]
            if not isinstance(entity_paths, list):
                continue
            
            for path in entity_paths:
                if not isinstance(path, list) or len(path) == 0:
                    continue
                
                # 检查能否检索到答案实体
                answer_entities = answer if isinstance(answer, list) else [answer]
                _, retrieve_answer = retriever.can_retrieve_path(
                    entity, path, answer_entities
                )
                
                if retrieve_answer:
                    can_retrieve_answer = True
                    break
            
            if can_retrieve_answer:
                break
        
        # 计算奖励
        reward = 3.0 if can_retrieve_answer else 0.0
        rewards.append(reward)
    
    return rewards


def correctness_reward_func(completions, answer, question, **kwargs) -> List[float]:
    """
    奖励函数5: 正确性奖励（调试用）
    用于打印样例调试信息
    """
    responses = [completion[0]['content'] for completion in completions]
    
    # 打印第一个样例用于调试
    if responses:
        print('-'*80)
        print(f"Question: {question[0] if isinstance(question, list) else question}")
        print(f"Answer: {answer[0] if isinstance(answer, list) else answer}")
        print(f"Response: {responses[0][:500]}...")  # 只打印前500字符
        print('-'*80)
    
    return [0.0] * len(completions)  # 不额外加分，由其他奖励函数组合


# ==================== 主训练流程 ====================

def main():
    # 配置参数
    model_name = "/data/shahy/models/SFT/GRPO_Path_cold_start_3.0_qwen2.5_1.5b"  # 初步训练过的模型
    data_path = "datasets/webqsp/train.json"
    
    output_dir = "outputs/GRPO_Path_Reasoning_Indicator"
    run_name = "GRPO-Path-Reasoning-v1"
    
    # 准备数据集 - 使用全部数据
    dataset = prepare_dataset(data_path, max_samples=None)
    
    # Wandb配置
    wandb_dir = os.path.join(output_dir, "wandb_logs")
    os.makedirs(wandb_dir, exist_ok=True)
    # os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")
    os.environ["WANDB_DIR"] = wandb_dir
    
    # 训练配置
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=4,  # 根据显存调整
        gradient_accumulation_steps=4,
        num_generations=4,  # 每个prompt生成4个候选
        max_prompt_length=512,
        max_completion_length=512,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False
    )
    
    # LoRA配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    
    # 加载模型
    attn_backend = "flash_attention_2" if find_spec("flash_attn") else "sdpa"
    
    print(f"正在加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_backend,
        device_map=None
    ).to("cuda")
    
    # # 配置生成参数，禁用思考模式
    # if hasattr(model, 'generation_config'):
    #     # model.generation_config.temperature = 0.7
    #     # model.generation_config.top_p = 0.8
    #     # model.generation_config.enable_thinking = False
    #     model.generation_config.chat_template_kwargs = {"enable_thinking": False}
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # # 在tokenizer中也设置chat_template_kwargs
    # if hasattr(tokenizer, 'chat_template_kwargs'):
    #     tokenizer.chat_template_kwargs = {"enable_thinking": False}
    
    # 创建训练器
    print("创建GRPO训练器...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,              # 格式奖励 (最高1分)
            relation_existence_reward_func,  # 关系存在奖励 (最高2分)
            path_validity_reward_func,       # 路径有效性奖励 (最高2分)
            answer_retrieval_reward_func,    # 答案检索奖励 (最高3分)
            correctness_reward_func,         # 调试用
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config
    )
    
    # 开始训练
    print("开始GRPO训练...")
    trainer.train()
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model")
    print(f"保存最终模型到: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("训练完成!")


if __name__ == "__main__":
    main()
