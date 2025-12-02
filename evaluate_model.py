import json
import random
import os
import re
import string
from openai import OpenAI
from typing import List, Tuple

def load_test_samples(file_path: str, num_samples: int = 10) -> List[dict]:
    """从test.json中随机加载指定数量的样本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机选择样本
    if len(data) > num_samples:
        samples = random.sample(data, num_samples)
    else:
        samples = data
    
    return samples

def query_model(client: OpenAI, question: str) -> str:
    """调用大模型回答问题"""
    try:
        completion = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please answer the question concisely and accurately."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error querying model: {e}")
        return ""

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction: str, answer: List[str]) -> float:
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction: str, answer: List[str]) -> int:
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_f1(prediction: List[str], answer: List[str]) -> Tuple[float, float, float]:
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def evaluate_model():
    """评估模型性能"""
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key="sk-96f7de02e5644f29a4e2192f59ff3e47",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 加载测试样本
    print("正在加载测试样本...")
    samples = load_test_samples('datasets/test.json', num_samples=10)
    print(f"已加载 {len(samples)} 个测试样本\n")
    
    # 评估指标
    total_samples = len(samples)
    hits = 0
    total_f1 = 0.0
    results = []
    
    # 对每个样本进行测试
    for idx, sample in enumerate(samples, 1):
        question = sample['question']
        ground_truth = sample['answer']
        
        print(f"[{idx}/{total_samples}] 问题: {question}")
        print(f"标准答案: {ground_truth}")
        
        # 查询模型
        prediction_str = query_model(client, question)
        print(f"模型回答: {prediction_str}")
        
        # 将预测字符串按行分割成列表
        prediction_list = prediction_str.split("\n") if prediction_str else []
        
        # 计算指标（使用与src_qa_prediction_evaluate_results.py一致的逻辑）
        acc = eval_acc(prediction_str, ground_truth)
        hit = eval_hit(prediction_str, ground_truth)
        f1, precision, recall = eval_f1(prediction_list, ground_truth)
        
        if hit:
            hits += 1
        
        total_f1 += f1
        
        print(f"准确率: {acc:.4f}, 命中: {hit}, F1: {f1:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")
        print("-" * 80)
        
        # 保存结果
        results.append({
            'id': sample.get('id', f'sample_{idx}'),
            'question': question,
            'ground_truth': ground_truth,
            'prediction': prediction_str,
            'prediction_list': prediction_list,
            'accuracy': acc,
            'hit': hit,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
    
    # 计算总体指标
    hit_rate = hits / total_samples
    avg_f1 = total_f1 / total_samples
    
    print("\n" + "=" * 80)
    print("评估结果汇总:")
    print(f"总样本数: {total_samples}")
    print(f"命中数: {hits}")
    print(f"命中率: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"平均F1分数: {avg_f1:.4f}")
    print("=" * 80)
    
    # 保存详细结果
    output_file = 'evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_samples': total_samples,
                'hits': hits,
                'hit_rate': hit_rate,
                'average_f1': avg_f1
            },
            'details': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    # 设置随机种子以确保可复现
    random.seed(42)
    evaluate_model()
