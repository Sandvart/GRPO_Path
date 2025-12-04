"""
测试GRPO奖励函数
模拟不同的模型输出，验证奖励函数的计算是否正确
"""

import json
from typing import List, Dict
from train_grpo_path import (
    format_reward_func,
    relation_existence_reward_func,
    path_validity_reward_func,
    answer_retrieval_reward_func,
    extract_json_from_response
)

def create_mock_completion(content: str) -> List[List[Dict]]:
    """创建模拟的completion对象"""
    return [[{'content': content}]]


def test_format_reward():
    """测试格式奖励函数"""
    print("="*80)
    print("测试1: 格式奖励函数 (format_reward_func)")
    print("="*80)
    
    test_cases = [
        {
            "name": "完美格式",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"],
                "relation_paths": {
                    "Justin Bieber": [["people.person.parents"]]
                }
            }),
            "expected": 1.0
        },
        {
            "name": "有JSON但缺少字段",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"]
            }),
            "expected": 0.3
        },
        {
            "name": "字段类型错误",
            "response": json.dumps({
                "question_entities": "Justin Bieber",  # 应该是list
                "relation_paths": {
                    "Justin Bieber": [["people.person.parents"]]
                }
            }),
            "expected": 0.5
        },
        {
            "name": "无效JSON",
            "response": "This is not a JSON response",
            "expected": 0.0
        },
        {
            "name": "带额外文本的JSON",
            "response": 'Sure! Here is the result:\n{"question_entities": ["Justin Bieber"], "relation_paths": {"Justin Bieber": [["people.person.parents"]]}}',
            "expected": 1.0
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        completions = create_mock_completion(test["response"])
        rewards = format_reward_func(completions)
        result = "✓" if abs(rewards[0] - test["expected"]) < 0.01 else "✗"
        print(f"\n{result} 测试 {i}: {test['name']}")
        print(f"  预期奖励: {test['expected']}")
        print(f"  实际奖励: {rewards[0]}")
        if result == "✗":
            print(f"  响应内容: {test['response'][:100]}...")


def test_relation_existence_reward():
    """测试关系存在奖励函数"""
    print("\n" + "="*80)
    print("测试2: 关系存在奖励函数 (relation_existence_reward_func)")
    print("="*80)
    
    test_cases = [
        {
            "name": "所有关系都存在",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"],
                "relation_paths": {
                    "Justin Bieber": [
                        ["people.person.parents"],
                        ["people.person.children"]
                    ]
                }
            }),
            "expected_min": 1.5,  # 至少75%的关系存在 (0.75 * 2.0)
            "note": "Freebase中应该存在这些关系"
        },
        {
            "name": "包含不存在的关系",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"],
                "relation_paths": {
                    "Justin Bieber": [
                        ["people.person.parents"],
                        ["fake.relation.does_not_exist"],
                        ["another.fake.relation"]
                    ]
                }
            }),
            "expected_max": 1.0,  # 最多50%存在 (0.5 * 2.0)
            "note": "包含假关系"
        },
        {
            "name": "没有关系",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"],
                "relation_paths": {}
            }),
            "expected": 0.0,
            "note": "空的relation_paths"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        completions = create_mock_completion(test["response"])
        rewards = relation_existence_reward_func(completions)
        
        if "expected" in test:
            result = "✓" if abs(rewards[0] - test["expected"]) < 0.01 else "✗"
            print(f"\n{result} 测试 {i}: {test['name']}")
            print(f"  预期奖励: {test['expected']}")
        elif "expected_min" in test:
            result = "✓" if rewards[0] >= test["expected_min"] else "✗"
            print(f"\n{result} 测试 {i}: {test['name']}")
            print(f"  预期奖励: >= {test['expected_min']}")
        elif "expected_max" in test:
            result = "✓" if rewards[0] <= test["expected_max"] else "✗"
            print(f"\n{result} 测试 {i}: {test['name']}")
            print(f"  预期奖励: <= {test['expected_max']}")
        
        print(f"  实际奖励: {rewards[0]:.4f}")
        print(f"  说明: {test['note']}")


def test_path_validity_reward():
    """测试路径有效性奖励函数"""
    print("\n" + "="*80)
    print("测试3: 路径有效性奖励函数 (path_validity_reward_func)")
    print("="*80)
    print("警告: 此测试需要连接到Freebase，可能较慢...")
    
    test_cases = [
        {
            "name": "正确的关系路径 - 应该能检索到实体",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"],
                "relation_paths": {
                    "Justin Bieber": [
                        ["people.person.parents"]
                    ]
                }
            }),
            "answer": ["Pattie Mallette", "Jeremy Bieber"],
            "expected": 2.0,
            "note": "正确的父母关系"
        },
        {
            "name": "错误的关系路径",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"],
                "relation_paths": {
                    "Justin Bieber": [
                        ["fake.relation.path"]
                    ]
                }
            }),
            "answer": ["Pattie Mallette", "Jeremy Bieber"],
            "expected": 0.0,
            "note": "假关系应该无法检索"
        },
        {
            "name": "实体不存在",
            "response": json.dumps({
                "question_entities": ["NonExistentEntity123456"],
                "relation_paths": {
                    "NonExistentEntity123456": [
                        ["people.person.parents"]
                    ]
                }
            }),
            "answer": ["Someone"],
            "expected": 0.0,
            "note": "实体不存在"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        completions = create_mock_completion(test["response"])
        print(f"\n测试 {i}: {test['name']}")
        print(f"  答案: {test['answer']}")
        print(f"  开始检索...")
        
        rewards = path_validity_reward_func(completions, answer=test["answer"])
        
        result = "✓" if abs(rewards[0] - test["expected"]) < 0.01 else "✗"
        print(f"{result} 预期奖励: {test['expected']}")
        print(f"  实际奖励: {rewards[0]:.4f}")
        print(f"  说明: {test['note']}")
        
        # 解析奖励含义
        if rewards[0] >= 2.0:
            print(f"  结果: 能沿着路径检索到实体 (+2)")
        else:
            print(f"  结果: 路径无效，无法检索到实体")


def test_answer_retrieval_reward():
    """测试答案检索奖励函数"""
    print("\n" + "="*80)
    print("测试4: 答案检索奖励函数 (answer_retrieval_reward_func)")
    print("="*80)
    print("警告: 此测试需要连接到Freebase，可能较慢...")
    
    test_cases = [
        {
            "name": "正确的关系路径 - 应该能检索到答案",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"],
                "relation_paths": {
                    "Justin Bieber": [
                        ["people.person.parents"]
                    ]
                }
            }),
            "answer": ["Pattie Mallette", "Jeremy Bieber"],
            "expected": 3.0,
            "note": "正确的父母关系"
        },
        {
            "name": "错误的关系路径",
            "response": json.dumps({
                "question_entities": ["Justin Bieber"],
                "relation_paths": {
                    "Justin Bieber": [
                        ["fake.relation.path"]
                    ]
                }
            }),
            "answer": ["Pattie Mallette", "Jeremy Bieber"],
            "expected": 0.0,
            "note": "假关系应该无法检索"
        },
        {
            "name": "实体不存在",
            "response": json.dumps({
                "question_entities": ["NonExistentEntity123456"],
                "relation_paths": {
                    "NonExistentEntity123456": [
                        ["people.person.parents"]
                    ]
                }
            }),
            "answer": ["Someone"],
            "expected": 0.0,
            "note": "实体不存在"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        completions = create_mock_completion(test["response"])
        print(f"\n测试 {i}: {test['name']}")
        print(f"  答案: {test['answer']}")
        print(f"  开始检索...")
        
        rewards = answer_retrieval_reward_func(completions, answer=test["answer"])
        
        result = "✓" if abs(rewards[0] - test["expected"]) < 0.01 else "✗"
        print(f"{result} 预期奖励: {test['expected']}")
        print(f"  实际奖励: {rewards[0]:.4f}")
        print(f"  说明: {test['note']}")
        
        # 解析奖励含义
        if rewards[0] >= 3.0:
            print(f"  结果: 成功检索到答案实体 (+3)")
        else:
            print(f"  结果: 未检索到答案实体")


def test_combined_rewards():
    """测试组合奖励"""
    print("\n" + "="*80)
    print("测试5: 组合奖励测试")
    print("="*80)
    
    test_response = json.dumps({
        "question_entities": ["Justin Bieber"],
        "relation_paths": {
            "Justin Bieber": [
                ["people.person.parents"]
            ]
        }
    })
    
    answer = ["Pattie Mallette", "Jeremy Bieber"]
    completions = create_mock_completion(test_response)
    
    print("\n测试响应:")
    print(json.dumps(json.loads(test_response), indent=2))
    print(f"\n标准答案: {answer}")
    
    # 计算各个奖励
    format_reward = format_reward_func(completions)[0]
    relation_reward = relation_existence_reward_func(completions)[0]
    print("\n正在计算检索奖励（可能需要几秒钟）...")
    path_validity_reward = path_validity_reward_func(completions, answer=answer)[0]
    answer_retrieval_reward = answer_retrieval_reward_func(completions, answer=answer)[0]
    
    total_reward = format_reward + relation_reward + path_validity_reward + answer_retrieval_reward
    
    print("\n奖励分解:")
    print(f"  格式奖励:         {format_reward:.4f} / 1.0")
    print(f"  关系存在奖励:     {relation_reward:.4f} / 2.0")
    print(f"  路径有效性奖励:   {path_validity_reward:.4f} / 2.0")
    print(f"  答案检索奖励:     {answer_retrieval_reward:.4f} / 3.0")
    print(f"  总奖励:           {total_reward:.4f} / 8.0")
    
    print("\n奖励评级:")
    if total_reward >= 7.0:
        print("  ★★★★★ 优秀 - 格式正确、关系有效、路径可检索、成功到达答案")
    elif total_reward >= 5.0:
        print("  ★★★★☆ 良好 - 能检索到实体，但可能未到达答案")
    elif total_reward >= 3.0:
        print("  ★★★☆☆ 中等 - 格式或关系有问题")
    else:
        print("  ★☆☆☆☆ 较差 - 多个方面存在问题")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*80)
    print("测试6: 边界情况测试")
    print("="*80)
    
    test_cases = [
        {
            "name": "空响应",
            "response": "",
            "answer": ["Test"]
        },
        {
            "name": "空JSON对象",
            "response": "{}",
            "answer": ["Test"]
        },
        {
            "name": "question_entities为空列表",
            "response": json.dumps({
                "question_entities": [],
                "relation_paths": {}
            }),
            "answer": ["Test"]
        },
        {
            "name": "嵌套的JSON",
            "response": json.dumps({
                "question_entities": ["Entity"],
                "relation_paths": {
                    "Entity": [[["nested", "path"]]]  # 错误的嵌套
                }
            }),
            "answer": ["Test"]
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test['name']}")
        completions = create_mock_completion(test["response"])
        
        try:
            format_reward = format_reward_func(completions)[0]
            relation_reward = relation_existence_reward_func(completions)[0]
            path_validity_reward = path_validity_reward_func(completions, answer=test["answer"])[0]
            answer_retrieval_reward = answer_retrieval_reward_func(completions, answer=test["answer"])[0]
            
            print(f"  ✓ 成功处理")
            print(f"    格式奖励:         {format_reward:.4f}")
            print(f"    关系奖励:         {relation_reward:.4f}")
            print(f"    路径有效性奖励:   {path_validity_reward:.4f}")
            print(f"    答案检索奖励:     {answer_retrieval_reward:.4f}")
        except Exception as e:
            print(f"  ✗ 出错: {e}")


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("GRPO奖励函数测试套件")
    print("="*80)
    
    try:
        # 基础测试（快速）
        test_format_reward()
        test_relation_existence_reward()
        
        # 需要用户确认的慢速测试
        print("\n" + "="*80)
        response = input("\n是否运行检索测试？(需要连接Freebase，可能较慢) [y/N]: ")
        if response.lower() == 'y':
            test_path_validity_reward()
            test_answer_retrieval_reward()
            test_combined_rewards()
        
        # 边界测试
        test_edge_cases()
        
        print("\n" + "="*80)
        print("测试完成！")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
