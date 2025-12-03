import json
import random
import hashlib
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple

SPARQLPATH = "http://10.201.173.146:3001/sparql"

class SubgraphExtractor:
    def __init__(self, sparql_endpoint: str):
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        
    def get_random_entities(self, limit: int = 100) -> List[Tuple[str, str]]:
        """随机获取一批实体作为起始点"""
        query = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?entity ?entityName
            WHERE {{
                ?entity ns:type.object.name ?entityName .
                FILTER (LANG(?entityName) = "en")
            }}
            LIMIT {limit}
        """
        
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
            entities = []
            if "results" in results and "bindings" in results["results"]:
                for binding in results["results"]["bindings"]:
                    entity_uri = binding.get("entity", {}).get("value", "")
                    entity_name = binding.get("entityName", {}).get("value", "")
                    if entity_uri and entity_name:
                        entities.append((entity_uri, entity_name))
            return entities
        except Exception as e:
            print(f"获取随机实体时出错: {e}")
            return []
    
    def get_entity_neighbors(self, entity_uri: str) -> List[Dict]:
        """获取实体的所有邻居（包括入边和出边）"""
        # 查询出边
        outgoing_query = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?relation ?neighbor ?neighborName
            WHERE {{
                <{entity_uri}> ?relation ?neighbor .
                ?neighbor ns:type.object.name ?neighborName .
                FILTER (isIRI(?neighbor) && LANG(?neighborName) = "en")
            }}
            LIMIT 50
        """
        
        # 查询入边
        incoming_query = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?relation ?neighbor ?neighborName
            WHERE {{
                ?neighbor ?relation <{entity_uri}> .
                ?neighbor ns:type.object.name ?neighborName .
                FILTER (isIRI(?neighbor) && LANG(?neighborName) = "en")
            }}
            LIMIT 50
        """
        
        neighbors = []
        
        # 处理出边
        self.sparql.setQuery(outgoing_query)
        try:
            results = self.sparql.query().convert()
            if "results" in results and "bindings" in results["results"]:
                for binding in results["results"]["bindings"]:
                    relation = binding.get("relation", {}).get("value", "")
                    neighbor_uri = binding.get("neighbor", {}).get("value", "")
                    neighbor_name = binding.get("neighborName", {}).get("value", "")
                    if relation and neighbor_uri and neighbor_name:
                        neighbors.append({
                            "direction": "outgoing",
                            "relation": relation,
                            "neighbor_uri": neighbor_uri,
                            "neighbor_name": neighbor_name
                        })
        except Exception as e:
            print(f"查询出边时出错: {e}")
        
        # 处理入边
        self.sparql.setQuery(incoming_query)
        try:
            results = self.sparql.query().convert()
            if "results" in results and "bindings" in results["results"]:
                for binding in results["results"]["bindings"]:
                    relation = binding.get("relation", {}).get("value", "")
                    neighbor_uri = binding.get("neighbor", {}).get("value", "")
                    neighbor_name = binding.get("neighborName", {}).get("value", "")
                    if relation and neighbor_uri and neighbor_name:
                        neighbors.append({
                            "direction": "incoming",
                            "relation": relation,
                            "neighbor_uri": neighbor_uri,
                            "neighbor_name": neighbor_name
                        })
        except Exception as e:
            print(f"查询入边时出错: {e}")
        
        return neighbors
    
    def extract_subgraph_bfs(self, start_entity_uri: str, start_entity_name: str, 
                             max_entities: int = 10) -> Dict:
        """使用BFS从起始实体提取子图"""
        visited_entities = {start_entity_uri: start_entity_name}
        triples = []
        queue = deque([start_entity_uri])
        
        while queue and len(visited_entities) < max_entities:
            current_entity = queue.popleft()
            neighbors = self.get_entity_neighbors(current_entity)
            
            # 随机打乱邻居顺序，增加多样性
            random.shuffle(neighbors)
            
            for neighbor_info in neighbors:
                if len(visited_entities) >= max_entities:
                    break
                
                neighbor_uri = neighbor_info["neighbor_uri"]
                neighbor_name = neighbor_info["neighbor_name"]
                relation = neighbor_info["relation"]
                direction = neighbor_info["direction"]
                
                # 记录三元组
                if direction == "outgoing":
                    triples.append({
                        "head_entity": visited_entities[current_entity],
                        "head_entity_uri": current_entity,
                        "relation": relation,
                        "tail_entity": neighbor_name,
                        "tail_entity_uri": neighbor_uri
                    })
                else:  # incoming
                    triples.append({
                        "head_entity": neighbor_name,
                        "head_entity_uri": neighbor_uri,
                        "relation": relation,
                        "tail_entity": visited_entities[current_entity],
                        "tail_entity_uri": current_entity
                    })
                
                # 添加新实体到访问列表
                if neighbor_uri not in visited_entities:
                    visited_entities[neighbor_uri] = neighbor_name
                    queue.append(neighbor_uri)
        
        return {
            "entities": visited_entities,
            "triples": triples
        }
    
    def check_subgraph_constraints(self, triples: List[Dict], entities: Dict) -> bool:
        """检查子图是否满足约束条件"""
        if len(entities) > 10 or len(entities) == 0:
            return False
        
        if len(triples) == 0:
            return False
        
        # 计算每个实体的入度和出度
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for triple in triples:
            head_uri = triple["head_entity_uri"]
            tail_uri = triple["tail_entity_uri"]
            out_degree[head_uri] += 1
            in_degree[tail_uri] += 1
        
        # 统计入度为0和出度为0的节点数量
        zero_in_degree_count = 0
        zero_out_degree_count = 0
        
        for entity_uri in entities.keys():
            if in_degree[entity_uri] == 0:
                zero_in_degree_count += 1
            if out_degree[entity_uri] == 0:
                zero_out_degree_count += 1
        
        # 检查约束：1 <= 入度为0的节点数 <= 4 且 1 <= 出度为0的节点数 <= 4
        if not (1 <= zero_in_degree_count <= 4):
            return False
        if not (1 <= zero_out_degree_count <= 4):
            return False
        
        return True
    
    def simplify_uri(self, uri: str) -> str:
        """简化URI，提取可读部分"""
        if "rdf.freebase.com/ns/" in uri:
            return uri.split("rdf.freebase.com/ns/")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]
        return uri
    
    def generate_subgraph_id(self, triples: List[Dict]) -> str:
        """生成子图的唯一ID"""
        # 使用三元组的哈希值生成ID
        triple_strs = []
        for triple in sorted(triples, key=lambda x: (x["head_entity"], x["relation"], x["tail_entity"])):
            triple_strs.append(f"{triple['head_entity']}_{triple['relation']}_{triple['tail_entity']}")
        content = "_".join(triple_strs)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def format_subgraph(self, subgraph_data: Dict) -> Dict:
        """格式化子图为输出格式"""
        triples = subgraph_data["triples"]
        
        formatted_triples = []
        for triple in triples:
            formatted_triples.append({
                "head_entity": triple["head_entity"],
                "head_entity_type": self.simplify_uri(triple["head_entity_uri"]),
                "relation": self.simplify_uri(triple["relation"]),
                "tail_entity": triple["tail_entity"],
                "tail_entity_type": self.simplify_uri(triple["tail_entity_uri"])
            })
        
        return {
            "subgraph_id": self.generate_subgraph_id(triples),
            "triples": formatted_triples
        }
    
    def extract_n_subgraphs(self, n: int, max_attempts: int = None) -> List[Dict]:
        """提取n个满足条件的子图"""
        if max_attempts is None:
            max_attempts = n * 20  # 默认尝试次数为目标数量的20倍
        
        subgraphs = []
        attempts = 0
        used_subgraph_ids = set()
        
        print(f"开始提取 {n} 个子图...")
        
        while len(subgraphs) < n and attempts < max_attempts:
            attempts += 1
            
            # 获取随机实体
            entities = self.get_random_entities(limit=50)
            if not entities:
                print("无法获取随机实体，跳过...")
                continue
            
            # 随机选择一个实体作为起始点
            start_entity_uri, start_entity_name = random.choice(entities)
            
            print(f"\r尝试 {attempts}/{max_attempts}, 已成功: {len(subgraphs)}/{n}, 当前起始实体: {start_entity_name[:30]}...", end="")
            
            # 提取子图
            try:
                subgraph = self.extract_subgraph_bfs(start_entity_uri, start_entity_name, 
                                                     max_entities=random.randint(3, 10))
                
                # 检查是否满足约束
                if self.check_subgraph_constraints(subgraph["triples"], subgraph["entities"]):
                    formatted_subgraph = self.format_subgraph(subgraph)
                    subgraph_id = formatted_subgraph["subgraph_id"]
                    
                    # 避免重复子图
                    if subgraph_id not in used_subgraph_ids:
                        subgraphs.append(formatted_subgraph)
                        used_subgraph_ids.add(subgraph_id)
                        print(f"\n✓ 成功提取子图 {len(subgraphs)}/{n}, 包含 {len(subgraph['entities'])} 个实体, {len(subgraph['triples'])} 个三元组")
            except Exception as e:
                print(f"\n提取子图时出错: {e}")
                continue
        
        print(f"\n\n完成! 共提取 {len(subgraphs)} 个子图 (尝试了 {attempts} 次)")
        return subgraphs


def main():
    # 参数配置
    n_subgraphs = 10  # 要提取的子图数量
    output_file = "freebase_subgraphs.json"  # 输出文件路径
    
    # 创建提取器
    extractor = SubgraphExtractor(SPARQLPATH)
    
    # 提取子图
    subgraphs = extractor.extract_n_subgraphs(n_subgraphs)
    
    # 保存到文件
    output_data = {"subgraphs": subgraphs}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    print(f"总共提取了 {len(subgraphs)} 个子图")
    
    # 打印统计信息
    if subgraphs:
        total_triples = sum(len(sg["triples"]) for sg in subgraphs)
        print(f"平均每个子图包含 {total_triples / len(subgraphs):.2f} 个三元组")


if __name__ == "__main__":
    main()
