import json
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQLPATH = "http://10.201.173.146:3001/sparql"

def test():
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        # 查询名称为"Jamaica"的实体的所有一跳邻居及关联关系
        sparql_txt = """PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?entity ?entityName ?relation ?neighbor (SAMPLE(?name) AS ?neighborName)
            WHERE {
                ?entity ns:type.object.name "Jamaica"@en .
                ?entity ns:type.object.name ?entityName .
                ?entity ?relation ?neighbor .
                ?neighbor ns:type.object.name ?name .
                FILTER (isIRI(?neighbor) && LANG(?name) = "en" && LANG(?entityName) = "en")
            }
            GROUP BY ?entity ?entityName ?relation ?neighbor
            LIMIT 10
        """
        print("查询语句:")
        print(sparql_txt)
        print("\n执行查询...")
        
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        # 提取并打印三元组
        triples = []
        if "results" in results and "bindings" in results["results"]:
            for binding in results["results"]["bindings"]:
                entity = binding.get("entity", {}).get("value", "N/A")
                entity_name = binding.get("entityName", {}).get("value", "N/A")
                relation = binding.get("relation", {}).get("value", "N/A")
                neighbor = binding.get("neighbor", {}).get("value", "N/A")
                neighbor_name = binding.get("neighborName", {}).get("value", "Unknown")
                triples.append((entity, entity_name, relation, neighbor, neighbor_name))
        
        print(f"\n找到 {len(triples)} 个三元组:")
        print("-" * 100)
        for i, (entity, entity_name, relation, neighbor, neighbor_name) in enumerate(triples, 1):
            print(f"\n三元组 {i}:")
            print(f"  主体: {entity}")
            print(f"  主体名称: {entity_name}")
            print(f"  关系: {relation}")
            print(f"  客体: {neighbor}")
            print(f"  客体名称: {neighbor_name}")
            
    except Exception as e:
        print(f'查询出错: {str(e)}')

test()
