import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from db.graph_client import GraphClient

def build_knowledge_graph():
    print("Building knowledge graph...")
    
    graph_client = GraphClient()
    
    knowledge_blocks = [
        {"block_id": "KB-001", "block_name": "Python基础", "difficulty": "初级"},
        {"block_id": "KB-002", "block_name": "PyTorch基础", "difficulty": "初级"},
        {"block_id": "KB-003", "block_name": "Transformer架构", "difficulty": "中级"},
        {"block_id": "KB-004", "block_name": "注意力机制", "difficulty": "中级"},
        {"block_id": "KB-005", "block_name": "LoRA微调原理", "difficulty": "中级"},
        {"block_id": "KB-006", "block_name": "QLoRA", "difficulty": "高级"},
        {"block_id": "KB-007", "block_name": "数据预处理", "difficulty": "初级"},
        {"block_id": "KB-008", "block_name": "训练策略", "difficulty": "中级"},
        {"block_id": "KB-009", "block_name": "模型部署", "difficulty": "中级"},
        {"block_id": "KB-010", "block_name": "推理优化", "difficulty": "高级"},
        {"block_id": "KB-011", "block_name": "量化技术", "difficulty": "高级"},
        {"block_id": "KB-012", "block_name": "分布式训练", "difficulty": "高级"},
        {"block_id": "KB-013", "block_name": "评估指标", "difficulty": "初级"},
        {"block_id": "KB-014", "block_name": "超参数调优", "difficulty": "中级"},
        {"block_id": "KB-015", "block_name": "安全与合规", "difficulty": "中级"},
    ]
    
    for block in knowledge_blocks:
        graph_client.add_node(block["block_id"], **block)
    
    dependencies = [
        ("KB-001", "KB-002"),
        ("KB-002", "KB-003"),
        ("KB-003", "KB-004"),
        ("KB-002", "KB-005"),
        ("KB-005", "KB-006"),
        ("KB-001", "KB-007"),
        ("KB-002", "KB-008"),
        ("KB-005", "KB-008"),
        ("KB-008", "KB-009"),
        ("KB-009", "KB-010"),
        ("KB-009", "KB-011"),
        ("KB-008", "KB-012"),
        ("KB-001", "KB-013"),
        ("KB-008", "KB-014"),
        ("KB-009", "KB-015"),
    ]
    
    for from_node, to_node in dependencies:
        graph_client.add_edge(from_node, to_node, relation="prerequisite")
    
    graph_client.save_graph()
    
    print(f"Knowledge graph built with {len(knowledge_blocks)} nodes and {len(dependencies)} edges")
    print("Graph saved successfully!")

if __name__ == "__main__":
    build_knowledge_graph()
