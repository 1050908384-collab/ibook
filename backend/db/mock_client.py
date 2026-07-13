import json
import os
from typing import List, Dict, Any


class MockDB:
    def __init__(self):
        self.data = self._load_mock_data()
    
    def _load_mock_data(self) -> List[Dict[str, Any]]:
        mock_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mock_data", "mock_knowledge.json")
        mock_path = os.path.abspath(mock_path)
        
        try:
            with open(mock_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("knowledge_base", [])
        except FileNotFoundError:
            return self._generate_default_data()
    
    def _generate_default_data(self) -> List[Dict[str, Any]]:
        return [
            {"id": "1", "title": "PyTorch基础", "content": "PyTorch是一个开源的机器学习框架...", "tags": ["PyTorch", "深度学习"]},
            {"id": "2", "title": "LoRA原理", "content": "LoRA是一种参数高效的微调方法...", "tags": ["LoRA", "微调"]},
            {"id": "3", "title": "分布式训练", "content": "分布式训练将任务分布到多个GPU...", "tags": ["分布式", "训练"]},
            {"id": "4", "title": "模型量化", "content": "模型量化将权重从FP32转换为INT8...", "tags": ["量化", "推理"]},
            {"id": "5", "title": "vLLM部署", "content": "vLLM是一个高效的LLM推理框架...", "tags": ["vLLM", "部署"]}
        ]
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = []
        for item in self.data:
            score = 0
            if query.lower() in item["title"].lower():
                score += 2
            if query.lower() in item["content"].lower():
                score += 1
            if any(query.lower() in tag.lower() for tag in item["tags"]):
                score += 1
            
            if score > 0:
                results.append({"score": score, **item})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]