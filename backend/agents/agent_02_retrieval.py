from backend.agents.base_agent import BaseAgent
from typing import Dict, Any, List


class RetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent_02_retrieval", "知识检索Agent")
        self.available_tools = [
            "vector_search_tool",
            "keyword_search_tool",
            "graph_query_tool",
            "rerank_tool"
        ]
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        learner_profile = input_data.get("learner_profile", {})
        
        retrieved_docs = self._mock_search(query)
        
        prompt = f"""你是一个知识检索专家，负责根据学习者需求检索相关知识。
        
查询内容：{query}
学习者画像：{learner_profile}

检索到的文档：{retrieved_docs}

请整理检索结果，输出：
1. 相关知识点列表
2. 每个知识点的详细解释
3. 知识点之间的关联关系
4. 推荐的学习顺序
"""
        
        response = self.call_llm(prompt)
        
        return {
            "knowledge_points": response,
            "references": retrieved_docs,
            "learning_order": self._generate_learning_order(response)
        }
    
    def _mock_search(self, query: str) -> List[Dict[str, Any]]:
        from backend.db.mock_client import MockDB
        db = MockDB()
        return db.search(query)
    
    def _generate_learning_order(self, knowledge: str) -> list:
        prompt = f"""根据以下知识点，生成推荐的学习顺序：
{knowledge}
"""
        result = self.call_llm(prompt)
        return result.split("\n") if result else []