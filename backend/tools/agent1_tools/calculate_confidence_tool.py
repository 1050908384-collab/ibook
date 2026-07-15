from typing import List, Dict, Any

class CalculateConfidenceTool:
    def __init__(self):
        pass

    async def calculate(self, user_input: str, skill_tags: List[str], knowledge_gaps: List[Dict[str, Any]]) -> float:
        if not user_input:
            return 0.3
        
        base_confidence = 0.6
        
        if skill_tags and len(skill_tags) >= 1:
            base_confidence += 0.1 * min(len(skill_tags), 3)
        
        if knowledge_gaps and len(knowledge_gaps) >= 1:
            base_confidence += 0.1 * min(len(knowledge_gaps), 3)
        
        return min(base_confidence, 0.95)