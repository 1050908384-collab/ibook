from typing import Dict, Any

class TaskRouter:
    def __init__(self):
        pass

    def route(self, agent_type: str, task: str) -> Dict[str, Any]:
        routing_configs = {
            "diagnostic_agent": {
                "model": "deepseek-r1:7b",
                "temperature": 0.3,
                "max_tokens": 2048
            },
            "stage_planner_agent": {
                "model": "deepseek-r1:7b",
                "temperature": 0.5,
                "max_tokens": 1024
            },
            "content_generator_agent": {
                "model": "deepseek-r1:7b",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "technical_review_agent": {
                "model": "deepseek-r1:7b",
                "temperature": 0.2,
                "max_tokens": 2048
            },
            "quality_review_agent": {
                "model": "deepseek-r1:7b",
                "temperature": 0.4,
                "max_tokens": 2048
            },
            "stage_decision_agent": {
                "model": "deepseek-r1:7b",
                "temperature": 0.3,
                "max_tokens": 1024
            }
        }
        return routing_configs.get(agent_type, routing_configs["content_generator_agent"])
