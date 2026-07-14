from typing import Dict, Any

class StrategySelectorTool:
    def __init__(self):
        pass

    async def select_strategy(self, accuracy: float, profile: Dict[str, Any]) -> str:
        return "normal"
