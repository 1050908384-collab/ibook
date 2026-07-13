from typing import Dict, Any, Optional, Tuple
from backend.config import settings
from backend.models.iteration_state import IterationStrategy


def check_termination(iteration_state: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    accuracy = iteration_state.get("accuracy", 0.0)
    coverage = iteration_state.get("coverage", 0.0)
    iteration_round = iteration_state.get("iteration_round", 0)
    debate_round = iteration_state.get("debate_round", 0)
    stagnant_count = iteration_state.get("stagnant_count", 0)
    
    if accuracy >= settings.ACCURACY_THRESHOLD and coverage >= settings.COVERAGE_THRESHOLD:
        return True, "达到目标：正确率>85%且覆盖率>90%", IterationStrategy.TERMINATE
    
    if iteration_round >= settings.MAX_ITERATIONS:
        return True, f"达到最大迭代次数({settings.MAX_ITERATIONS})", IterationStrategy.TERMINATE
    
    if stagnant_count >= 2:
        return True, "连续2轮迭代后指标提升<3%，判定为停滞", IterationStrategy.REINFORCE
    
    if debate_round >= settings.MAX_DEBATE_ROUNDS:
        return True, f"辩论次数达到上限({settings.MAX_DEBATE_ROUNDS})，触发熔断", IterationStrategy.TERMINATE
    
    return False, None, None