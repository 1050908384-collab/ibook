from typing import List, Dict, Any
import re

class RuleCheckTool:
    def __init__(self):
        pass

    async def check_rules(self, content: str) -> List[Dict[str, Any]]:
        errors = []
        normalized = "".join(content.lower().split())

        decomposition_patterns = (
            r"(?:原始)?权重(?:矩阵)?w.*?分解.*?u.*?v",
            r"w.*?=.*?u.*?(?:sigma|σ).*?v",
            r"lora.*?(?:svd|奇异值分解)",
        )
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in decomposition_patterns):
            errors.append({
                "error_type": "lora_mechanism_error",
                "severity": "critical",
                "location": "LoRA原理说明",
                "description": "将LoRA描述为对原始权重W执行SVD或训练U/Σ/V，混淆了LoRA与权重矩阵分解。",
                "evidence": "标准LoRA冻结W并训练低秩增量ΔW=(α/r)BA。",
                "suggestion": "改为说明冻结原权重，仅训练低秩矩阵A和B形成增量更新。",
            })

        parameter_match = re.search(
            r"(?:原模型|原始模型|模型).*?(\d+(?:\.\d+)?)万(?:个)?参数.*?(?:微调参数|参数量).*?(\d+(?:\.\d+)?)万",
            normalized,
        )
        if parameter_match:
            original = float(parameter_match.group(1))
            trainable = float(parameter_match.group(2))
            claims_reduction = any(word in normalized for word in ("减少", "降低", "仅为", "只有"))
            if claims_reduction and trainable >= original:
                errors.append({
                    "error_type": "parameter_count_error",
                    "severity": "critical",
                    "location": "参数量示例",
                    "description": f"示例声称参数量下降，但给出的可训练参数{trainable:g}万不小于原参数{original:g}万。",
                    "evidence": "LoRA参数量应按r(d_in+d_out)计算，并与原矩阵d_in*d_out比较。",
                    "suggestion": "重新选择合理维度和秩，并逐步计算参数量。",
                })
        return errors

    async def check_code_rules(self, code: str) -> List[Dict[str, Any]]:
        return []
