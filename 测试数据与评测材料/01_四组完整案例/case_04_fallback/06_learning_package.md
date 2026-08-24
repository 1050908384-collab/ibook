# 学习包 - case_04_fallback（案例D：云端故障自动回退）

## 回退事件
- block_s1_01/02 由 DeepSeek（deepseek-chat）生成
- block_s1_03 生成时 DeepSeek 超时 → 自动回退 Ollama（deepseek-r1:7b）完成
- 审核阶段 DeepSeek 再次超时 → Ollama 接管审核
- **会话、内容块编号、内容版本、审核Trace ID 全程连续，未丢失状态，未重复写入内容**

## 阶段1：LoRA与QLoRA（KB-12，中级，12GB卡）
### block_s1_03（Ollama生成）：LoRA r=8 配置
（回退后由本地模型完成，内容版本连续为 v1）
