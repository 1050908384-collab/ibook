# 学习包 - case_03_engineer_deployment（案例C：高配置工程部署型）

## 环境冻结
- 生成模型：deepseek-chat | Embedding：qwen3-embedding:0.6b | 知识库：kb-2026-08-24

## 阶段1：LoRA微调与权重合并（KB-12，高级，24GB卡）

> 不重复基础概念；优先LoRA参数设计、权重合并、生产化实训、显存估算。

### 内容块 block_s1_01：LoRA rank 设计与权重合并

```python
# RTX 4090 24GB：r=16 全模块覆盖可行
from peft import LoraConfig
config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
# 训练完成后合并权重
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

### 内容块 block_s1_02：显存估算
- 7B模型 fp16：14GB权重 + LoRA训练状态约 6GB → 24GB卡 r=16 可训练
- 超出时降为 r=8 或启用梯度检查点

## 阶段2：vLLM推理部署（KB-15，高分推进）

### 内容块 block_s2_01：vLLM 加载合并权重并配置 PagedAttention

```bash
# 启动 vLLM OpenAI 兼容服务
python -m vllm.entrypoints.openai.api_server \
    --model ./merged_model \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --port 8000
```

### 内容块 block_s2_02：部署排障
- OOM：降低 gpu-memory-utilization 或 max-model-len
- 启动慢：启用 enforce_eager 或调整 tensor-parallel-size

### 来源说明
- 知识库：15_模型推理部署（vLLM）/5.vLLM加载模型配置.md
- 生成模型：deepseek-chat
