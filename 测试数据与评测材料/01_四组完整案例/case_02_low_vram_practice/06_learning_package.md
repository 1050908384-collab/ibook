# 学习包 - case_02_low_vram_practice（案例B：低显存实操型）

## 环境冻结
- 生成模型：deepseek-chat | Embedding：qwen3-embedding:0.6b | 知识库：kb-2026-08-24

## 阶段1：LoRA与QLoRA（KB-12，中级，适配8GB显存）

> 跳过已掌握的PyTorch/Transformer基础；生成适合8GB显存的QLoRA 4-bit量化方案；包含代码、实验步骤、错误排查。

### 内容块 block_s1_01：QLoRA 4-bit量化配置（NF4 + 双重量化）

```python
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig
import torch

# 8GB显存关键配置：4-bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NF4 量化
    bnb_4bit_use_double_quant=True,       # 双重量化节省显存
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",  # 小模型适配8GB
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
```

### 内容块 block_s1_02：显存估算与 OOM 排查
- 0.5B模型 4-bit：约 0.4GB 权重 + 训练状态约 3GB → 8GB 卡可训练
- 常见 OOM：增大 batch_size 时需配合梯度累积

### 来源说明
- 知识库：12_参数高效微调（LoRA与QLoRA）/5.QLoRA的4-bit量化.md
- 生成模型：deepseek-chat

## 阶段2：PyTorch进阶训练技巧（KB-13，高分推进）
- 梯度累积配合QLoRA微调
