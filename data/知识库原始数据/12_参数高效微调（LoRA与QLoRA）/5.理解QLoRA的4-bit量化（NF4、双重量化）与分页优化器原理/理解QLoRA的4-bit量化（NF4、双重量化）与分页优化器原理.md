# QLoRA原理详解：4-bit量化（NF4、双重量化）与分页优化器机制（最终版）

## 参考资料

① 权威：《QLoRA: Efficient Finetuning of Quantized LLMs》
　　网址：https://arxiv.org/abs/2305.14314

② 权威：《Hugging Face GPU Performance Optimization》
　　网址：https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one

③ 权威：《Hugging Face bitsandbytes量化文档》
　　网址：https://huggingface.co/docs/transformers/quantization/bitsandbytes

④ 底稿：《QLoRA原理详解：NF4量化、双重量化与Paged Optimizer》
　　网址：https://blog.csdn.net/weixin_44081096/article/details/163221776

⑤ 参考文章：《QLoRA低显存微调技术详解》
　　网址：https://blog.csdn.net/l35633/article/details/161399952

⑥ 参考文章：《大模型QLoRA量化训练原理分析》
　　网址：https://blog.csdn.net/l35633/article/details/161148157

---

**适合人群**：已经学习 LoRA 原理，并完成 PEFT、bitsandbytes 环境准备，希望理解 QLoRA 为什么可以在消费级GPU上微调大模型的学习者。

本文重点解释：

* QLoRA相比LoRA做了什么优化；
* 为什么使用4-bit量化；
* NF4为什么适合大模型权重；
* 双重量化如何进一步降低显存；
* 分页优化器如何解决显存峰值问题。

---

## 一、为什么需要QLoRA？（①④）
### 1.1 LoRA仍然需要加载完整模型
LoRA虽然减少训练参数：

```text
原模型参数：
冻结

LoRA参数：
训练
```

但是：

基础模型仍然需要加载。

例如：

7B模型：

FP16：

```text
7B × 2 bytes

≈14GB显存
```

如果加入：

* 激活值；
* 梯度；
* 优化器；

实际需求更高。

---

### 1.2 QLoRA核心思想
QLoRA：

Quantized LoRA

即：

> 对基础模型进行低比特量化，再结合LoRA训练。

流程：

```text
FP16基础模型

↓

4-bit量化

↓

加载量化模型

↓

插入LoRA

↓

训练LoRA参数
```

区别：

LoRA：

```text
FP16模型 + LoRA
```

QLoRA：

```text
4-bit模型 + LoRA
```

---

## 二、QLoRA整体结构（①④）
QLoRA训练时：

基础模型：

```text
4-bit权重

冻结
```

LoRA：

```text
FP16/BF16参数

训练
```

结构：

```text
输入

↓

4-bit量化模型

↓

LoRA Adapter

↓

输出
```

重点：

量化的是：

```text
基础模型权重
```

不是：

```text
LoRA参数
```

---

## 三、为什么需要4-bit量化？（①③④）
### 3.1 模型存储方式
模型参数通常：

| 类型        |   每个参数占用 |
| --------- | -------: |
| FP32      |  4 Bytes |
| FP16/BF16 |  2 Bytes |
| INT8      |   1 Byte |
| INT4      | 0.5 Byte |

---

例如：

7B模型：

FP16：

```text
7B × 2

≈14GB
```

4-bit：

```text
7B × 0.5

≈3.5GB
```

显存减少约75%。

> 说明：上面的 `3.5GB`、`减少约75%` 只是**理想权重位数的估算**（7B × 0.5 Byte），仅统计了量化后的权重本身。实际显存占用通常明显更高，因为还需要包含：量化常数/缩放因子、分组元数据（NF4 按组量化，每组都要存额外参数）、未量化的模块（如部分 norm、embedding、lm_head、LoRA 参数），以及推理/训练过程中的计算缓存、激活值与运行时开销。因此请把它当作**理论下界**，真实占用以实际运行为准。

---

### 3.2 为什么不用普通INT4？
传统整数：

INT4：

范围：

```text
-8 ~ 7
```

问题：

神经网络权重通常：

* 接近正态分布；
* 小数较多；
* 分布中心接近0。

直接整数映射：

容易损失精度。

因此QLoRA提出：

NF4。

---

## 四、NF4（NormalFloat4）量化原理（重点）（①④⑤）
### 4.1 NF4是什么？
NF4：

NormalFloat4

中文：

正态分布4位浮点。

QLoRA论文提出：

> 针对神经网络权重正态分布特点设计的4-bit数据类型。

> 说明：这里的"正态分布4位浮点"是一种**通俗叫法**，不要理解成 IEEE 754 那种"通用的 4 位浮点格式"。NF4（NormalFloat4）本质上是**一套针对"近似正态分布"权重设计的 4-bit 量化数据类型/码本（codebook）**：它预先确定了 16 个（2⁴）量化取值点，这些取值点按标准正态分布的分位数排布，使得 0 附近（权重密集区）划分更细。量化时把每组权重归一化后映射到最接近的码本取值，并保存对应的 4-bit 索引；反量化时再按码本与缩放因子还原。它不是可以任意表示实数的"浮点格式"，而是**为特定分布量身定制的离散码本**。

---

### 4.2 为什么适合模型权重？
大模型训练后的权重通常：

近似：

```text
正态分布
```

例如：

```text
大量参数集中在0附近
```

普通INT4：

平均划分：

```text
[-8,-7,...,7]
```

无法很好表示：

中心密集区域。

---

NF4：

根据正态分布：

设计量化范围。

特点：

* 0附近精度更高；
* 更符合权重分布；
* 精度损失更低。

---

### 4.3 NF4量化过程
原始：

```text
FP16权重
```

例如：

```text
0.023

-0.145

0.876
```

经过：

```text
归一化

↓

映射到NF4

↓

保存4-bit值
```

推理时：

再反量化：

```text
NF4

↓

近似FP16
```

---

## 五、双重量化（Double Quantization）（重点）（①④⑥）
### 5.1 为什么需要双重量化？
4-bit量化需要保存：

```text
量化参数
```

例如：

每组权重：

需要：

* scale；
* zero point。

这些额外参数也占显存。

---

### 5.2 双重量化思想
普通量化：

```text
FP16权重

↓

4-bit权重

+

scale参数
```

双重量化：

进一步：

```text
scale参数

↓

再次量化
```

即：

第一次：

量化模型权重。

第二次：

量化量化参数。

---

结构：

```text
FP16权重

↓

NF4量化

↓

保存4-bit权重


scale参数

↓

再次量化

↓

减少scale存储
```

---

### 5.3 双重量化优势
减少：

* 额外量化参数；
* 显存占用。

尤其：

大模型：

参数数量巨大。

少量优化：

也会节省大量显存。

---

## 六、分页优化器（Paged Optimizers）原理（重点）（①④⑤）
### 6.1 为什么需要分页优化器？
训练过程中：

显存峰值来源：

除了模型：

还有：

* 梯度；
* 优化器状态。

例如：

Adam：

需要保存：

```text
一阶动量

二阶动量
```

---

QLoRA虽然降低模型显存：

但是训练过程中：

仍可能：

```text
显存突然不足
```

---

### 6.2 分页优化器思想
Paged Optimizer借鉴：

操作系统虚拟内存。

核心：

> 借助 NVIDIA 的统一内存（Unified Memory），在显存出现峰值、即将 OOM 时，把优化器状态自动"分页"换出到 CPU 内存，需要时再换回 GPU。

流程：

```text
GPU显存

↓

出现峰值（接近/超出显存上限）

↓

优化器状态按页换出到CPU内存（Unified Memory）

↓

需要参与计算时再换回GPU
```

> 说明：不要把它简单理解成"平时就把一部分优化器状态常驻 CPU、需要时交换"。QLoRA 论文里分页优化器的重点是**应对显存峰值**：借助 Unified Memory 的自动分页机制，在显存不足的瞬间把优化器状态临时换出，避免因为偶发峰值直接 OOM 而中断训练；平稳阶段这些状态仍尽量留在 GPU。它解决的是"峰值 OOM"，而不是"常态化地把优化器放到 CPU 以省显存"。

---

### 6.3 类比理解
普通方式：

```text
所有东西放桌面
```

桌面满：

无法继续。

分页优化器：

```text
桌面放常用文件

柜子保存暂不用文件

需要时取出
```

---

### 6.4 如何在训练中实际启用分页优化器？
上面讲的是原理。落到代码上，分页优化器是通过 `TrainingArguments` 的 `optim` 参数选择的（不需要手写换页逻辑，由 bitsandbytes 实现）：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,     # 配合 QLoRA 进一步降低显存峰值
    # 关键：使用 bitsandbytes 的分页优化器
    optim="paged_adamw_8bit",        # 或 "paged_adamw_32bit"（精度更高、显存更大）
)
```

> 说明：
> - `optim="paged_adamw_8bit"` / `"paged_adamw_32bit"` 是 QLoRA 论文里常用的**分页优化器**取值（依赖 bitsandbytes）。前者优化器状态用 8-bit 存储、更省显存；后者精度更高但占用更多。
> - **参数名/可用取值随版本变化**：不同 Transformers / bitsandbytes 版本支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等），请以你所装版本 `TrainingArguments` 文档或 `--help` 为准；若报"未知 optim"错误，换用当前版本支持的名称即可。
> - 分页优化器主要缓解**峰值 OOM**，通常与 `gradient_checkpointing=True`、小 `per_device_train_batch_size` + 梯度累积一起使用以控制显存。

---

## 七、QLoRA三大显存优化技术总结（①）
QLoRA能够低显存训练，依靠三个核心：

### 7.1 4-bit量化
作用：

减少模型权重大小。

```text
FP16

↓

NF4 4-bit
```

---

### 7.2 双重量化
作用：

减少量化参数占用。

```text
scale

↓

再次量化
```

---

### 7.3 分页优化器
作用：

降低训练峰值显存。

```text
GPU

+

CPU内存交换
```

---

## 八、QLoRA与LoRA区别（④⑤）
| 项目   | LoRA      | QLoRA             |
| ---- | --------- | ----------------- |
| 基础模型 | FP16/BF16 | 4-bit（本教程用 NF4）    |
| 训练参数 | LoRA      | LoRA              |
| 量化   | 无         | 4-bit（NF4 为常见选择）   |
| 显存   | 较低        | 通常更低              |
| 依赖   | PEFT      | PEFT+bitsandbytes |
| 适合   | 中高显存GPU   | 显存更受限的场景（含消费级GPU） |

> 说明：上表是**面向本教程配置的对比**，不是绝对定义。(1) QLoRA 的量化类型并非只能是 NF4——bitsandbytes 也支持 `fp4` 等其它 4-bit 类型，NF4 只是 QLoRA 论文推荐、也是本教程采用的**常见选择**。(2) "适合消费级 GPU"是 QLoRA 的**典型收益场景**，但它本质是"用更低显存换取一定精度/速度代价"的方法，并不局限于消费级显卡，服务器级 GPU 上同样可用来在更大模型/更长序列上省显存。请把"NF4""消费级 GPU"理解为常见搭配，而非固定绑定。

---

## 九、bitsandbytes中的QLoRA配置（③）
实际使用：

```python
import torch
from transformers import BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

参数：

### load_in_4bit
开启：

4-bit加载。

---

### bnb_4bit_quant_type
设置：

```python
"nf4"
```

使用：

NF4量化。

---

### bnb_4bit_use_double_quant
开启：

双重量化。

---

### bnb_4bit_compute_dtype
设置：

```python
torch.bfloat16
```

说明：

> 需要区分**存储 dtype** 与**计算 dtype**。权重以 4-bit（NF4）**存储**以节省显存，但在参与矩阵运算时会**反量化**成一个更高精度的 dtype 来计算，这个计算精度由 `bnb_4bit_compute_dtype` 指定。常用 `torch.bfloat16`（较新 GPU 支持较好）或 `torch.float16`；不显式设置时会用默认值（通常为 `float32`），计算会更慢、显存开销也更高。因此 QLoRA 配置里通常会**显式指定** `bnb_4bit_compute_dtype`。具体可用取值与默认值随 bitsandbytes / Transformers 版本变化，请以所用版本文档为准。

> **按 GPU 能力在 BF16 与 FP16 之间选择（不要固定写死 bfloat16）**：
> `torch.bfloat16` 需要 GPU 支持（一般 NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100 等）。较老的 GPU（如部分 Turing/RTX 20 系及更早）**不支持 BF16**，此时应改用 `torch.float16`。可用如下方式**自动判断**：
>
> ```python
> import torch
> from transformers import BitsAndBytesConfig
>
> # 根据 GPU 能力选择计算 dtype：支持 BF16 就用 bfloat16，否则退回 float16
> compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
>
> bnb_config = BitsAndBytesConfig(
>     load_in_4bit=True,
>     bnb_4bit_quant_type="nf4",
>     bnb_4bit_use_double_quant=True,
>     bnb_4bit_compute_dtype=compute_dtype,
> )
> ```
>
> 说明：BF16 与 FP16 都是 16-bit，但**数值范围/精度分布不同**——BF16 动态范围更大、更不易溢出，训练通常更稳定；FP16 精度略高但范围小、更易上溢/下溢。在都支持的较新 GPU 上通常优先 BF16；只有 FP16 可用时用 FP16 并注意数值稳定性（必要时配合 loss scaling / 梯度裁剪）。请以 `torch.cuda.is_bf16_supported()` 的实际返回和你所用版本文档为准。

---

### 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）
上面各段是**分开**讲的：`BitsAndBytesConfig`（怎么 4-bit 加载）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。学习者常见的困惑是"这些拼在一起到底怎么组成一个能跑的 QLoRA 训练"。这里给出一个**最小可运行骨架**，展示它们之间的完整关系（模型/数据请替换为你自己的）：

```python
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "your/base-causal-lm"   # 替换为实际的 Causal LM

# 1) 4-bit 量化配置（按 GPU 能力选择计算 dtype）
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

# 2) 以 4-bit 加载基础模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 3) 关键：为 k-bit 训练做准备（处理 norm/输出层精度、并开启梯度检查点所需的输入梯度等）
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,   # 与 TrainingArguments 的 gradient_checkpointing 对应
)

# 4) 注入 LoRA（只训练低秩增量参数）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],   # 按实际模型模块名调整，或用 "all-linear"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5) 极简数据（真实场景替换为足够规模的数据；Causal LM 的 labels 直接复制 input_ids）
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6) 训练参数：分页优化器 + 梯度检查点
training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,        # 与第 3 步的 use_gradient_checkpointing 呼应
    optim="paged_adamw_8bit",           # 分页优化器（依赖 bitsandbytes；名称随版本可能不同）
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,          # 旧版本用 tokenizer=tokenizer
)

trainer.train()
```

> 关系梳理（这几步缺一不可）：
> 1. **`BitsAndBytesConfig`** 决定基础模型如何 4-bit 加载（NF4 + 双重量化 + 计算 dtype）；
> 2. **`prepare_model_for_kbit_training()`** 是 QLoRA 的**必要一步**：它会对量化模型做适配（如把部分层转成计算精度、开启输入梯度以支持梯度检查点等），**跳过它常导致训练不收敛或报错**；
> 3. **`get_peft_model(..., LoraConfig)`** 把可训练的 LoRA 注入冻结的 4-bit 主干；
> 4. **`gradient_checkpointing=True`**（TrainingArguments）与第 2 步的 `use_gradient_checkpointing=True` **要对应开启**；
> 5. **`optim="paged_adamw_8bit"`** 启用分页优化器缓解峰值显存。
>
> 上例是"能说明关系"的最小骨架；真实训练需替换为足够规模的数据、合理的超参，并按你所用版本核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数与 `processing_class`/`tokenizer` 写法。

---

## 十、电脑配置批注
本章节包含QLoRA运行相关内容。

如果只学习原理：

无需GPU。

---

如果实际运行QLoRA：

最低：

CPU：

* 4核+

内存：

* 16GB+

GPU：

* NVIDIA CUDA GPU

显存：

QLoRA 的设计目标就是在更小的显存上微调大模型，实际可运行的下限取决于模型规模、序列长度与 batch 大小。下面为 7B 级别的**经验参考**（更短序列/更小 batch 时可更低），以实际运行为准：

参考下限：

* 约 8~12GB+

推荐：

* 16GB以上。

---

常见情况（下面为**带条件经验参考**，不是保证值；实际下限随模型规模、序列长度、batch、梯度累积、精度与量化配置变化，请以实际运行为准）：

> 下表隐含的**测试条件**（换条件结论就变）：以 **7B 级别 Causal LM** 为主、**NF4 4-bit + 双重量化**、`bnb_4bit_compute_dtype` 为 BF16/FP16、**较短序列（如 ≤512）**、**小 batch（如 per_device=1）+ 梯度累积**、**开启梯度检查点**、LoRA 目标模块为**少量注意力投影（如 q/v）**。若增大模型/序列/batch、关闭梯度检查点或把 LoRA 加到更多模块，占用都会明显上升。

| 显存    | 适合（经验参考）           |
| ----- | ------------ |
| 8GB   | 小模型 LoRA，或较短序列/小 batch 的 7B QLoRA（较紧张）      |
| 12GB  | 常见可跑 7B QLoRA（序列/ batch 适中时）     |
| 16GB+ | 7B QLoRA 较宽松；13B QLoRA 需配合较短序列/小 batch |
| 24GB+ | 更大模型或更长序列 QLoRA         |

> 说明：上表只给出**大致对应关系**，帮助建立直觉，不能当作"某显存一定能跑某规模"的硬保证。同一张显卡能否跑起来，还取决于序列长度、batch size、梯度累积步数、是否开启梯度检查点、计算 dtype、LoRA 目标模块数量等；显存偏紧时可通过缩短序列、减小 batch、开梯度检查点等方式降低占用。

---

## 十一、总结
完成本章后，你已经理解：

✅ QLoRA为什么比LoRA更省显存
✅ 4-bit量化原理
✅ NF4为什么适合LLM权重
✅ 双重量化如何节省额外空间
✅ 分页优化器如何降低显存峰值
✅ bitsandbytes QLoRA配置方式

QLoRA核心公式：

```text
4-bit基础模型

+

LoRA Adapter

+

显存优化技术

=

低成本大模型微调
```

下一阶段：

**使用QLoRA + PEFT + Trainer完成低显存大模型微调实战**
