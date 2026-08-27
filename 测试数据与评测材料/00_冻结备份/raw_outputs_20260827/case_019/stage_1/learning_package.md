# 学习包

## 块 1

## 一、为什么需要QLoRA？

你已经学过LoRA，知道它只训练一小部分参数，大幅减少了**可训练参数量**。但这里有个关键点：**LoRA只是减少了要训练的参数，并没有减少你要加载的模型大小**。

### 动手算一笔账

假设你要微调一个7B模型（70亿参数），用FP16精度存储：

```
7B × 2 bytes ≈ 14GB 显存
```

这14GB只是**把模型加载进显存**的固定开销。真正跑起来时，还需要额外空间给：

- **激活值**：前向传播时每层的中间结果
- **梯度**：反向传播时计算出来的更新方向
- **优化器状态**：比如Adam的动量项

所以实际显存需求远高于14GB。这就是为什么很多人用LoRA时，发现显存还是不够——**瓶颈不在训练参数，而在基础模型本身**。

### 核心思路

QLoRA的出发点很直接：**既然LoRA省的是训练参数，那能不能把基础模型也压缩一下？** 答案是：把基础模型从FP16（16位）降到4-bit量化存储，这样7B模型的显存

## 块 2

**QLoRA核心思想（动手版）**

QLoRA = **Quantized LoRA**，一句话：先把基础模型压成4-bit，再在上面挂LoRA训练。

**流程（跟着做一遍）：**
1. 加载FP16基础模型
2. 转成4-bit量化
3. 加载量化后的模型
4. 插入LoRA适配器
5. 只训练LoRA参数

**对比记忆：**
- LoRA：FP16模型 + LoRA
- QLoRA：4-bit模型 + LoRA

**关键区别：**
- 被量化的是**基础模型权重**（冻结不动）
- LoRA参数保持FP16/BF16，**正常训练**

**结构流：** 输入

## 块 3

## 3.1 模型存储方式：从“斤两”看显存

模型参数在显存里占多大地方，取决于**每个参数用多少字节**。看这张“价目表”：

| 类型 | 每个参数占用 |
|------|------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| INT4 | 0.5 Byte |

**动手算一下（7B模型）：**

- **FP16**：7B × 2 ≈ **14GB**
- **4-bit**：7B × 0.5 ≈ **3.5GB**

显存直接减少约**75%**——这就是4-bit量化最直接的吸引力。

> ⚠️ **注意**：这个3.5GB只是**理论下界**（理想权重估算）。实际占用会更高，因为还要算上量化常数/缩放因子、分组元数据（NF4按组量化，每组都要存

## 块 4

### NF4是什么？——给“动手派”的快速解释

NF4，全称 NormalFloat4，中文叫“正态分布4位浮点”。这是 QLoRA 论文里为**神经网络权重**专门设计的一种 4-bit 数据类型。

**别被名字骗了**：它不是 IEEE 754 那种通用浮点格式，而是一套**离散码本（codebook）**。核心思路是：

- 预先定好 **16 个取值点**（因为 4-bit 能表示 2⁴ = 16 个值）。
- 这些点按**标准正态分布的分位数**排布——权重在 0 附近最密集，所以 0 附近划分更细，精度更高。
-

## 块 5

## 4.2 为什么NF4适合模型权重？

大模型训练完的权重有个特点：**近似正态分布**——大量参数都挤在0附近，越往两边越少。

普通INT4量化是**平均划分**区间，比如[-8,-7,...,7]，每个档位等宽。但权重都集中在中心，这种均匀切法对密集区域“照顾不周”，精度损失大。

**NF4的思路**：既然权重服从正态分布，那就**按正态分布来设计量化范围**——中心区域分得更细，精度更高；两侧分得粗一些。结果就是：更贴合真实权重分布，精度损失更低。

## 4.3 NF4量化过程（实操视角）

你手上有FP16权重，比如 `0.023`

## 块 6

**本块学习检查**：读完5.1-5.3后，请用一句话回答：双重量化“第二次量化”的对象是什么？答案就在文中，找出来即可。

---

**5.1 为什么需要双重量化？**

4-bit量化不只是存权重，还得存每组权重的**量化参数**：`scale`（缩放因子）和`zero point`（零点）。这些额外参数同样占显存，不能忽略。

**5.2 双重量化思想**

普通量化：FP16权重 → 4-bit权重 + scale参数。

双重量化：把scale参数**再量化一次**。第一次量化模型权重，第二次量化量化参数本身。结构上就是：FP16权重 → NF4量化 → 保存4-bit权重；同时scale参数 → 再次量化 → 减少scale

## 块 7

## 6.1 为什么需要分页优化器？

训练大模型时，显存峰值不只看模型本身，**梯度**和**优化器状态**也会占显存。以 Adam 优化器为例，它需要额外保存**一阶动量**和**二阶动量**两份状态。

QLoRA 虽然大幅降低了模型的显存占用，但训练过程中仍可能遇到**显存突然不足（OOM）** 的情况——尤其是某个瞬间的显存峰值，可能直接让训练中断。

## 6.2 分页优化器思想

Paged Optimizer 借鉴了操作系统里的**虚拟内存**思想，核心机制是：

> 借助 NVIDIA 的**统一内存（Unified Memory）**，在显存出现峰值、即将 OOM 时，把优化器状态自动"分页"换出到 CPU 内存，需要时再换回 GPU。

流程如下：

```text
GPU显

## 块 8

好的，我们开始讲解分页优化器。想象一下你的电脑桌面就是显存，普通优化器会把所有东西都堆在桌面上，桌面满了就没法继续工作，也就是显存溢出（OOM）。

分页优化器的思路很聪明：它把常用的文件（比如当前正在计算的参数状态）放在桌面，把暂时不用的文件（比如历史梯度状态）存到柜子里（也就是CPU内存）。当需要用到柜子里的文件时，再临时取出来。这样桌面就不会被占满，训练就能继续下去。

现在，我们来看看如何在代码里启用它。其实很简单，不需要你手动写换页逻辑，`bitsandbytes` 库已经帮你实现了。你只需要在 `TrainingArguments` 里设置 `optim` 参数就行。

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

这里有几点要注意：
- `optim="paged_adamw_8bit"` 表示优化器状态用8-bit存储，更省显存；`"paged_adamw_32bit"` 精度更高但占用更多。
- 不同版本的 Transformers 或 bitsandbytes 支持的 `optim` 取值可能不同（比如还有 `adamw_bnb_8bit`）。如果报错说未知的优化器，就去查一下你当前版本的文档，换用支持的名称即可。
- 分页优化器主要解决的是显存峰值过高的问题，它通常和 `gradient_checkpointing=True`、较小的 `per_device_train_batch_size` 以及梯度累积配合使用，来共同控制显存占用。

**本块学习检查**：请打开你的代码编辑器，找到你之前配置训练参数的地方，

## 块 9

### 7.1 4-bit量化：给模型“瘦身”

QLoRA的第一步，是把模型权重从FP16（16位浮点）压缩到NF4（4位浮点）。你可以把它想象成把高清照片转成压缩格式——文件变小了，但画面依然可辨。这一步直接减少了模型权重的存储大小，是显存优化的基石。

> **动手理解**：FP16下，一个权重占2字节；NF4下只占0.5字节。同样显存，能塞下的模型参数直接翻4倍。

---

### 7.2 双重量化：连“尺子”也压缩

量化时，每个权重块需要配一个“缩放因子”（scale）来还原数值。这些scale本身也占空间。双重量化就是把这些scale**再量化一次**（例如使用NF4量化scale），进一步减少量化参数的占用。相当于你不仅压缩了照片，连照片的“元数据”也一并压缩了。

---

### 7.3 分页优化器：显存不够，内存来凑

训练时优化器状态（如Adam的动量）很吃显存。分

## 块 10

好的，我们直接进入正题。既然你的目标是QLoRA微调，而且喜欢动手实践，那我们就用“配置一个能跑起来的QLoRA”这个视角，来拆解这四个参数。你可以把下面的代码块当作你的“起点模板”。

> **模型许可证**：本配置示例基于 Llama 2 模型架构，其权重遵循 Llama 2 Community License。若你的应用场景为商用且月活用户超过7亿，需另行获得 Meta 的授权。请在使用前确认你的模型来源与许可条款。

### 核心配置：让模型“瘦身”加载

在`transformers`库中，`BitsAndBytesConfig`是控制模型加载方式的总开关。我们这里要配置的四个参数，都是为了让模型以4-bit的“瘦身”状态加载进显存，从而在有限的硬件上跑起大模型微调。

```python
from transformers import BitsAndBytesConfig
import torch

# 1. 根据你的GPU能力，自动选择计算精度（后面会详细讲）
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 开启4-bit加载
    bnb_4bit_quant_type="nf4",        # 使用NF4量化类型
    bnb_4bit_use_double_quant=True,   # 开启双重量化
    bnb_4bit_compute_dtype=compute_dtype, # 设置计算时的精度
)
```

### 逐个参数，动手理解

**1. `load_in_4bit=True`：这是总开关**
把它设为`True`，就是告诉`transformers`：“我要用4-bit的方式加载模型”。这是QLoRA节省显存的基础，没有这一步，后面都无从谈起。

**2. `bnb_4bit_quant_type="nf4"`：选择“压缩算法”**
4-bit加载不是简单地把数字四舍五入，而是一种量化技术。`"nf4"`（NormalFloat4）是一种专门为神经网络权重设计的量化格式，能在极低精度下尽量保留模型性能。在QLoRA的配置中，这就是标准选项。

**3. `bnb_4bit_use_double_quant=True`：再省一笔显存**
这个参数开启“双重量化”。简单理解，就是量化过程中的一些“辅助参数”本身也再被量化一次，能额外节省

## 块 11

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（4-bit 加载）、分页优化器、计算精度这些零件。现在把它们**拼成一台能跑的机器**——下面是一个最小可运行骨架，展示各部分如何协作。模型和数据路径请替换为你自己的。

### 代码骨架（按步骤拆解）

```python
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "your/base-causal-lm"   # 替换为实际的 Causal LM
# 模型许可证：请确认所选模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：以 Llama-2-7b FP16 为例，加载至少需要14GB显存；4-bit 量化后显存需求显著降低
```

**第 1 步：4-bit 量化配置**

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

这里做了三件事：启用 4-bit 加载、用 NF4 量化类型、开启双重量化。计算精度按 GPU 能力自动选择——支持 bf16 就用 bf16，否则退回 fp16。

**第 2 步：以 4-bit 加载基础模型**

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    raise
```

`quantization_config` 把第 1 步的配置传给模型加载；`device_map="auto"` 让模型自动分配到可用设备。同时补上 pad_token（若缺失则用 eos_token 代替）。模型加载涉及网络下载和文件读取，使用 try-except 捕获异常并记录日志。

**第 3 步：为 k-bit 训练做准备（关键衔接）**

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,   # 与 TrainingArguments 的 gradient_checkpointing 对应
)
```

这一步是**量化模型能正常训练的前提**：它处理 norm 层和输出层的精度问题，并开启梯度检查点所需的输入梯度。注意这里的 `use_gradient_checkpointing=True` 必须与后面 `TrainingArguments` 里的 `gradient_checkpointing=True` 呼应。

**第 4 步：注入 LoRA**

```python
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],   # 按实际模型模块名调整，或用 "all-linear"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

只训练低秩增量参数，原始 4-bit 权重保持冻结。`target_modules` 指定注入哪些模块（可按模型调整，或直接用 `"all-linear"`）。`print_trainable_parameters()` 会显示可训练参数量，帮你确认 LoRA 注入生效。

**第 5 步：极简训练数据**

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()   # Causal LM 的 labels 直接复制 input_ids
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

真实场景请替换为足够规模的数据。关键点：Causal LM 的 `labels` 直接复制 `input_ids`，不需要额外标注。

**第 6 步：训练参数——分页优化器 + 梯度检查点**

```python
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

try:
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, data_collator=data_collator,
        tokenizer=tokenizer,             # 兼容旧版本 transformers；新版本可用 processing_class=tokenizer
    )
    trainer.train()
except Exception as e:
    logger.error(f"训练过程出错: {e}")
    raise
```

注意：`Trainer` 的 `tokenizer` 参数在旧版本 transformers 中有效；新版本推荐使用 `processing_class`。若使用旧版本（如 transformers<4.46），请保留 `tokenizer=tokenizer` 写法。

### 整体关系图

```
BitsAndBytesConfig（4-bit 量化配置）
        ↓
AutoModelForCausalLM.from_pretrained（4-bit 加载）
        ↓
prepare_model_for_kbit_training（精度处理 + 梯度检查点）
        ↓
get_peft_model（注入 LoRA）
        ↓
Trainer（分页优化器 + 梯度检查点 + 训练）
```

> **来源说明**：本块内容依据来源块 LOCAL-KBS-12-04-C011 整理，该来源块已通过人工复核确认其技术细节与代码逻辑的准确性。

## 块 12

## 九、把 QLoRA 的五个关键步骤串起来（可跑通骨架）

你现在已经知道 NF4 量化、双重量化、分页优化器各自是干什么的了。但真正写代码时，这些技术要按顺序配合使用，缺一步都可能出问题。下面这 5 步就是 QLoRA 微调的“最小骨架”，帮你建立整体流程感。

**第 1 步：`BitsAndBytesConfig`——决定模型怎么被 4-bit 加载**

这一步是“入口配置”。它一次性指定三件事：用 NF4 做 4-bit 量化、开启双重量化、设定计算 dtype（比如 BF16/FP16）。也就是说，模型加载时的“底层格式”由它说了算。

**第 2 步：`prepare_model_for_kbit_training()`——QLoRA 的必要适配**

这是**最容易跳过但绝不能跳**的一步。量化后的模型不能直接拿去训练，这个函数会冻结量化模型的参数，并启用输入梯度以支持梯度检查点等训练适配。**跳过它，常见后果是训练不收敛或直接报错**。

**第 3 步：`get_peft_model(..., LoraConfig)`——注入 LoRA**

这一步把可训练的 LoRA 适配器“插”到冻结的 4-bit 主干模型上。主干保持量化冻结，只有 LoRA 参数在更新。

**第 4 步：梯度检查点**

在 `TrainingArguments` 中设置 `gradient_checkpointing=True` 即可启用梯度检查点；若使用 `prepare_model_for_kbit_training`，可同时设置 `use_gradient_checkpointing=True` 以保持一致性，但并非强制。

**第 5 步：`optim="paged_adamw_8bit"`——分页优化器**

这个设置启用分页优化器，通过统一内存管理在 GPU 显存不足时将优化器状态分页到 CPU 内存，从而缓解峰值显存压力。

> **注意**：上面是“能说明关系”的最小骨架。真实训练时，你需要换成足够规模的数据、合理的超参，并且按你实际使用的库版本核对 `optim` 的名称、`prepare_model_for_kbit_training` 的参数，以及 `processing_class`/`tokenizer` 的写法。

---

## 十、你的电脑能跑吗？——配置参考

**只学原理**：不需要 GPU，纯看代码和概念就行。

**实际运行 QLoRA**，最低要求：

- **CPU**：4 核以上
- **内存**：16GB 以上
- **GPU**：NVIDIA CUDA

## 测评题目

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1202-003
在其他条件相同时，增大 LoRA rank 最可能带来什么？

### Q-KB1201-003
LoRA中，`r`（rank）参数的作用是？
