# 学习包

## 块 1

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

先明确一个关键点：**LoRA训练过程中，基础模型是被冻结的**。真正被训练的是额外的Adapter参数。

训练时的结构是这样的：

```text
基础模型（冻结） + LoRA Adapter（训练）
```

以Llama-7B为例，训练完成后保存的通常**不是完整模型**，而是两个小文件：

```text
adapter_model.safetensors   # LoRA权重（低秩矩阵A和B）
adapter_config.json         # 配置信息
```

其中`adapter_model.safetensors`包含的是低秩增量矩阵A和B。合并时需计算ΔW=(α/r)BA，并将其加回原权重。

### 1.2 Adapter有什么特点？

**优点：**
- 文件小，保存快
- 同一个基础模型可以搭配多个任务Adapter

比如同一个基础模型，可以分别搭配：

```text
客服Adapter / 医疗Adapter / 法律Adapter
```

**缺点：**
部署时**基础模型和Adapter必须同时存在**，缺一不可。

---

## 二、为什么需要合并LoRA权重？

因为部署推理时，如果每次都同时加载基础模型和Adapter，既麻烦又容易出错。**合并**就是把Adapter的权重“写回”基础模型，得到一个完整的、独立的模型文件，之后只需加载这一个模型即可。

```text
# 注意：Llama-2-7b FP16 需要至少14GB显存。
```

---

## 本块学习检查

在继续之前，先做一个最小回顾：**模型保存与加载（checkpoint）**。

请用你熟悉的框架（如PyTorch

## 块 2

**本块学习检查**：读完本块后，请合上资料，用一句话说明“合并权重”前后，模型在推理时的结构分别长什么样。

---

### 推理时为什么要合并？

**不合并时**，模型是“双件套”：

- 输入 → **基础模型** + **LoRA适配器** → 输出

这意味着每次推理都要加载两份东西：基础模型和适配器。

**部署时**，你通常只想要一个“完整模型”：

- 输入 → **完整模型** → 输出

所以，需要把适配器**合并**进基础模型，得到一个单一模型。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：

## 块 3

## 3.1 merge_and_unload() 是干什么的？

一句话：**把LoRA训练得到的低秩增量加到基础模型权重里，并移除PEFT包装，使模型不再依赖外部Adapter。**

PEFT库提供了一个方法：

```python
merge_and_unload()
```

它的作用非常直接：

> 将LoRA权重合并到基础模型，并移除PEFT包装。

用图来理解，执行前你是这样的：

```text
Base Model
+
LoRA Adapter
```

执行后变成：

```text
完整模型
```

也就是说，LoRA那套“旁路”参数被真正写回了基础模型的权重里，模型不再是“基础模型+外挂Adapter”的组合体，而是一个**完整的、独立的模型**。

---

## 3.2 合并前后，模型结构发生了什么变化？

**合并前**，你的模型在内存里长这样：

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

**合并后**，变成了：

```text
普通Transformer模型
包含LoRA更新后的权重
```

关键区别：合并后的模型**不再是PeftModel**，而是一个普通的Transformer模型——LoRA的痕迹已经“融入血液”，不再有单独的Adapter挂件。

---

## 四、LoRA模型合并的完整流程

整体流程就6步，非常清晰：

```text
训练LoRA
    ↓
保存Adapter
    ↓
加载基础模型
    ↓
加载Adapter
    ↓
merge_and_unload()
    ↓
保存完整模型
```

---

## 五、加载LoRA Adapter（动手前必看）

假设你已经有：

```python
base_model_path = "model_path"   # 基础模型路径
adapter_path = "adapter_path"    # LoRA适配器路径
```

加载代码：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 第一步：加载基础模型
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained(
    base_model_path
)

# 第二步：把LoRA Adapter挂上去
model = PeftModel.from_pretrained(
    model,
    adapter_path
)
```

此时模型结构是：

```text
Base Model
+
LoRA Adapter
```

---

## 六、执行合并（核心一步）

代码只有一行：

```python
merged_model = model.merge_and_unload()
```

这一步在数学上做的事：

```text
W + (α/r)BA → W'
```

（W是原始权重，BA是LoRA学到的低秩更新，α是缩放系数，r是秩，W'是合并后的新权重）

执行完后，`merged_model` **不再是PeftModel**，而是：

```text
普通Transformer模型
```

---

## 七、导出合并后的模型

合并完，保存到本地：

```python
merged_model.save_pretrained(
    "./merged_model"
)
```

**别忘了Tokenizer也要一起保存**：

```python
tokenizer.save_pretrained(
    "./merged_model"
)
```

保存后的目录结构（示意）：

```text
merged_model/
├── config

## 块 4

## 三、merge_and_unload 是什么？（重点）

一句话理解：**merge_and_unload 就是把 LoRA 小补丁“焊死”进原模型，得到一个完整的普通模型**。之后加载推理不再需要 PEFT 库，直接用 Transformers 就能跑。

### 合并后加载推理（③④）

合并后，加载方式变得非常简单：

```python
from transformers import AutoModelForCausalLM

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model"
)
```

流程就是：**用户输入 → 完整模型 → 生成结果**，中间不再有 Adapter 这一层。

### 合并 vs 不合并（②③）

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base+Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 PEFT | 直接 Transformers |

### 什么时候应该合并？（③④）

核心原则：**合并是权衡，不是必须**。取决于三点：

- 是否需要**多个 Adapter 动态切换**（需要就不合并）
- 部署框架是否**原生支持 LoRA**（不支持就合并）
- 对**存储体积**和**精度**（尤其 QLoRA）的接受度

### 合并后必查清单（重点！）

导出后建议核查以下四项，避免“能加载但行为不对”：

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中**新增/修改过的特殊 token**（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里；若扩过词表，还要确认 embedding/lm_head 尺寸与词表匹配。
2. **generation

## 块 5

**本块学习检查**：读完下面两个场景后，请用自己的话说一说：什么情况下“合并”更合适？什么情况下“不合并”反而更好？

---

### 10.1 什么时候该合并？

**场景1：上线部署（只用一套Adapter，追求简单）**

想象你的模型要放到生产环境：服务器 → 直接加载模型 → 提供API。

如果只需要**一个固定的Adapter**，并且希望部署链路越简单越好（不额外装PEFT、不做运行时加载），那**合并更方便**——一步到位，省心。

> ⚠️ 但注意：如果你的推理框架本身支持直接加载LoRA（比如vLLM的多LoRA服务），或者你需要**在同一基础模型上热切换多个Adapter**，那**不合并、用Adapter形式部署反而更灵活**。这时合不合并，要看框架能力和业务需求。

> 📌 模型许可证：Llama

## 块 6

## 10.2 不建议合并场景

你是动手实践型学习者，记住一个核心判断：**合并不是必须的，保留Adapter更灵活**。

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了多个Adapter（A、B、C），分别对应不同任务。如果合并，你就把A、B、C全焊死在模型里，想切换任务得重新加载整个模型；保留Adapter，随时按需加载、卸载，灵活切换。

**场景2：持续训练**

如果你打算继续调整LoRA，同样保留Adapter。合并后LoRA参数就“固化”进模型了，没法再单独微调。

**实操建议**：除非你确定模型不再改动、且部署环境不支持加载Adapter，否则优先保留

## 块 7

## 11.1 直接合并可能遇到的情况

你现在做 LoRA 微调，最后一步是把训练好的 Adapter 合并回基础模型。如果你直接在 4-bit 量化状态下执行 `merge_and_unload()`，可能会遇到两类问题：

1. **精度损失**——4-bit 权重需要先反量化再合并，这个过程与全程用高精度计算并不完全等价，合并后的模型可能会有精度上的偏差。
2. **类型或版本不兼容**——可能出现权重类型不匹配，或者你用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

> 注意：上面说的是**可能**出现的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你所用库的版本、量化配置，以及是否按下面的流程先反量化。最终以你所用版本的文档和实际运行结果为准。

## 11.2 常见做法：先反量化到高精度再合并

一个稳妥的思路是：**不要在 4-bit 状态下直接合并**，而是先用 FP16/BF16 高精度重新加载一份基础模型，再挂载同一个 Adapter 后合并。

流程如下：

```
以高精度（FP16/BF16）重新加载基础模型
        ↓
加载同一份 LoRA Adapter
        ↓
merge_and_unload()
        ↓
保存完整模型
```

这样合并发生在高精度权重上，绕开了 4-bit 状态下的类型和精度问题。

下面是完整可执行的代码：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "your/base-causal-lm"   # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的 LoRA Adapter 目录
merged_out = "./merged_model"

# 1) 关键：以高精度（FP16/BF16）重新加载基础模型，而不是再走 4-bit 量化
#    合并要发生在高精度权重上，所以这里不传 quantization_config
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=compute_dtype,
    device_map="auto",        # 显存不足时可改为 device_map={"": "cpu"} 在 CPU 上合并
)

# 2) 挂载训练得到的同一份 Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3) 合并；safe_merge=True 会在合并时做数值检查（如检测 NaN）
merged_model = model.merge_and_unload(safe_merge=True)

# 4) 保存完整模型与 tokenizer（tokenizer 从基础模型或训练目录加载均可，保持与训练一致）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
merged_model.save_pretrained(merged_out)
tokenizer.save_pretrained(merged_out)
```

**代码要点拆解：**

- **第 1 步是关键**：重新加载时只指定 `torch_dtype`，不传 `quantization_config`，这样模型就是高精度权重，合并不会遇到 4-bit 的问题。
- **第 3 步**：`safe_merge=True` 会在合并时做数值检查（如检测 NaN）。
- **模型许可证**：合并后的模型沿用基础模型的许可证。例如，若基础模型为 Llama 2，则适用 Llama 2 Community License（商用月活>7亿需Meta授权）。请在使用

## 块 8

### 动手前先看：QLoRA合并的3个“坑”

你正走在LoRA微调的路上，这里直接给你合并时的关键提醒，帮你少走弯路。

**1. 版本决定“能不能直接合并”**
不同PEFT版本对量化模型合并的支持不一样。有的版本要求先反量化。**最稳妥的做法**：加载基础模型时，**不要传** `quantization_config` 或 `load_in_4bit` 参数，直接用高精度权重来合并。

**2. 内存不够？换CPU**
用高精度重新加载完整模型，会占用约一份FP16/BF16完整模型的内存/显存。如果显存不足，把 `device_map` 改成 `{"": "cpu"

## 块 9

**动手做：合并前先上保险，合并后验一验**

合并模型时，直接调用合并方法即可完成模型合并，无需额外参数。合并完成后，建议使用 `safe_serialization=True` 确保模型以安全格式保存，防止合并过程产生异常值，相当于给合并操作上了道保险：

```python
merged_model = model.merge_and_unload(safe_serialization=True)
```

合并完别急着用，先验证一下合并没有引入明显偏差。操作很简单：准备一小批固定输入（prompt），分别用合并前的 PeftModel 和合并后的模型推理，对比两者的 logits 或生成文本是否基本一致。

**最小验证代码**：用固定 prompt，先比较 logits 的最大绝对差，

## 块 10

## 合并后别急着用——先做一次“前后对照”验证

你已经完成了 `merge_and_unload()` 合并，模型也成功导出了。但这里有个关键问题：**合并后的模型，输出和原来一样吗？** 尤其是 QLoRA 场景下，因为涉及量化精度损失，合并前后会有微小差异。这一块教你如何用“对照实验”来验证合并质量。

### 为什么必须“重新加载一份未合并模型”？

先记住一个坑：`merge_and_unload()` 会返回一个合并后的新模型，**原始的 `PeftModel` 对象保持不变**，仍可用于对比或后续操作。但为了确保对比基准的纯净性，建议**重新从 `base_model_path` 加载一份基础模型，再挂上 adapter，得到一份全新的“未合并 PeftModel”**，专门用来做对比。

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

### 验证一：比较 logits（数值层面）

用同一个固定输入（比如 `"请用一句话介绍你自己。"`，换成你任务里有代表性的句子），分别让 `model_before` 和 `merged_model` 跑一遍，比较输出的 logits 差异：

```python
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果？** 这个值越接近 0，说明合并前后模型内部计算越一致。QLoRA 场景下因为量化精度损失，通常不会是 0，而是一个较小的近似值——只要不大到离谱，就说明合并是成功的。

### 验证二：比较生成文本（行为层面）

数值接近还不够，还要看**实际生成效果**。用贪心生成（`do_sample=False` 去掉随机性，保证可复现对比）：

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print("合并前:", text_before)
print("合并后:", text_after)
print("

## 块 11

## 十一、合并时开启校验并做一致性验证（建议）

### 为什么要单独再加载一份模型？

`merge_and_unload()` 默认返回合并后的新模型对象，原模型保持不变。如果你拿它和"合并前"对比，需要先保存一份未合并的模型副本作为对照基准，否则验证就完全失去意义了。

**正确做法**：在合并前保存一份未合并的模型副本（如 `model_before = copy.deepcopy(model)`），或使用 `get_peft_model` 重新加载原始基础模型并挂载 adapter 作为对照，这样才能真正比较合并前后的差异。

### 显存不够怎么办？

如果无法同时放下两个模型，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍，把 logits 或生成文本**存下来**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比结果。

两次运行需设置相同的随机种子（如 `torch.manual_seed(42)`）和推理参数（如 temperature、top_p），以确保对比的公平性。

### 结果预期：别追求逐位相同

- **QLoRA 场景**：合并前后**通常只是近似一致**。差异主要源于量化反量化误差和采样随机性，`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 不同。建议通过固定随机种子和多次采样取平均来评估一致性，重点确认**没有明显退化或 NaN**。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。

是否可接受，以你的**任务指标与验证集**为准。

---

## 十二、电脑配置批注

本章节需要**实际加载模型**，涉及：加载完整模型、合并权重、导出模型。

### 模型许可证

部署

## 块 12

## 十一、QLoRA模型合并注意事项（重点）（③）

### 11.3 合并时开启校验并做一致性验证（建议）

**核心思路**：合并模型不是“点一下按钮就完事”，建议开启 `safe_merge=True` 并做一致性验证，确保合并后的模型输出与预期一致。

---

**先理解一个关键概念：反量化 ≠ 手动还原**

很多同学以为“反量化”是把 4-bit 权重逐层手动还原成高精度，**不是**。正确做法是：

> **重新加载一份不带量化配置的基础模型**，再挂载同一个 Adapter 合并。

具体代码模式：
```python
# 不传 quantization_config / load_in_4bit
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16  # 或 float16
)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload(safe_merge=True)
```

这样合并发生在高精度权重上，天然避开“在 4-bit 状态下合并”的类型/精度问题。

---

**为什么必须这样做？**

不同 PEFT / bitsandbytes / Transformers 版本对“量化模型能否直接调用 `merge_and_unload`”支持不同：
- 部分版本直接报错或警告
- 部分较新版本有一定支持，但仍有精度损失风险

所以**“先反量化（重载高精度）再合并”是跨版本最稳妥的做法**。是否可以在量化状态下直接合并，以你所装版本的文档和实际运行结果为准。

---

**合并后必做：一致性验证**

合并完成后，建议用同一段输入分别跑：
1. 合并后的完整模型
2. 原始 PeftModel（Adapter 未合并）

对比两者输出是否一致。若差异明显，说明合并过程有问题，回到排查表检查。

---

**常见问题速查表（本块重点）**

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 PeftModel（可能忘了挂载 Adapter，或已合并成普通模型） | 确认用 `PeftModel.from_pretrained` 挂载后再调用 |
| 合并后模型变大 | 正常：合并后是完整模型，不再是 Adapter | 无需处理；在意体积可保留 Adapter 不合并 |
| 显存不足 | 高精度完整模型占用高；或用了 FP32、未 offload/分片 | 改用 CPU 合并、用 FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存 tokenizer；特殊 token / generation_config 未导出 | 重新 `save_pretrained`，核查

## 块 13

## 路线A：合并后部署完整模型——把LoRA“焊死”进模型

你已经训练好了一个LoRA模型，现在到了“上生产”的环节。路线A的思路很直接：**把LoRA合并进基础模型，得到一个“完整模型”，然后像用普通模型一样去部署它。**

### 操作流程（三步走）

```text
merge_and_unload()   ← 第一步：在代码里调用这个方法，把LoRA权重合并回基础模型

↓

保存完整模型          ← 第二步：把合并后的模型保存到磁盘

↓

推理框架按普通模型加载  ← 第三步：比如用vLLM直接加载这个merged_model
```

### 这条路线的特点

**优点：**

- **部署链路简单**——推理框架完全不需要知道LoRA的存在，它看到的就是一个普普通通的完整模型，直接加载就行；
- **推理时零额外开销**——因为Adapter已经“焊死”进模型了，所以推理过程中**不存在Adapter切换的成本**。

**代价（你要权衡的地方）：**

- 每合并一个Adapter，你就要保存和分发**一份完整的模型文件**。如果你有多个任务、多个Adapter，那就得存多份完整模型——**存储成本会直线上升**。

### 一句话总结

> 路线A = 合并 → 保存 → 当普通模型用。适合任务少、追求部署简单的场景；任务多了存储会吃不消。

---

**动手验证一下（可选）**：

## 块 14

## 14.3 如何选择：合并还是保留？

你是来学 LoRA 微调的，这里给你一套**快速决策参考**（非硬性规则，以实际部署为准）：

- **只调了一个 Adapter，想部署最简单** → 选**路线A（合并）**，直接得到一个完整模型，部署省心。
- **同一个基础模型要服务多个 Adapter，且推理框架原生支持 LoRA** → 选**路线B（保留 Adapter 动态服务）**，不用重复加载多个完整模型。
- **QLoRA 精度敏感** → 无论走哪条路线，都建议按 §11.3 做合并前后/上线前的一致性验证。

**核心认知**：合并不是唯一部署方式。具体框架用法留到第 15 板块（vLLM 部署）展开。

---

## 十五、总结

本章你已掌握

## 测评题目

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB1201-003
LoRA中，`r`（rank）参数的作用是？

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。
