# LoRA权重合并与模型导出详解：merge_and_unload使用方法（最终版）

## 参考资料

① 权威：《Hugging Face PEFT LoraModel.merge_and_unload API》
　　网址：https://huggingface.co/docs/peft/package_reference/lora#peft.LoraModel.merge_and_unload

② 底稿：《LoRA训练后模型合并与导出详解》
　　网址：https://blog.csdn.net/l35633/article/details/161430812

③ 参考文章：《QLoRA/LoRA训练完成后的模型保存与合并》
　　网址：https://blog.csdn.net/weixin_44081096/article/details/163345345

④ 参考文章：《PEFT中LoRA权重合并实践》
　　网址：https://blog.csdn.net/weixin_33773084/article/details/157409170

---

**适合人群**：已经完成 LoRA / QLoRA 微调训练，希望将训练得到的 Adapter 转换为完整模型并用于部署推理的学习者。

本文重点学习：

* LoRA训练后保存的是什么；
* 为什么需要merge；
* `merge_and_unload()`如何工作；
* 如何导出完整模型；
* 合并后如何独立加载模型。

---

## 一、LoRA训练后保存的是什么？（①②）
### 1.1 LoRA不是直接修改基础模型
LoRA训练过程中：

基础模型：

```text
Base Model

冻结
```

LoRA参数：

```text
Adapter

训练
```

最终结构：

```text
基础模型

+

LoRA Adapter
```

---

例如：

训练：

```text
Llama-7B

+

LoRA Adapter
```

保存后：

通常只有：

```text
adapter_model.safetensors

adapter_config.json
```

而不是完整模型。

---

### 1.2 Adapter有什么特点？
优点：

* 文件小；
* 保存快；
* 可以多个任务共享一个基础模型。

例如：

同一个基础模型：

```text
Base Model
```

可以搭配：

```text
客服Adapter

医疗Adapter

法律Adapter
```

---

缺点：

部署时需要：

```text
基础模型

+

Adapter
```

两个部分同时存在。

---

## 二、为什么需要合并LoRA权重？（②③）
### 2.1 推理时的问题
未合并：

```text
输入

↓

Base Model

+

LoRA Adapter

↓

输出
```

需要：

* 加载基础模型；
* 加载Adapter。

---

对于部署：

可能希望：

```text
输入

↓

完整模型

↓

输出
```

因此需要：

合并权重。

---

### 2.2 LoRA合并数学原理（②）
LoRA训练时：

公式：

```text
W' = W + ΔW
```

其中：

```text
ΔW = BA
```

合并：

就是将：

```text
BA
```

加入：

```text
W
```

得到：

```text
W'
```

---

合并后：

原来的：

```text
W

+

Adapter
```

变成：

```text
W'
```

---

## 三、merge_and_unload是什么？（重点）（①②）
### 3.1 方法作用
PEFT提供：

```python
merge_and_unload()
```

作用：

> 将LoRA权重合并到基础模型，并移除PEFT包装。

简单理解：

执行：

```text
Base Model

+

LoRA Adapter

↓

完整模型
```

---

### 3.2 合并后的变化
合并前：

```text
PeftModel

├── Base Model

└── LoRA Adapter
```

---

合并后：

```text
普通Transformer模型

包含LoRA更新后的权重
```

---

## 四、LoRA模型合并完整流程（重点）（②③）
整体流程：

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

## 五、加载LoRA Adapter（②）
假设：

基础模型：

```python
base_model_path = "model_path"
```

Adapter：

```python
adapter_path = "adapter_path"
```

加载：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel


model = AutoModelForCausalLM.from_pretrained(
    base_model_path
)


model = PeftModel.from_pretrained(
    model,
    adapter_path
)
```

此时：

模型结构：

```text
Base Model

+

LoRA Adapter
```

---

## 六、使用merge_and_unload合并权重（重点）（①②）
执行：

```python
merged_model = model.merge_and_unload()
```

完成：

```text
W

+

BA

↓

W'
```

---

此时：

`merged_model`

已经不是：

```text
PeftModel
```

而是：

```text
普通Transformer模型
```

---

## 七、导出合并后的模型（②④）
合并后：

保存：

```python
merged_model.save_pretrained(
    "./merged_model"
)
```

同时保存Tokenizer：

```python
tokenizer.save_pretrained(
    "./merged_model"
)
```

目录：

```text
merged_model/

├── config.json

├── model.safetensors

├── tokenizer.json

└── tokenizer_config.json
```

> 说明：上面的目录结构是**示意**，实际文件会随模型/分词器类型变化（例如可能还有 `generation_config.json`、`special_tokens_map.json`、`model.safetensors.index.json` + 分片权重等；有些模型的 tokenizer 是 `vocab.json`/`merges.txt` 或 `tokenizer.model`）。

> 导出后建议核查以下几项，避免"能加载但行为不对"：
> - **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中**新增/修改过的特殊 token**（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里；若训练时扩过词表，还要确认 embedding/lm_head 尺寸与词表匹配。
> - **generation_config.json**：如果依赖特定的生成默认值（eos/pad token id、`max_new_tokens`、采样参数等），确认它一并导出，否则推理时可能回退到不合适的默认行为。
> - **dtype**：确认合并/保存时的精度符合预期（FP16/BF16 还是 FP32），dtype 不一致会影响显存占用与数值表现。
> - **config.json 完整性**：确认模型结构配置（层数、hidden size、任务头等）正确，且已**移除 PEFT/adapter 相关配置**（合并后应是普通 Transformers 模型，不应再残留 `adapter_config.json`）。

---

## 八、合并后如何加载推理？（③④）
合并后：

不需要：

```text
peft
```

只需要：

```python
from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    "./merged_model"
)
```

流程：

```text
用户输入

↓

完整模型

↓

生成结果
```

---

## 九、LoRA合并与不合并区别（②③）
| 项目     | 不合并          | 合并             |
| ------ | ------------ | -------------- |
| 模型结构   | Base+Adapter | 完整模型           |
| 文件大小   | 小            | 大              |
| 部署复杂度  | 较高           | 简单             |
| 多个任务切换 | 方便           | 不方便            |
| 推理加载   | 需要PEFT       | 直接Transformers |

---

## 十、什么时候应该合并？（③④）
> 总体原则：合并与否是一种**权衡**，不是"上线就一定要合并、分享就一定合并"。是否合并取决于：是否需要**多个 Adapter 动态切换**、部署所用的**推理框架**是否原生支持 LoRA、对**存储/分发体积**的要求，以及对**精度**（尤其 QLoRA 场景）的接受度。下面的"推荐/不建议"只是常见情形，实际以你的部署方式为准。

### 10.1 推荐合并场景
#### 场景1：上线部署（单一 Adapter、追求简单）
例如：

生产环境：

```text
服务器

↓

直接加载模型

↓

提供API
```

当只需要**一个固定 Adapter**、且希望部署链路尽量简单（不额外依赖 PEFT、不做运行时 Adapter 加载）时，合并通常更方便。

> 注意：如果推理框架本身支持直接加载 LoRA（例如 vLLM 的多 LoRA 服务，见 §十四），或你需要在同一基础模型上**热切换多个 Adapter**，那么不合并、以 Adapter 形式部署反而更灵活；此时是否合并要看框架能力与业务需求。

---

#### 场景2：分享模型
如果希望对方**开箱即用、无需理解 Adapter 机制**，可以发布合并后的完整模型：

```text
merged_model
```

用户无需额外下载 Adapter。

> 权衡：合并后的完整模型体积**明显更大**（等于一份完整基础模型），分发/存储成本更高；如果对方已经有同一基础模型，单独分享体积很小的 Adapter 往往更省带宽与存储。两种方式各有取舍，可按受众与分发成本选择。

---

### 10.2 不建议合并场景
#### 场景1：多个任务Adapter
例如：

一个基础模型：

```text
Base Model
```

多个：

```text
Adapter A

Adapter B

Adapter C
```

保留Adapter更灵活。

---

#### 场景2：持续训练
如果继续调整LoRA：

保留Adapter。

---

## 十一、QLoRA模型合并注意事项（重点）（③）
QLoRA：

基础模型：

```text
4-bit量化
```

LoRA：

```text
FP16/BF16
```

QLoRA 训练时基础模型是以 4-bit（NF4）**存储**的，而 LoRA 参数是较高精度（FP16/BF16）。因此把 LoRA 更新量直接加回 4-bit 权重，涉及**精度与权重类型的处理**，需要留意以下几点。

### 11.1 直接合并可能遇到的情况
直接在 4-bit 状态下合并，可能出现：

* 精度损失（4-bit 权重反量化再合并，与全程高精度不完全等价）；
* 权重类型不匹配、或所用版本不支持在量化状态下直接 `merge_and_unload`。

> 说明：上面只是**可能出现的情况**，并非每次都必然报错。是否能顺利合并、以及合并质量如何，还取决于 PEFT / bitsandbytes / Transformers 的**版本**、具体量化配置、以及是否按下面的流程先反量化。请以你所用版本的文档与实际运行结果为准。

---

### 11.2 常见做法：先反量化到高精度再合并
一种常见且稳妥的做法是：**不在 4-bit 状态下直接合并**，而是先用高精度（FP16/BF16）重新加载一份基础模型，再挂载同一份 Adapter 后合并。

```text
以高精度（FP16/BF16）重新加载基础模型

↓

加载同一份 LoRA Adapter

↓

merge_and_unload()

↓

保存完整模型
```

这样合并发生在高精度权重上，避免了 4-bit 状态下的类型/精度问题。

上面流程对应的**可执行完整代码**如下（把基础模型以 FP16/BF16 重新加载、挂载同一份 Adapter、`merge_and_unload(safe_merge=True)`、保存模型与 tokenizer）：

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

> 提示（版本/内存）：
> - **API/版本**：不同 PEFT 版本对"量化模型能否直接合并"的支持不同，部分版本要求或推荐先反量化；请查阅所用版本的 `merge_and_unload` 文档。上例的关键是**加载基础模型时不再传 `quantization_config`/`load_in_4bit`**，从而得到高精度权重再合并。
> - **内存**：以高精度重新加载完整基础模型会占用**较多内存/显存**（约等于一份 FP16/BF16 完整模型），显存不足时可把 `device_map` 改为 `{"": "cpu"}` 在 CPU 上加载并合并（较慢），或使用 offload。
> - **一致性**：QLoRA 合并得到的高精度模型，与"训练时 4-bit + Adapter"的推理结果**不会逐位相同**，通常只是**近似**；对精度敏感的场景需要按 §11.3 做验证。

---

### 11.3 合并时开启校验并做一致性验证（建议）
`merge_and_unload` 支持 `safe_merge=True`，在合并时会做数值检查（例如检测是否产生 NaN），可作为一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，建议**对比合并前后模型在同一批输入上的输出**，确认合并没有引入明显偏差：

```text
准备一小批固定输入（prompt）

↓

分别用「合并前的 PeftModel」和「合并后的模型」推理

↓

对比 logits / 生成文本是否基本一致
```

对应的**最小验证代码**如下（用固定 prompt，先比 logits 的最大绝对差，再比贪心生成文本是否一致）：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 变量沿用 §11.2：base_model_path、adapter_path、compute_dtype、tokenizer
prompt = "请用一句话介绍你自己。"        # 换成你任务里有代表性的固定输入

# 重要：merge_and_unload() 是"原地"操作，会就地修改并返回底层模型，
#       合并后原来的 PeftModel 已经不再是"未合并"状态。
#       所以要单独再加载一份"未合并的 PeftModel"来当作对比基准，
#       不能拿 §11.2 里被合并过的对象当"合并前"。
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter

inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

# 1) 比较 logits：未合并的 PeftModel vs §11.2 合并后的 merged_model
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值

# 2) 比较贪心生成文本（do_sample=False 去掉随机性，便于对比）
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print("合并前:", text_before)
print("合并后:", text_after)
print("生成文本是否一致:", text_before == text_after)
```

> 说明：
> - **为什么要单独再加载一份**：`merge_and_unload()` 会**原地**修改 §11.2 里的 `model`（把 Adapter 合进底层权重并移除 PEFT 包装），因此不能再拿它当"合并前"的对照——那样两边其实是同一份已合并权重，`max_abs_diff` 会恒为 0，验证就失去意义。上面用 `base_for_ref` + `model_before` 重新挂一份**未合并** Adapter 作为对照。
> - **显存不够时**：若无法同时放下两个模型，就按"分两次跑"——先用未合并的 PeftModel 跑一遍、把 logits/文本存下来并释放显存，再加载合并后的模型对比。
> - **结果预期**：QLoRA 场景下合并前后**通常只是近似一致**（见 §11.2），`max_abs_diff` 一般是个较小的非零值、生成文本也可能有个别 token 差异，重点是确认没有明显退化或 NaN，而不是要求逐位相同；LoRA（非量化）场景一般更接近一致，`max_abs_diff` 可以非常小。是否可接受以你的任务指标与验证集为准。

---

## 十二、电脑配置批注
本章节涉及：

* 加载完整模型；
* 合并权重；
* 导出模型。

需要实际加载模型。

最低：

CPU：

* 4核+

内存：

* 16GB+

磁盘：

* 预留模型大小2倍空间。

---

GPU推荐（下面为**带条件经验参考**，不是固定门槛）：

7B模型：

经验参考：

* 约 16GB 显存起（宽松些约 24GB+）

> 说明：合并本身是"权重相加"的一次性计算，**不一定必须在 GPU 上完成**——很多情况下直接在 **CPU + 内存**里加载 FP16/BF16 模型并合并即可，只是速度较慢。上面的 16GB / 24GB 只是**在 GPU 上一次性放下高精度模型**时的经验参考，真实需求还受以下因素影响：
> - 是否用 `device_map`/offload 把权重分散到 CPU 或多卡；
> - 合并时使用的 **dtype**（FP16/BF16 比 FP32 省一半）；
> - 基础模型是否为**分片权重**（分片加载可降低瞬时峰值）；
> - CPU 内存是否充足（显存不够时可退回 CPU 合并）。
>
> 因此显存不足时，优先考虑 CPU 合并或 offload，而不是把 16GB 当作硬性下限。

原因：

合并阶段通常需要同时存在：

* 基础模型（高精度）；
* LoRA 参数；
* 合并后权重。

---

## 十三、常见问题排查
| 问题                  | 常见原因（可能不止一种）           | 排查方向                 |
| ------------------- | ------------ | ------------------ |
| merge_and_unload不存在 | 当前对象不是 PeftModel（可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或已经合并/加载成普通模型） | 确认加载方式，先得到 PeftModel 再调用             |
| 合并后模型变大             | 正常现象：合并后是完整模型，不再是小体积 Adapter         | 无需处理；若在意体积可保留 Adapter 不合并         |
| 显存不足                | 一次性加载高精度完整模型占用高；也可能是 dtype 用了 FP32、未做 offload/分片 | 改用 CPU 合并或 offload、用 FP16/BF16、分片加载 |
| Tokenizer/配置缺失         | 只保存了模型未保存 tokenizer；或特殊 token / generation_config 未一并导出 | 重新 `save_pretrained` 并按 §七核查 tokenizer、generation_config、config  |
| QLoRA合并失败或结果异常           | 原因可能不止一种：在 4-bit 状态直接合并、PEFT/bitsandbytes 版本不支持、量化包装未正确反量化、内存/显存不足 | 按 §十一先反量化到 FP16/BF16 再合并；核对版本；必要时用 CPU 合并；开 `safe_merge=True` 并做一致性验证              |

> 关于"先反量化到 FP16/BF16"的**具体做法**：这里的"反量化"**不是**去手动逐层还原 4-bit 权重，而是——**重新加载一份不带量化配置的基础模型**。也就是用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`（**不传** `quantization_config`/`load_in_4bit`）得到高精度权重，再 `PeftModel.from_pretrained` 挂载**同一份 Adapter** 后合并（完整代码见 §11.2）。这样合并发生在高精度权重上，天然避开了"在 4-bit 状态下合并"的类型/精度问题。
>
> **适用版本提示**：不同 PEFT / bitsandbytes / Transformers 版本对"量化模型能否直接调用 `merge_and_unload`"的支持不同——部分版本会直接报错或给出警告，部分较新版本对某些情形有一定支持但仍可能有精度损失。因此**"先反量化（重载高精度）再合并"是跨版本最稳妥的做法**；是否可以在量化状态下直接合并，请以你所装版本的 `merge_and_unload` 文档与实际运行结果为准。

---

## 十四、与部署阶段的衔接：合并部署 vs 保留 Adapter 动态服务
合并只是"训练后 → 部署前"的一种选择。到了实际推理部署（详见后续第 15 板块 vLLM 相关内容），通常会在两条路线之间权衡：

### 14.1 路线A：合并后部署完整模型
```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

特点：

* 部署链路简单，推理框架当作**普通模型**加载即可，无需理解 LoRA；
* 推理时**不再有 Adapter 切换成本**；
* 代价：每个 Adapter 都要保存/分发一份**完整模型**，多任务时存储成本高。

### 14.2 路线B：保留 Adapter，用支持 LoRA 的推理框架动态服务
一些推理框架（如 **vLLM**）支持在**同一份基础模型**上运行时加载/切换**多个 LoRA Adapter**（Multi-LoRA serving）。

```text
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

特点：

* 多任务/多租户时**只需一份基础模型 + 多个小体积 Adapter**，节省显存与存储；
* 支持**热切换**，无需为每个任务各存一份完整模型；
* 代价：依赖框架对 LoRA 的支持与相应配置，链路比"直接加载合并模型"略复杂；不同框架/版本对 LoRA 的支持范围与限制不同。

> **vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）**：以 vLLM 为例，启动 OpenAI 兼容服务时通常需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**，请求里再用 Adapter 名字指定用哪个。命令行大致形如：
>
> ```text
> vllm serve <base_model_path> \
>     --enable-lora \
>     --lora-modules adapterA=/path/to/adapterA adapterB=/path/to/adapterB \
>     --max-loras 2 \
>     --max-lora-rank 16
> ```
>
> 其中 `--enable-lora` 开启多 LoRA 服务，`--lora-modules 名字=路径` 声明可用 Adapter（这里的**路径就是训练保存下来的 Adapter 目录**，即含 `adapter_model.safetensors` / `adapter_config.json` 的目录，**不是**合并后的完整模型），`--max-loras` 限制同时驻留的 Adapter 数、`--max-lora-rank` 需 **≥ 训练时 LoRA 的 r**。请求端通过把 `model` 字段设为对应 Adapter 名字来选择使用哪个。
>
> **限制与版本提示（重要）**：
> - vLLM 的多 LoRA 是**运行时挂 Adapter**，因此走的是**路线B**——请传入的是**未合并的 Adapter 目录**，而不是 §11 合并后的完整模型（合并后的模型属于路线A，按普通模型直接加载）。
> - 具体**参数名、默认值、是否支持运行时动态增删 Adapter、是否支持 QLoRA/量化基座 + LoRA 组合、以及对 `target_modules`/rank 的限制**都**随 vLLM 版本变化**；上面的 flag 只是**示意形态**，务必以你所装 vLLM 版本的官方文档（`vllm serve --help` / Multi-LoRA 章节）为准。
> - 完整用法、性能与限制留待**第 15 板块（vLLM 部署）**结合当时实际版本展开，本章只需建立"保留 Adapter → 用支持 LoRA 的框架动态服务"这一认知。

### 14.3 如何选择
> 经验参考（非硬性规则，以实际部署为准）：
> - **单一 Adapter、追求最简部署** → 倾向路线A（合并）；
> - **同一基础模型要服务多个 Adapter、且推理框架原生支持 LoRA** → 倾向路线B（保留 Adapter 动态服务）；
> - **QLoRA 精度敏感** → 无论哪条路线，都建议按 §11.3 做合并前后/上线前的一致性验证。

具体框架用法与限制留待第 15 板块（vLLM 部署）展开，这里只需建立"合并不是唯一部署方式"的认知。

---

## 十五、总结
完成本章后，你已经掌握：

✅ LoRA Adapter保存方式
✅ 为什么需要权重合并
✅ merge_and_unload作用
✅ LoRA数学合并过程
✅ 完整模型导出流程
✅ 合并模型独立部署方式
✅ LoRA与QLoRA合并注意事项

核心流程：

```text
LoRA训练

↓

保存Adapter

↓

加载Base Model

↓

PeftModel加载Adapter

↓

merge_and_unload()

↓

保存完整模型
```

下一阶段：

**LoRA/QLoRA完整微调实战：PEFT + Trainer训练大语言模型**

将进入端到端训练流程。
