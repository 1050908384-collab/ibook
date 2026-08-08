> **参考文章链接**
> 
> ① 底稿：《vLLM采样模块架构与优化实践解析》
> 　　网址：https://blog.csdn.net/weixin_34005090/article/details/163297267
> 
> ② 权威文章1：《Sampling Parameters — vLLM》
> 　　网址：https://docs.vllm.ai/en/v0.6.2/dev/sampling_params.html
> 
> ③ 权威文章2：《Inference Parameters — vLLM》
> 　　网址：https://docs.vllm.ai/en/v0.8.4/api/inference_params.html
> 
> ④ 参考文章1：《10-vLLM-采样与生成策略分析》
> 　　网址：https://blog.csdn.net/ld326/article/details/162582707

---

**适合人群**：已掌握贪心搜索与束搜索的确定性生成策略，希望理解随机性采样的数学原理，掌握 Temperature、Top-k、Top-p 三个核心参数的配置方法，并能根据业务场景平衡输出多样性与质量的进阶学习者。每一步都拆解到"是什么、为什么、怎么配"，你只需要跟着理解。完成本章后，你将能够熟练使用 vLLM 的 `SamplingParams` 配置随机采样策略，为后续"理解 vLLM 的 PagedAttention 与 Continuous Batching 机制"打下坚实基础。

**预计学习时长**：约 1 小时  
**环境准备**：复用上一章的 vLLM 环境（Python 3.10~3.13、vLLM 已安装、NVIDIA GPU 显存 ≥16GB）

---

## 第一章：随机采样的定位——从"唯一答案"到"多种可能"

### 1.1 为什么需要随机采样？

在上一章中，我们学习了**贪心搜索**和**束搜索**——它们都是**确定性较强**的生成策略，在相同软硬件与执行配置下，相同输入通常得到相同输出。这在数学计算、代码生成、机器翻译等场景中往往更合适。

但在很多场景中，**多样性本身就是价值**：

| 场景 | 为什么需要多样性？ | 确定性策略的问题 |
|------|------------------|---------------|
| **聊天对话** | 同样的问候不应得到一模一样的回复 | 贪心搜索会让机器人像"复读机" |
| **创意写作** | 诗歌、故事需要不同风格和表达 | 束搜索输出单调、缺乏惊喜 |
| **头脑风暴** | 需要多角度、多思路的建议 | 确定性输出永远走同一条路 |
| **角色扮演** | 不同角色应有不同说话风格 | 温度太低会让所有角色"同一个声音" |

**为什么这样写**：随机采样的核心思想不是"随便选"，而是**在高质量候选中引入可控的随机性**。它既避免了贪心搜索的单调，又不会像完全随机那样输出 nonsense。

### 1.2 随机采样的工作流程（①）

vLLM 的采样模块在每个解码步骤中执行以下流程：

```
模型输出 logits（原始分数）
    ↓
① 应用温度调节（Temperature Scaling）
    ↓
② 应用 Top-k 过滤（固定候选池）
    ↓
③ 应用 Top-p 过滤（动态候选池）
    ↓
④ 概率归一化（Softmax）
    ↓
⑤ 从过滤后的分布中随机采样
    ↓
输出下一个 token
```

**为什么这样写**：这三个参数（Temperature、Top-k、Top-p）是**层层递进、相互配合**的。Temperature 先"软化"概率分布，Top-k 和 Top-p 再"裁剪"候选范围，最后从剩余候选中随机抽取。

---

## 第二章：Temperature 温度系数——控制随机性的"旋钮"

### 2.1 核心原理：对 logits 做"除法"（①④）

Temperature 的数学操作非常简单：将模型输出的原始 logits（未归一化的分数）除以一个温度系数 $T$：

$$\text{adjusted\_logits}_i = \frac{\text{logits}_i}{T}$$

然后对调整后的 logits 做 Softmax，得到最终的概率分布：

$$P(\text{token}_i) = \frac{\exp(\text{adjusted\_logits}_i)}{\sum_j \exp(\text{adjusted\_logits}_j)}$$

**为什么这样写**：这个"除法"操作的效果非常直观：
- 当 $T < 1$（低温）：logits 被"放大"，概率分布变得更"尖锐"，高概率 token 的优势被放大，随机性降低
- 当 $T > 1$（高温）：logits 被"缩小"，概率分布变得更"平坦"，原本概率较低的 token 也有机会被选中，随机性增加
- 当 $T = 1$：不做任何调整，使用模型原始概率分布
- 当 $T \to 0$：趋近于贪心搜索（argmax）

### 2.2 温度系数的直观效果（①④）

假设模型对 5 个 token 的原始 logits 为：`[2.0, 1.5, 0.5, 0.1, -0.5]`

| 温度 $T$ | 效果描述 | 概率分布特点 | 适用场景 |
|---------|---------|------------|---------|
| **0.1** | 极低温 | 几乎 100% 选最高分 token | 数学计算、代码生成 |
| **0.3** | 低温 | 高分 token 占主导，偶有变化 | 事实问答、摘要 |
| **0.7** | 中温 | 高质量候选都有机会 | 通用对话、翻译 |
| **1.0** | 常温 | 原始分布，自然随机 | 开放式写作 |
| **1.5** | 高温 | 低分 token 也有显著概率 | 头脑风暴、创意写作 |
| **2.0+** | 极高温 | 接近均匀分布，输出混乱 | 一般不推荐 |

**为什么这样写**：Temperature 是控制生成随机性**最直接、最常用**的参数。它不改变候选池的大小，只改变候选池内各 token 的"竞争力"。

### 2.3 在 vLLM 中配置 Temperature（②③）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 低温：确定性输出
sampling_params = SamplingParams(temperature=0.3, max_tokens=100)
outputs = llm.generate("请总结以下文章：...", sampling_params)

# 中温：平衡质量与多样性
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate("写一首关于春天的诗", sampling_params)

# 高温：高多样性
sampling_params = SamplingParams(temperature=1.2, max_tokens=100)
outputs = llm.generate("给我三个不同的创业点子", sampling_params)
```

**为什么这样写**：vLLM 的 `SamplingParams` 中，`temperature` 默认值为 `1.0`。设为 `0` 时触发贪心搜索（如上一章所述），大于 `0` 时启用随机采样（②）。

---

## 第三章：Top-k 采样——固定大小的候选池（①④）

### 3.1 核心原理：只保留概率最高的 k 个 token

Top-k 采样的规则很简单：

> **每一步，只考虑概率最高的 k 个 token，其余 token 的概率强制设为 0，然后在这 k 个 token 中按概率采样。**

```
原始概率分布（词表大小 50000）：
  token_A: 0.25
  token_B: 0.18
  token_C: 0.12
  token_D: 0.08
  ...（剩余 49996 个 token 的概率之和 = 0.37）

Top-k=3 过滤后：
  token_A: 0.25 → 0.25/0.55 = 0.455
  token_B: 0.18 → 0.18/0.55 = 0.327
  token_C: 0.12 → 0.12/0.55 = 0.218
  其余: 0（被过滤）

然后从 {A, B, C} 中按新概率采样
```

**为什么这样写**：Top-k 的核心价值是**防止选中概率极低的"垃圾 token"**。即使 Temperature 很高，Top-k 也能保证只在"头部候选"中随机，避免模型说出完全无意义的内容。

### 3.2 Top-k 的取值建议（①④）

| Top-k 值 | 效果 | 适用场景 |
|---------|------|---------|
| **k=1** | 等价于贪心搜索 | 需要唯一正确答案 |
| **k=10** | 非常保守，只考虑最可能的 10 个词 | 技术文档、代码 |
| **k=40~50** | 社区常见经验取值，兼顾质量与多样性（非官方默认，需按任务实测） | 通用对话、翻译 |
| **k=100** | 较宽松，允许更多词汇变化 | 创意写作 |
| **k=-1 或 0** | 禁用 Top-k，考虑全部词表 | 特殊实验场景 |

**为什么这样写**：Top-k 的"k"与词表大小无关。无论词表是 3 万还是 10 万，Top-k=50 都只保留概率最高的 50 个 token。这使得 Top-k 成为一个**与模型规模无关**的稳健参数。

### 3.3 在 vLLM 中配置 Top-k（②③）

```python
from vllm import SamplingParams

# Top-k=50（通用配置）
sampling_params = SamplingParams(
    temperature=0.8,
    top_k=50,           # 只考虑概率最高的 50 个 token
    max_tokens=100
)

# Top-k=10（保守配置）
sampling_params = SamplingParams(
    temperature=0.5,
    top_k=10,           # 只考虑前 10 个 token
    max_tokens=100
)

# 禁用 Top-k（考虑全部词表）
sampling_params = SamplingParams(
    temperature=1.0,
    top_k=-1,           # -1 表示禁用
    max_tokens=100
)
```

**为什么这样写**：vLLM 中 `top_k` 默认值为 `-1`（即禁用）。当与 `top_p` 同时设置时，vLLM 会先应用 Top-k 过滤，再应用 Top-p 过滤（②）。

---

## 第四章：Top-p 采样（核采样）——动态大小的候选池（①④）

### 4.1 核心原理：按累积概率动态裁剪

Top-p 采样（Nucleus Sampling）比 Top-k 更灵活。它不按"固定数量"裁剪，而是按"累积概率"裁剪：

> **从概率最高的 token 开始累加，直到累积概率达到 p，只保留这些 token，其余丢弃。**

```
原始概率分布（已按概率降序排列）：
  token_A: 0.30  → 累积: 0.30
  token_B: 0.25  → 累积: 0.55
  token_C: 0.15  → 累积: 0.70
  token_D: 0.10  → 累积: 0.80
  token_E: 0.08  → 累积: 0.88
  ...（剩余 token）

Top-p=0.9 时：
  保留 token_A + B + C + D + E（累积 0.88 < 0.9，继续）
  假设 token_F: 0.05 → 累积 0.93 > 0.9，停止
  最终保留 {A, B, C, D, E, F}，共 6 个 token

Top-p=0.6 时：
  保留 token_A + B + C（累积 0.70 > 0.6，停止）
  最终保留 {A, B, C}，共 3 个 token
```

**为什么这样写**：Top-p 的"动态"体现在：概率分布"尖锐"时（某个 token 概率极高），保留的候选少；概率分布"平坦"时（多个 token 概率接近），保留的候选多。这比 Top-k 的"固定数量"更智能。

### 4.2 Top-p vs Top-k 的对比（①④）

| 维度 | Top-k | Top-p |
|------|-------|-------|
| **裁剪依据** | 固定数量（k 个） | 累积概率（达到 p 为止） |
| **候选池大小** | 固定为 k | 动态变化（1 ~ 词表大小） |
| **对分布形状的适应** | 不敏感 | 敏感（尖锐时池小，平坦时池大） |
| **极端情况** | 可能包含极低概率 token | 不会包含累积概率外的 token |
| **推荐用法** | 与 Top-p 配合使用 | 作为主要过滤手段 |

**生活化比喻**：
- **Top-k** 像"考试只录取前 50 名"——不管第 50 名和第 51 名差距多大，一刀切
- **Top-p** 像"录取直到累计分数达到总分的 90%"——如果前几名分数很高，可能只录取 3 人；如果分数很接近，可能录取 100 人

### 4.3 在 vLLM 中配置 Top-p（②③）

```python
from vllm import SamplingParams

# 通用配置：Top-p=0.95
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,         # 保留累积概率达到 95% 的候选
    top_k=50,           # 先 Top-k=50，再 Top-p=0.95
    max_tokens=100
)

# 高多样性：Top-p=0.99（保留更多候选）
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.99,
    max_tokens=100
)

# 低多样性：Top-p=0.5（只保留头部候选）
sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.5,
    max_tokens=100
)

# 禁用 Top-p（考虑全部词表）
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,          # 1.0 表示禁用
    max_tokens=100
)
```

**为什么这样写**：vLLM 中 `top_p` 默认值为 `1.0`（即禁用）。当 `top_p < 1.0` 时启用核采样。vLLM 的采样顺序是：**先 Temperature 缩放 → 再 Top-k 过滤 → 再 Top-p 过滤 → 最后随机采样**（①）。

---

## 第五章：三种参数的组合使用与选型

### 5.1 参数组合的优先级（①）

vLLM 采样模块的执行顺序是严格固定的：

```
logits（模型原始输出）
    ↓ Step 1: Temperature 缩放
    ↓ Step 2: Top-k 过滤（保留前 k 个）
    ↓ Step 3: Top-p 过滤（保留累积概率达 p 的）
    ↓ Step 4: Softmax 归一化
    ↓ Step 5: 随机采样
```

**为什么这样写**：这个顺序意味着：
- Temperature 影响**所有 token** 的相对概率
- Top-k 在 Temperature 之后执行，裁剪的是"缩放后的"候选池
- Top-p 在 Top-k 之后执行，进一步缩小范围
- 最终采样只在"三重过滤后"的候选中进行

### 5.2 典型场景配置模板

| 场景 | Temperature | Top-k | Top-p | 说明 |
|------|------------|-------|-------|------|
| **数学/代码** | 0 | - | - | 贪心搜索，唯一正确答案 |
| **事实问答** | 0.3 | 10 | 0.5 | 低温 + 小候选池，确保准确 |
| **通用对话** | 0.7~0.8 | 40~50 | 0.9~0.95 | 业界通用配置，平衡质量与多样性 |
| **创意写作** | 1.0~1.2 | 100 | 0.99 | 高温 + 大候选池，鼓励多样性 |
| **头脑风暴** | 1.2~1.5 | -1 | 0.99 | 极高温，最大化随机性 |
| **翻译** | 0.5~0.7 | 20~30 | 0.85 | 中低温，保证翻译质量 |

**为什么这样写**：这三个参数是**相互配合**的。单独调一个往往效果不佳，需要组合调整。例如"高温 + 小 Top-k"会产生"在少数候选中随机选"的效果，与"低温 + 大 Top-p"的"在多数候选中选大概率"效果完全不同。

### 5.3 调试技巧：固定种子提升可复现性（②③）

在调试采样参数时，建议固定随机种子，在受控环境下尽量让每次运行结果一致：

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    seed=42,            # 固定随机种子
    max_tokens=100
)
```

**为什么这样写**：`seed` 用于固定采样时随机数生成器的状态。在同一 vLLM 版本、相同的 `prompt` 与采样参数下，固定 `seed` 通常能得到一致的输出，非常适合 A/B 测试不同参数组合。需要留意：跨不同硬件、不同 batch 组合、不同框架版本时，浮点计算顺序等差异仍可能带来细微不一致，因此"完全逐位可复现"要在受控环境下才成立，具体行为以所装版本为准。

---

## 第六章：vLLM 中的完整配置实践

### 6.1 `SamplingParams` 完整参数速查（②③）

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    n=1,                        # 返回序列数
    temperature=0.8,            # 温度系数
    top_p=0.95,                 # 核采样阈值
    top_k=50,                   # Top-k 候选数
    min_p=0.0,                  # 最小相对概率阈值（新增）
    presence_penalty=0.0,       # 存在惩罚（鼓励新词）
    frequency_penalty=0.0,      # 频率惩罚（ discourages repetition）
    repetition_penalty=1.0,     # 重复惩罚（>1 减少重复）
    seed=None,                  # 随机种子
    max_tokens=100,             # 最大生成 token 数
    min_tokens=0,               # 最小生成 token 数
    stop=None,                  # 停止词列表
    logprobs=None,              # 返回 top logprobs 数量
)
```

**为什么这样写**：`min_p` 是 vLLM 新增参数，表示 token 概率必须不低于"最可能 token 概率 × min_p"才能被考虑。例如 `min_p=0.05` 时，如果最可能 token 概率是 0.3，则所有概率低于 0.015 的 token 都会被过滤。这是 Top-p/Top-k 之外的第三层过滤。

### 6.2 服务启动与采样参数的设置方式（④）

采样参数（`temperature`/`top_p`/`top_k`/`min_p` 等）**不建议在启动命令里写死**——不同 vLLM 版本对"启动时设默认采样参数"的支持差异很大，很多版本根本没有对应开关。最通用、最不受版本影响的做法是：`vllm serve` 只负责把模型服务跑起来，采样参数在**每次客户端请求**（`SamplingParams` 或 HTTP 请求体）中传入。

```bash
# 只启动服务，不在此处写采样参数
vllm serve Qwen/Qwen2.5-7B-Instruct
```

随后在请求里按场景传采样参数（见下方 6.3）。如需服务侧统一默认值，请先用 `vllm serve --help` 确认你所装版本是否提供相应选项及其准确名称，不要照抄他处的参数名。

### 6.3 API 调用示例（④）

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "写一首关于秋天的诗"}],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 200
    }'
```

---

## 第七章：电脑配置批注

本章节涉及 vLLM 推理服务的采样参数实验与调优。

### 7.1 本地环境配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 1 张 NVIDIA GPU，显存 **≥16GB** | 1 张 RTX 4090（24GB）/ A100（40GB/80GB） |
| **CPU** | 8 核 | 16 核+ |
| **内存** | 32GB | 64GB+ |
| **磁盘** | 100GB SSD | 500GB+ NVMe SSD |
| **系统** | Linux（Ubuntu 20.04+）或 WSL2 | 原生 Linux（Ubuntu 22.04+） |
| **是否支持 CPU 运行** | ⚠️ 支持但极慢（仅调试） | — |

### 7.2 云端环境配置参考

| 场景 | 推荐实例 | 适用采样实验 |
|------|---------|-------------|
| 低成本参数调优 | AutoDL RTX 3090（24GB） | 7B 模型，单请求采样对比 |
| 标准服务 | AWS g5.xlarge（A10G 24GB） | 7B~13B 模型，多并发采样 |
| 高并发 A/B 测试 | AWS g5.12xlarge（4×A10G） | 多组采样参数并行对比 |

---

## 第八章：常见问题与避坑清单

| 坑点 | 现象 | 解决方案 |
|------|------|---------|
| **temperature=0 但 top_p/top_k 仍生效** | 输出仍有随机性 | `temperature=0` 时 vLLM 自动忽略 top_p/top_k，检查是否有 `seed` 未固定或 `logits_processors` 引入随机性 |
| **top_k 和 top_p 同时设置冲突** | 候选池比预期小 | 记住执行顺序：先 Top-k 再 Top-p。如果 Top-k=10 后只剩 5 个候选，Top-p 再过滤可能只剩 3 个 |
| **高温 + 大 top_p 输出混乱** | 生成无意义内容 | 高温（>1.5）时务必配合适中的 top_p（0.9~0.95），不要同时设 top_p=0.99 和 temperature=1.5 |
| **不同请求结果差异大** | 难以评估参数效果 | 固定 `seed`，使用相同 prompt 做 A/B 测试 |
| **top_k=-1 不生效** | 仍被过滤 | vLLM 中 `top_k=-1` 表示禁用，但如果同时设置了 `top_p<1.0`，Top-p 过滤仍会生效 |
| **重复输出** | "很好很好很好..." | 增加 `repetition_penalty`（如 1.1~1.2），或降低 temperature |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **Temperature** | 控制随机性的"旋钮"，$T<1$ 更确定，$T>1$ 更随机，$T=0$ 即贪心搜索 |
| **Top-k** | 固定保留概率最高的 k 个 token，防止选中垃圾词，k=50 是社区常见经验取值（vLLM 默认 -1 即禁用） |
| **Top-p** | 按累积概率动态裁剪候选池，p=0.95 表示保留累积概率达 95% 的候选 |
| **执行顺序** | Temperature → Top-k → Top-p → Softmax → 随机采样 |
| **组合原则** | 需要确定性：低温 + 小 top_k + 小 top_p；需要多样性：高温 + 大 top_k + 大 top_p |
| **min_p** | 新增参数，token 概率必须不低于"最可能 token × min_p" |
| **seed** | 固定随机种子，确保采样结果可复现，便于 A/B 测试 |
| **vLLM 默认值** | temperature=1.0, top_p=1.0（禁用）, top_k=-1（禁用） |

---

## 养成的好习惯

1. **从通用配置开始**——`temperature=0.8, top_p=0.95, top_k=50` 是业界最稳妥的起点
2. **调参时只动一个变量**——同时改 temperature 和 top_p 会让你无法判断是哪个参数在起作用
3. **固定 seed 做 A/B 测试**——相同的 prompt + 相同的 seed = 可对比的输出
4. **低温不等于高质量**——temperature=0.1 可能让输出变得机械和重复，0.3~0.5 往往更自然
5. **Top-p 比 Top-k 更推荐作为主要过滤手段**——Top-p 的动态适应性更好，Top-k 作为辅助保险
6. **监控重复率**——如果输出频繁重复，先调 `repetition_penalty`，再调 temperature

---

## 下一步预告

掌握了 Temperature、Top-p、Top-k 采样参数后，你已经能够灵活控制大模型输出的随机性与多样性。接下来，你将进入 **"理解 vLLM 的 PagedAttention 与 Continuous Batching 机制"** 的学习——深入理解 vLLM 如何通过"分页管理 KV Cache"解决显存碎片问题，以及如何通过"连续批处理"在高并发下把推理吞吐量提升数倍到十几倍（具体幅度视模型与负载而定）。这是从"会调参"进化为"懂引擎"的关键一步，继续加油！