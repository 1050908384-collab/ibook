# 学习包

## 块 1

# 第一章：vLLM 的定位——为什么需要专门的推理引擎？

## 1.1 从训练到推理的鸿沟

在训练阶段，我们关注的是**如何让模型学会**——通过反向传播更新参数。但在部署阶段，核心问题变成了**如何让模型快且省地服务用户**。

使用原始的 HuggingFace Transformers 进行推理时，你会遇到以下问题：

| 问题 | 现象 | 原因 |
|------|------|------|
| **吞吐量低** | 单卡只能同时处理 1~2 个请求 | 每个请求独立计算，GPU 算力大量空闲等待 |
| **显存浪费** | 即使短请求也要为长请求预留最大长度显存 | 静态分配 KV Cache，按最大长度预留 |
| **延迟高** | 第一个 token 等待时间长 | 没有批处理机制，请求串行处理 |

**验证任务**：运行以下命令并观察输出，确认是否存在问题：
```bash
vllm serve --help
```

## 块 2

**vLLM 的核心价值**：通过 **PagedAttention** 和 **Continuous Batching** 技术，在高并发场景下相比朴素的逐请求推理通常能带来数倍到十几倍的吞吐提升（具体倍数取决于模型、序列长度、并发量与对比基线，官方与各类评测中常见"约一个数量级"的量级，需以自己场景实测为准），同时保持较低延迟。

### 1.2 vLLM 是什么？

vLLM 是由 UC Berkeley 开发的高吞吐量、低延迟的大语言模型推理和服务引擎。它的核心特性包括：

- **PagedAttention**：将 KV Cache 分页管理，消除显存碎片和过度预留
- **Continuous Batching**：动态批处理，新请求随时加入当前 batch
- **OpenAI API 兼容**：一键部署为兼容 OpenAI 接口的服务
- **多硬件支持**：NVIDIA CUDA、AMD ROCm、Intel XPU、TPU、Apple Silicon

**为什么这样写**：训练好的模型只有部署成服务才能产生价值。vLLM 是目前工业界最主流的大模型推理引擎，掌握它是从"会训练"到"能上线"的必经之路。

---

## 第二章：安装前环境检查——确保 CUDA 就绪

（个性化提示：学习风格：hands-on）

## 块 3

### 2.1 检查 NVIDIA 驱动与 CUDA 版本

在安装 vLLM 库之前，必须确认你的 NVIDIA 驱动支持的最高 CUDA 版本。这是确保安装成功的关键步骤。

**预期输出**（右上角显示 CUDA Version）：

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------|
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/C

## 块 4

| 驱动版本（示例） | `nvidia-smi` 显示的 CUDA Version | 通常可安装的 vLLM CUDA 构建 |
|---------|-------------|----------------|
| ≥525.60.13 | CUDA 12.0 | CUDA 11.8 / 12.1（借助次版本兼容，常可运行 12.x wheel） |
| ≥535.54.03 | CUDA 12.2 | CUDA 11.8 / 12.1 / 12.x（以官方矩阵为准） |
| ≥550.54.15 | CUDA 12.4 | CUDA 11.8 / 12.1 / 12.4 |

> **说明**：表中"可安装的 CUDA 构建"是经验性示意而非绝对边界。是否成功取决于驱动版本、目标 CUDA 次版本、以及 vLLM 该发行的构建方式；遇到 `no kernel image is available` 一类报错时，优先对照官方兼容性矩阵，或改装更低 CUDA 版本的 vLLM wheel。

（个性化提示：学习风格：hands-on）

## 块 5

### 2.2 检查 Python 版本（②③）

vLLM 对 Python 版本有要求，具体支持范围随 vLLM 版本变化。就本文示例而言，预编译 wheel 常见覆盖 Python 3.10 / 3.11 / 3.12：

```bash
python --version
```

**为什么这样写**：vLLM 的预编译 wheel 通常提供 Python 3.10/3.11/3.12 的版本。更早（如 3.9）或更新（如 3.13）的 Python 可能缺少现成 wheel，需要从源码编译，会明显增加安装复杂度。到底支持到哪个版本，请以你所装 vLLM 版本的官方发行说明为准，不要把这里的 3.10~3.12 当成永久结论。

（个性化提示：学习风格：hands-on）

## 块 6

### 2.3 创建隔离的虚拟环境（②③）

vLLM 需要编译多个 CUDA 内核，与现有 PyTorch 版本极易冲突。**强烈建议**使用全新的虚拟环境：

**本地环境（conda）**：
```bash
conda create -n vllm python=3.12 -y
conda activate vllm
```

**云端环境（AutoDL/阿里云等）**：
云服务器通常已预装 PyTorch，但仍建议新建环境：
```bash
conda create -n vllm python=3.12 -y
conda activate vllm
```

**为什么这样写**：vLLM 的 CUDA 内核编译对 PyTorch 版本和 CUDA 版本有严格的二进制兼容性要求。即使是相同版本的 PyTorch，不同的构建配置也可能导致不兼容（②）。

---

## 第三章：安装 vLLM——PyTorch 版本兼容性是关键

（个性化提示：学习风格：hands-on）

## 块 7

```markdown
# vLLM 安装指南

## 什么是 vLLM？
vLLM 是一个开源的模型推理框架，基于 Hugging Face 的生态系统，支持快速部署和推理。它可以帮助用户轻松搭建模型推理环境，满足多种应用场景的需求。

## 安装 vLLM
### 方法 1：直接安装（推荐）
vLLM 的官方预编译 wheel 通常

## 块 8

### 3.3 安装方式三：使用 uv（官方推荐）（③）

`uv` 是一个极快的 Python 包管理器，vLLM 官方推荐使用：

```bash
# 安装 uv
pip install uv

# 创建环境并安装 vLLM（自动检测 CUDA 版本）
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

**为什么这样写**：`--torch-backend=auto` 会让 uv 自动检测本机 CUDA 驱动版本，选择最合适的 PyTorch 索引。这比手动指定 CUDA 版本更不容易出错。

（个性化提示：学习风格：hands-on）

## 块 9

### 3.4 国内网络环境加速（①③）

国内访问 HuggingFace 和 PyPI 可能较慢，建议配置镜像：

```bash
# 使用清华 PyPI 镜像
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/

# 使用 ModelScope 替代 HuggingFace 下载模型
export VLLM_USE_MODELSCOPE=True
export MODELSCOPE_CACHE=/mnt/disk1/models
```

**为什么这样写**：vLLM 默认从 HuggingFace Hub 下载模型。在国内环境中，`VLLM_USE_MODELSCOPE=True` 会将模型下载源切换到 ModelScope（魔搭社区），大幅提升下载速度（③）。

---

## 第四章：电脑配置批注

本章节涉及 vLLM 推理服务的本地部署与云端部署。

（个性化提示：学习风格：hands-on）

## 块 10

### 4.1 本地环境配置要求

先区分两类"最低要求"：一是**能装上、能验证 vLLM** 的软件/硬件底线，二是**跑本文 7B 示例**的推荐配置——两者的显存门槛不同，不要混为一谈。

**软件安装与环境验证的最低要求**（用小模型如 `opt-125m`/`Qwen2.5-0.5B` 即可）：

| 项目 | 最低要求 | 说明 |
|------|---------|------|
| **操作系统** | Linux（Ubuntu 20.04+）或 WSL2 | 生产建议原生 Linux |
| **GPU** | 一张受支持的 NVIDIA GPU 即可 | 小模型验证对显存要求不高，几 GB 显存也能跑通 `opt-125m` |
| **NVIDIA 驱动** | 驱动支持 CUDA 11.8+ | 以 `nvidia-smi` 为准 |
| **Python** | 以所装 vLLM 支持范围为准（本文示例 3.10~3.12） | 3.13 可能需源码编译 |
| **是否支持 CPU 运行** | ⚠️ 支持但极慢（仅调试） | 生产必须 GPU |

**运行本文 7B 模型示例的推荐配置**：

| 项目 | 推荐配置 | 说明 |
|------|---------|------|
| **GPU 显存** | ≥16GB 起步，24GB 更从容 | 16GB 是 7B 场景的经验下限，非"安装 vLLM 的统一门槛" |
| **CPU** | 16 核+ | — |
| **内存** | 32GB（推荐 64GB+） | — |
| **磁盘** | 100GB SSD（推荐 500GB+ NVMe） | 模型权重占用大 |

**为什么这样写**：7B 模型 FP16 权重约 14GB，加载后还需预留 KV Cache，因此 16GB 只是**7B 这一具体场景**的经验下限，24GB 才能舒适运行并支持一定并发。更小的模型可以在显存更小的设备上完成环境验证；显存门槛应随所选模型规模而定，不能作为"安装 vLLM 的统一最低要求"。

（个性化提示：学习风格：hands-on）

## 块 11

### 4.2 云端环境配置参考

| 场景 | 推荐实例 | 可运行模型 |
|------|---------|-----------|
| 低成本实验 | AutoDL RTX 3090（24GB）×1 | 7B（FP16，单卡） |
| 标准推理 | AWS g5.xlarge（A10G 24GB） | 7B（FP16）；13B 需量化（AWQ/GPTQ 等）才能单卡放下，FP16 的 13B 单卡放不下（见 4.3） |
| 高并发服务 | AWS g5.12xlarge（4×A10G） | 70B 模型（TP=4） |
| 超大模型 | AWS p4d.24xlarge（8×A100） | 数百亿至千亿级模型（需张量并行 TP，视模型与显存可能还要配合流水线并行 PP） |

> **注**：表中"可运行模型"默认按 **FP16 权重**估算，且未把长上下文的 KV Cache 开销算满。凡是写"某卡能跑某模型"的结论都隐含**精度/量化前提**：同一张 24GB 卡，跑 13B **FP16** 会因权重（约 26GB，见 4.3）超出显存而放不下，但**量化到 4bit（AWQ/GPTQ）** 后 13B 就有机会单卡运行。具体仍以 4.3/4.4 的估算和实测显存为准。

（个性化提示：学习风格：hands-on）

## 块 12

### 4.3 模型规模与显存对照表

下表只给出 **FP16 权重**的粗略量级（权重相对确定），KV Cache 单列去掉——因为它高度依赖 KV 头数、层数、head_dim、缓存精度和序列长度，用一个固定数字概括很容易误导（见 4.4）：

| 模型规模 | FP16 权重（粗略） | 单卡 16GB | 单卡 24GB | 单卡 80GB |
|---------|----------|----------|----------|----------|
| 7B | ~14 GB | ⚠️ 紧张（KV Cache 余量小） | ✅ 可运行 | ✅ 高并发 |
| 13B | ~26 GB | ❌ 无法加载 | ⚠️ 紧张 | ✅ 可运行 |
| 70B | ~140 GB | ❌ 无法加载 | ❌ 无法加载 | ⚠️ 需 TP≥2 |

**为什么这样写**：能否跑起来不只看权重，还要看 KV Cache 等运行开销，而 KV Cache 与模型结构、精度、上下文长度强相关。表中"紧张/可运行"是量级判断，实际请结合 4.4 的公式估算 KV Cache，并以 vLLM 启动日志里实际分配的显存为准，再用 `--gpu-memory-utilization` 调整。

（个性化提示：学习风格：hands-on）

## 块 13

### 4.4 KV Cache 显存怎么估算（重要）

不要直接套用网上的"某模型 KV Cache = 固定 X GB"。这类数字往往没有正确体现 **GQA/MQA 的 KV 头数**、层数、head_dim、缓存精度与序列长度，很容易偏大或偏小。更稳妥的做法是用公式估算，再和实测对照。

**单请求 KV Cache 字节数的粗略公式**：

```text
单请求 KV Cache 字节数
≈ 2 × 层数(num_hidden_layers)
     × KV头数(num_key_value_heads)
     × head_dim
     × token 数(序列长度)
     × 每元素字节数(FP16=2, FP8=1)
```

其中开头的 `2` 表示 Key 和 Value 各一份。**对于 GQA/MQA 模型，必须用 KV 头数 `num_key_value_heads`，而不是 Query 头数** `num_attention_heads`——很多模型的 KV 头数远小于 Query 头数，直接用 Query 头数会把 KV Cache 估算成数倍大小。

推荐的估算流程：

1. 打开模型的 `config.json`，读取 `num_hidden_layers`、`num_key_value_heads`（若无则用 `num_attention_heads`）、`hidden_size` / `num_attention_heads` 推出 `head_dim`。
2. 代入上面的公式，乘以你计划支持的最大序列长度和并发请求数，得到理论估算。
3. 启动 vLLM 后，对照启动日志中实际为 KV Cache 分配的显存（以及可容纳的 token/block 数），以**实测为准**修正估算。

---

## 第五章：安装验证——确认 vLLM 就绪

（个性化提示：学习风格：hands-on）

## 块 14

### 5.1 检查 vLLM 与 PyTorch 版本（①②）

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**预期输出**（下面的版本号仅为示例，实际会随你安装的 vLLM/PyTorch/CUDA 版本不同而变化）：
```
0.6.1          # ← 你安装的 vLLM 版本
2.4.0+cu121    # ← 对应的 PyTorch 版本与 CUDA 后端
True
```

**为什么这样写**：`torch.cuda.is_available()` 返回 `True` 说明 PyTorch 能识别 GPU。如果返回 `False`，说明安装了 CPU 版 PyTorch，vLLM 无法使用 GPU 推理。

（个性化提示：学习风格：hands-on）

## 块 15

### 5.2 最小推理测试（②）

创建 `test_vllm.py`：

```python
from vllm import LLM, SamplingParams

# 加载一个小模型（首次会自动下载）
llm = LLM(model="facebook/opt-125m")

# 定义采样参数
sampling_params = SamplingParams(temperature=0.8,

## 块 16

### 5.3 验证在线服务（②）

启动 OpenAI 兼容的 API 服务：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
```

测试服务：
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

**为什么这样写**：`vllm serve` 是 vLLM 的一键部署命令，自动启动兼容 OpenAI API 的服务器。验证服务能正常响应，说明整个推理链路（模型加载 → 请求处理 → 文本生成）都已就绪。

---

## 第六章：关键参数初探——`gpu_memory_utilization`

（个性化提示：学习风格：hands-on）

## 块 17

```markdown
## 为什么需要控制显存利用率？

在使用vLLM进行模型推理时，显存利用率的控制非常重要。默认情况下，vLLM会使用GPU显存的90%来存储模型权重和KV缓存。然而，在以下情况下，需要手动调整显存利用率：

- **共享GPU**：如果在同一台机器上有其他服务占用显存，建议降低显存利用率。
- **长上下文推理**：为长序列推理预留足够的KV缓存空间，建议将显存利用率降低到80%。
- **多模型加载**：如果同时加载多个LoRA适配器，显存

## 块 18

# vLLM 部署：环境准备篇

## 课程目标
掌握 vLLM 模型的安装与部署环境配置，确保其在高性能推理服务中的稳定运行。

## 课程背景
vLLM 是一个专为高吞吐量、低延迟的大模型推理引擎，广泛应用于工业场景。为了实现其最佳性能，需要在合适的硬件环境和正确的参数配置下运行。本课时将详细讲解 vLLM 的安装和环境准备，帮助你快速搭建一个高效的推理服务环境。

---

## **学习者画像**
- **学习目标**：vLLM

## 块 19

# 生成策略：贪心搜索与束搜索的实践与理解

## 第一章：生成策略的本质——从概率分布到文本输出

大语言模型在推理过程中，每一步都会输出一个概率分布，即词表中每个token作为下一个词的可能性。例如，当输入为"今天天气"时，模型可能会输出以下概率分布：

```
  "很好"  → 0.35
  "不错"  → 0.28
  "真棒"  → 0.15
  "糟糕"  → 0.08
  ...（词表中所有 token 的概率之和 = 1.0）
```

**生成策略**就是从这个概率分布中，如何选出下一个 token。不同的选择方式，造就了完全不同的输出风格。

### 贪心搜索（Greedy Decoding）

**核心行为**：每一步都选择当前概率最高的 token 作为输出。

**输出特点**：
- **确定性高**：输出通常非常稳定，不会出现大量变数。
- **保守性强**：倾向于选择概率较高的词，避免出现低概率或罕见的token。
- **可能重复**：在重复的语境下，可能会导致输出的重复性。

**应用场景**：
- **新闻摘要生成**：适合需要高度确定性和准确性的场景，如生成新闻标题或段落摘要。
- **对话系统**：在对话过程中，用户通常希望得到明确、直接的回应，贪心搜索能够满足这种需求。

### 束搜索（beam search）

**核心行为**：在每一步中，维护多个候选序列（候选路径），并在有限的束宽（beam width）下，优先选择最优的候选路径。

**输出特点**：
- **质量更高**：由于考虑了多个候选路径，生成的文本通常质量更高，包含更多的优质选项。
- **显存占用大**：需要同时维护多个候选路径，对显存资源有一定要求。
- **多样性提升**：在面对复杂或模糊的输入时，束搜索能够更好地捕捉多种

## 块 20

### 2.1 核心原理：Argmax 一条路走到黑

贪心搜索是最简单的生成策略。它的规则只有一条：

> **每一步，选择当前概率最高的那个 token。**

```
时间步 1: "今天天气" → 选 "很好"(0.35) → 序列: "今天天气很好"
时间步 2: "今天天气很好" → 选 "啊"(0.42) → 序列: "今天天气很好啊"
时间步 3: "今天天气很好啊" → 选 "！"(0.55) → 序列: "今天天气很好啊！"
```

**为什么这样写**：贪心搜索的数学表达就是 $\text{token}_t = \arg\max P(\text{token} | \text{context})$。它不考虑未来的可能性，只关注当下的最优选择。

（个性化提示：学习风格：hands-on）

## 块 21

# 掌握模型生成策略：贪心搜索（Greedy Decoding）

## 2.2 贪心搜索的优缺点

### 优点
1. **强确定性**：在相同的硬件配置和软件环境中，相同的输入通常会得到相同的输出。这种一致性使得贪心搜索在需要稳定性和可重复性的工作场景中表现出色。
2. **速度最快**：由于贪心搜索只需要维护一条当前最优的路径，因此其计算速度非常快，适合需要实时响应的应用场景。
3. **显存最小**：相比其他复杂的生成算法，贪心搜索只需要维护当前生成序列，因此占用的显存资源非常少，适合设备资源有限的场景。

### 缺点
1. **局部最优陷阱**：贪心搜索在每一步都选择当前看起来最优的选项，但这种

## 块 22

### 3.1 核心原理：同时走多条路，最后选最好的

束搜索（Beam Search）是对贪心搜索的改进。它不再只保留一个序列，而是同时维护 **$k$ 个候选序列**（$k$ 称为 beam width）。

```
Beam Width = 2 的示例：

Step 1: "今天天气"
  候选1: "很好"(0.35) → 得分: log(0.35) = -1.05
  候选2: "不错"(0.28) → 得分: log(0.28) = -1.27

Step 2: 从两个候选分别扩展
  候选1+"啊"(0.42) → 得分: -1.05 + log(0.42) = -1.93
  候选1+"呢"(0.30) → 得分: -1.05 + log(0.30) = -2.25
  候选2+"啊"(0.35) → 得分: -1.27 + log(0.35) = -2.32
  候选2+"真"(0.25) → 得分: -1.27 + log(0.25) = -2.66
  
  保留 Top 2: 候选1+"啊"(-1.93), 候选1+"呢"(-2.25)

Step 3: 继续扩展并保留 Top 2...
最终：从所有完成的候选中选得分最高的输出
```

**为什么这样写**：束搜索通过"多路探索"缓解了贪心搜索过早陷入局部最优的问题。它用对数概率之和衡量序列质量，在保留的 $k$ 条候选里选出得分最高的输出。需要注意的是，束搜索只保留有限数量（束宽 $k$）的候选，被早期剪掉的序列后续仍可能成为更优序列，因此它是一种**近似搜索**，**并不保证找到全局最优结果**——只是相比贪心考虑了更多候选。

（个性化提示：学习风格：hands-on）

## 块 23

### 3.2 束搜索的核心参数（①）

> **重要版本提示**：**当前主流 vLLM（v0.6.3 起）已把 `use_beam_search` 从 `SamplingParams` 移除**，束搜索改由独立入口 `llm.beam_search()` 配合 `BeamSearchParams` 调用（束宽用 `beam_width` 表示）。本章正文与示例统一采用**新版入口**作为可运行主路径；旧版 `SamplingParams(use_beam_search=True)` 的写法仅作历史对照，集中放在文末【附录：旧版束搜索接口兼容说明】。实际调用请以你所安装的 vLLM 版本文档为准。

下表参数以新版 `BeamSearchParams` / 束搜索语义为准：

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `beam_width` | 束宽度（beam width），即同时维护的候选序列数（旧版 `SamplingParams` 里叫 `best_of`） | 2~5 |
| `length_penalty` | 长度惩罚，控制对长序列的偏好 | 1.0（不惩罚） |
| `early_stopping` | 提前停止条件 | `False` / `True` / `"never"` |

**为什么这样写**：
- `beam_width` 越大，搜索空间越大，输出质量可能越高，但**显存和计算量线性增长**
- `length_penalty > 1.0` 鼓励生成更长的序列；`< 1.0` 鼓励更短的序列
- `early_stopping` 控制何时停止搜索（①）

（个性化提示：学习风格：hands-on）

## 块 24

### 3.3 `early_stopping` 的三种模式（①）

在新版 vLLM 中，束搜索通过独立入口 `llm.beam_search()` 调用，束宽和停止策略放在 `BeamSearchParams` 里（下面是本章推荐的主用法）：

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

beam_params = BeamSearchParams(
    beam_width=4,        # 束宽（同时维护 4 个候选序列）
    max_tokens=100,
    # early_stopping 语义：
    #   False  → 启发式停止（当不太可能找到更好候选时提前停止）
    #   True   → 只要生成够 beam_width 个完整候选就停止
    #   "never"→ 经典束搜索，直到所有候选达到 max_tokens 才停止
)

outputs = llm.beam_search(
    prompts=["今天天气"],
    params=beam_params,
)
```

**为什么这样写**：`early_stopping=False` 使用启发式策略，在"不太可能找到更好候选"时提前停止，平衡质量和速度；`"never"` 是最严格的经典算法，但计算量最大。旧版 `SamplingParams(use_beam_search=True, ...)` 的等价写法见文末【附录：旧版束搜索接口兼容说明】，此处不再作为主用法。

（个性化提示：学习风格：hands-on）

## 块 25

### 3.4 束搜索的优缺点

| 优点 | 缺点 |
|------|------|
| **质量更高**：通过多路探索缓解局部最优 | **显存占用大**：需同时维护 $k$ 个序列的 KV Cache |
| **确定性较强**：相同软硬件与配置下相同输入通常产生相同输出 | **速度更慢**：每步需计算和排序 $k \times V$ 个候选 |
| **适合结构化输出**：翻译、摘要、代码 | **不适合开放式生成**：长文本创意写作效果一般 |

**生活化比喻**：束搜索就像探险队**同时派出 $k$ 个小队分头探路**，每走一段就评估各队进度，淘汰最差的，保留最好的，最终选择最先到达目的地的队伍。

（个性化提示：学习风格：hands-on）

## 块 26

### 3.5 在 vLLM 中启用束搜索（①④）

**新版接口（v0.6.3+，推荐主用法）——通过独立入口 `llm.beam_search()` + `BeamSearchParams`：**

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams  # 具体导入路径以所装版本为准

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

beam_params = BeamSearchParams(
    beam_width=4,           # 束宽度（同时维护 4 个候选，对应旧版的 best_of）
    max_tokens=100,
    length_penalty=1.0,
)
# 新接口接收 prompt 列表，返回结果的字段名也可能随版本不同，
# 请对照你所装版本的示例代码调整
outputs = llm.beam_search(
    prompts=["请将以下中文翻译成英文：你好世界"],
    params=beam_params,
)
```

**为什么这样写**：
- `beam_width=4` 表示内部用 4 个 beam 搜索，vLLM 会自动管理这些序列的并行计算
- 束搜索在 vLLM 内部会把多个 beam 作为并行的"子序列"来处理（①）
- 旧版（约 v0.6.2 及更早）曾通过 `SamplingParams(use_beam_search=True, best_of=4, ...)` + `llm.generate()` 启用束搜索，该字段在 v0.6.3+ 已移除，传入会报 `TypeError`；等价写法见文末【附录：旧版束搜索接口兼容说明】，此处不再作为主用法。新旧版本的**原理一致，差别主要在调用入口和参数命名**（`best_of` → `beam_width`）。

---

## 第四章：采样策略简介——与确定性生成的对比

（个性化提示：学习风格：hands-on）

## 块 27

### 4.1 为什么需要随机采样？

贪心搜索和束搜索都是**确定性**策略——在相同软硬件与执行配置下，相同输入通常产生相同输出。但在很多场景（如聊天、创意写作、头脑风暴），我们希望模型输出**多样化**。

随机采样策略通过引入"随机性"来实现多样性：

| 策略 | 核心机制 | 随机性来源 |
|------|---------|-----------|
| **Temperature 采样** | 对 logits 除以一个温度系数 | 温度越高，分布越"平坦"，随机性越强 |
| **Top-k 采样** | 只从概率最高的 k 个 token 中采样 | 限制候选池大小 |
| **Top-p（核采样）** | 从累积概率达到 p 的最小 token 集合中采样 | 动态调整候选池 |

**为什么这样写**：这三种采样策略是"随机性生成"的代表，与贪心/束搜索的"确定性生成"形成鲜明对比。理解它们的区别，才能在实际业务中做出正确选型。

### 4.2 确定性 vs 随机性：选型对照表

| 场景 | 推荐策略 | 理由 |
|------|---------|------|
| 数学计算、代码生成 | **贪心搜索** | 需要唯一正确答案 |
| 机器翻译、文本摘要 | **束搜索** | 需要高质量、结构化输出 |
| 聊天对话、创意写作 | **Temperature + Top-p 采样** | 需要多样性、避免重复 |
| 事实问答、知识抽取 | **贪心搜索** | 需要准确、可复现 |

---

## 第五章：vLLM 中的配置实践

（个性化提示：学习风格：hands-on）

## 块 28

### 5.1 `SamplingParams` 完整配置示例（①）

```python
from vllm import SamplingParams

# 场景1：贪心搜索（确定性）
greedy_params = SamplingParams(
    temperature=0,          # < 1e-5 触发贪婪解码
    max_tokens=256,
)

# 场景3：随机采样（多样性）
sampling_params = SamplingParams(
    temperature=0.8,        # 温度>0，启用随机采样
    top_p=0.95,             # 核采样
    top_k=50,               # 只考虑前50个token
    max_tokens=256,
)
```

> **场景2：束搜索**不再通过 `SamplingParams` 配置。新版请用 `BeamSearchParams` + `llm.beam_search()`（见 3.3 节的主用法示例）：
>
> ```python
> from vllm.sampling_params import BeamSearchParams
>
> beam_params = BeamSearchParams(beam_width=4, max_tokens=256, length_penalty=1.0)
> outputs = llm.beam_search(prompts=["翻译：Hello World"], params=beam_params)
> ```
>
> 旧版 `SamplingParams(use_beam_search=True, best_of=4, ...)` 的写法见文末附录，不要在新版环境里使用。

（个性化提示：学习风格：hands-on）

## 块 29

### 5.2 服务启动与请求参数（④）

采样策略统一在**客户端请求体**里指定，`vllm serve` 只负责把模型跑起来：

```bash
# 正常启动服务，不在启动命令里写采样参数
vllm serve Qwen/Qwen2.5-7B-Instruct
```

```bash
# 贪心搜索：请求体里传 temperature=0
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "翻译：Hello World",
        "max_tokens": 100,
        "temperature": 0
    }'
```

> **束搜索的服务端调用**：不同 vLLM 版本对"通过 OpenAI 兼容接口做束搜索"的支持差异较大，旧版曾支持在请求体里传 `use_beam_search`/`best_of`，但新版可能不再接受该字段。因此**服务端束搜索请以你所部署版本的 API 文档为准**；若需要稳定使用束搜索，推荐走离线的 `llm.beam_search()`（见 3.3 节）。旧版请求体写法见文末附录，不作为主推路径。

（个性化提示：学习风格：hands-on）

## 块 30

### 5.3 提升输出可复现性的完整 checklist（④）

如果你需要**尽可能可复现**的输出，可以检查以下设置（注意：这些设置能显著提升同环境下的确定性，但无法保证跨版本、跨硬件的绝对一致）：

```python
sampling_params = SamplingParams(
    temperature=0,          # 关闭随机采样
    seed=42,                # 可选：固定随机种子（双重保险）
    top_p=1.0,              # 会被自动忽略，但显式设置更清晰
    top_k=-1,               # 同上
    max_tokens=100,
)
```

**为什么这样写**：`seed` 在 `temperature=0` 时理论上不需要（因为已经是确定性 argmax），但在混合 batch 场景下（部分请求贪心、部分请求随机），固定种子有助于提升整个系统的可复现性。要真正对齐结果，还需固定 vLLM/PyTorch/CUDA/模型版本、GPU 型号、并行配置和 batch 组成等因素；这些一旦变化，浮点归约顺序不同仍可能带来细微差异。

---

## 第六章：电脑配置批注

本章节涉及 vLLM 推理服务的本地部署与实验。

（个性化提示：学习风格：hands-on）

## 块 31

### 6.1 本地环境配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 1 张 NVIDIA GPU，显存 **≥16GB** | 1 张 RTX 4090（24GB）/ A100（40GB/80GB） |
| **CPU** | 8 核 | 16 核+ |
| **内存** | 32GB | 64GB+ |
| **磁盘** | 100GB SSD | 500GB+ NVMe SSD |
| **系统** | Linux（Ubuntu 20.04+）或 WSL2 | 原生 Linux（Ubuntu 22.04+） |
| **是否支持 CPU 运行** | ⚠️ 支持但极慢（仅调试） | — |

（个性化提示：学习风格：hands-on）

## 块 32

### 6.2 束搜索的额外显存需求

| beam width | 额外显存占用 | 说明 |
|-----------|-------------|------|
| 1（贪心） | 基准 | 只维护 1 个序列 |
| 2 | 约 2× KV Cache | 大致同时维护 2 个序列的 KV Cache |
| 4 | 约 4× KV Cache | KV Cache 随束宽近似线性增长 |
| 8 | 约 8× KV Cache | 束宽较大时显存压力显著，需要较大显存的卡才能舒适运行 |

**为什么这样写**：束搜索需要为多个 beam 各自维护 KV Cache，因此其 KV Cache 部分的显存占用与 `best_of`（beam width）**近似线性正相关**（上表倍数为量级示意，实际还受序列长度、模型结构、框架实现影响）。在 24GB 显存上运行 7B 模型时，较大的 `best_of` 可能接近上限，建议从 `best_of=2` 开始逐步尝试。

（个性化提示：学习风格：hands-on）

## 块 33

### 6.3 云端环境配置参考

#### 6.3.1 贪心搜索与束搜索策略选择
在模型推理部署中，选择合适的生成策略至关重要。以下是两种主要策略及其适用场景：

1. **贪心搜索（Greedy Search）**
   - **特点**：每一步都选择概率最高的下一个token，`temperature=0`时表现最佳。
   - **优点**：计算效率高，显存占用低。
   - **缺点**：容易陷入局部最优，输出可能单调重复。

2. **束搜索（Beam Search）**
   - **特点**：同时维护多个候选序列，通过`beam_width`参数控制候选数量。
   - **优点**：减少重复输出，生成内容更具多样性和质量。
   - **缺点**：显存占用较高，计算复杂度增加。

#### 6.3.2 应用场景配置建议
根据场景选择合适的参数组合：

| 场景         | 推荐实例       | 生成策略         |
|--------------|-----------------|------------------|
| 低成本实验  | AutoDL RTX 3090（24GB） | 贪心搜索（`temperature=0`） + `best_of=2` |
| 标准推理    | AWS g5.xlarge（A10G 24GB） |

## 块 34

# 掌握模型推理部署（vLLM）：贪心搜索与束搜索策略

## 6.3 云端环境配置参考

### 学习目标
通过本次学习，你将掌握贪心搜索和束搜索策略，学会在模型推理部署中合理配置模型推理参数，以实现模型输出的多样性和质量的平衡。

### 学习方式
本次学习采用实践导向的方式，通过实际操作熟悉贪心搜索和束搜索的参数配置，结合实际案例理解其应用场景。

### 学习证据
- [贪心搜索与束搜索的区别](#贪心搜索与束搜索的区别)
- [束搜索参数设置与显存管理](#束搜索参数设置与显存管理)

---

## 贪心搜索与束搜索的区别

### 贪心搜索
- **应用场景**：适用于需要唯一正确答案的场景，如翻译任务。
- **参数**：
  - `temperature=0`：开启贪心搜索，模型将选择概率最高的下一个词。
  - `best_of=1`：指定生成文本的长度，模型将输出固定长度的文本。
- **优点**：输出稳定，质量高。
- **缺点**：缺乏多样性，无法满足需要多样化的

## 块 35

```markdown
# 随机采样的定位——从"唯一答案"到"多种可能"

## 1.1 为什么需要随机采样？

在上一章中，我们学习了**贪心搜索**和**束搜索**，这两种方法都是**确定性较强**的生成策略。它们在相同输入下通常会输出相同的结果，这在数学计算、代码生成和机器翻译等场景中非常有用。然而，在许多实际应用中，**多样性**才是关键。

| 场景 | 为什么需要多样性？ | 确定性策略的问题 |
|------|------------------|---------------|
| **聊天对话** | 同样的问候不应得到一模一样的回复 | 贪心搜索会让机器人像"复读机" |
| **创意写作** | 诗歌、故事需要不同风格和表达 | 束搜索输出单调、缺乏惊喜 |
| **头脑风暴** | 需要多角度、多思路的建议 | 确定性输出永远走同一条路 |
| **角色扮演** | 不同角色

## 块 36

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

（个性化提示：学习风格：hands-on）

## 块 37

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

（个性化提示：学习风格：hands-on）

## 块 38

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

（个性化提示：学习风格：hands-on）

## 块 39

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

（个性化提示：学习风格：hands-on）

## 块 40

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

（个性化提示：学习风格：hands-on）

## 块 41

### 3.2 Top-k 的取值建议（①④）

| Top-k 值 | 效果 | 适用场景 |
|---------|------|---------|
| **k=1** | 等价于贪心搜索 | 需要唯一正确答案 |
| **k=10** | 非常保守，只考虑最可能的 10 个词 | 技术文档、代码 |
| **k=40~50** | 社区常见经验取值，兼顾质量与多样性（非官方默认，需按任务实测） | 通用对话、翻译 |
| **k=100** | 较宽松，允许更多词汇变化 | 创意写作 |
| **k=-1 或 0** | 禁用 Top-k，考虑全部词表 | 特殊实验场景 |

**为什么这样写**：Top-k 的"k"与词表大小无关。无论词表是 3 万还是 10 万，Top-k=50 都只保留概率最高的 50 个 token。这使得 Top-k 成为一个**与模型规模无关**的稳健参数。

（个性化提示：学习风格：hands-on）

## 块 42

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

（个性化提示：学习风格：hands-on）

## 块 43

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

（个性化提示：学习风格：hands-on）

## 块 44

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

（个性化提示：学习风格：hands-on）

## 块 45

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

（个性化提示：学习风格：hands-on）

## 块 46

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

（个性化提示：学习风格：hands-on）

## 块 47

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

（个性化提示：学习风格：hands-on）

## 块 48

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

（个性化提示：学习风格：hands-on）

## 块 49

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

（个性化提示：学习风格：hands-on）

## 块 50

### 6.2 服务启动与采样参数的设置方式（④）

采样参数（`temperature`/`top_p`/`top_k`/`min_p` 等）**不建议在启动命令里写死**——不同 vLLM 版本对"启动时设默认采样参数"的支持差异很大，很多版本根本没有对应开关。最通用、最不受版本影响的做法是：`vllm serve` 只负责把模型服务跑起来，采样参数在**每次客户端请求**（`SamplingParams` 或 HTTP 请求体）中传入。

```bash
# 只启动服务，不在此处写采样参数
vllm serve Qwen/Qwen2.5-7B-Instruct
```

随后在请求里按场景传采样参数（见下方 6.3）。如需服务侧统一默认值，请先用 `vllm serve --help` 确认你所装版本是否提供相应选项及其准确名称，不要照抄他处的参数名。

（个性化提示：学习风格：hands-on）

## 块 51

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

（个性化提示：学习风格：hands-on）

## 块 52

### 7.1 本地环境配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 1 张 NVIDIA GPU，显存 **≥16GB** | 1 张 RTX 4090（24GB）/ A100（40GB/80GB） |
| **CPU** | 8 核 | 16 核+ |
| **内存** | 32GB | 64GB+ |
| **磁盘** | 100GB SSD | 500GB+ NVMe SSD |
| **系统** | Linux（Ubuntu 20.04+）或 WSL2 | 原生 Linux（Ubuntu 22.04+） |
| **是否支持 CPU 运行** | ⚠️ 支持但极慢（仅调试） | — |

（个性化提示：学习风格：hands-on）

## 块 53

```markdown
# vLLM 部

## 块 54

1. **从通用配置开始**——`temperature=0.8, top_p=0.95, top_k=50` 是业界最稳妥的起点
2. **调参时只动一个变量**——同时改 temperature 和 top_p 会让你无法判断是哪个参数在起作用
3. **固定 seed 做 A/B 测试**——相同的 prompt + 相同的 seed = 可对比的输出
4. **低温不等于高质量**——temperature=0.1 可能让输出变得机械和重复，0.3~0.5 往往更自然
5. **Top-p 比 Top-k 更推荐作为主要过滤手段**——Top-p 的动态适应性更好，Top-k 作为辅助保险
6. **监控重复率**——如果输出频繁重复，先调 `repetition_penalty`，再调 temperature

---

## 下一步预告

掌握了 Temperature、Top-p、Top-k 采样参数后，你已经能够灵活控制大模型输出的随机性与多样性。接下来，你将进入 **"理解 vLLM 的 PagedAttention 与 Continuous Batching 机制"** 的学习——深入理解 vLLM 如何通过"分页管理 KV Cache"解决显存碎片问题，以及如何通过"连续批处理"在高并发下把推理吞吐量提升数倍到十几倍（具体幅度视模型与负载而定）。这是从"会调参"进化为"懂引擎"的关键一步，继续加油！

（个性化提示：学习风格：hands-on）

## 块 55

# 理解vLLM的PagedAttention与Continuous Batching机制

## 第一章：PagedAttention 的定位——为什么需要分页管理 KV Cache？

### 1.1 传统 KV Cache 管理的两大浪费

在大模型自回归推理中，每生成一个新 token，都需要缓存该 token 对应的 Key 和 Value 向量（即 KV Cache）。随着序列增长，KV Cache 不断膨胀，成为推理显存的主要消耗者。

在 vLLM 出现之前，传统推理框架（如 FasterTransformer、ONNX Runtime）采用**静态预分配**策略：

> **为每个请求预先分配一块按最大序列长度计算的连续显存空间。**

例如，模型支持的最大上下文为 2048 tokens，那么即使某个请求实际只生成 50 个 token，也要为其预留能装 2048 tokens 的连续 KV Cache 空间。

这导致了两种严重的显存浪费（①）：

1. **资源浪费**：为每个请求预留了远超实际需求的显存空间，导致内存利用率低下。
2. **性能瓶颈**：在

## 块 56

| 浪费类型 | 说明 | 类比 |
|---------|------|------|
| **内部碎片（Internal Fragmentation）** | 预留了 2048，实际只用了 50，剩余 1998 的空间被白白占用 | 租了 100 平米的办公室，只坐了 5 个人 |
| **外部碎片（External Fragmentation）** | 每个请求需要连续的大块显存，导致显存被切成大小不一的碎片，总空闲够但找不到足够大的连续空间 | 停车场有很多空位，但都是分散的小车位，大巴停不进去 |

**为什么这样写**：在这种静态预分配方案下，KV Cache 的显存利用率往往只能达到**几成左右**（不同来源常给出约 20%~40% 的经验区间，具体数值与序列长度分布、最大长度设置、请求模式强相关，仅为示意），相当一部分显存被碎片浪费。而显存是推理的命根子——可用于 KV Cache 的显存越多，能支持的并发请求数和上下文长度就越长。浪费显存 = 直接限制吞吐量和用户体验。

（个性化提示：学习风格：hands-on）

## 块 57

### 1.2 PagedAttention 的核心洞察

vLLM 的 PagedAttention 借鉴了操作系统管理物理内存的经典方案——**虚拟内存分页**：

> **操作系统不要求进程的虚拟地址空间在物理内存中连续存放。内存被切成固定大小的页（Page），按需分配给进程，通过页表记录逻辑地址到物理地址的映射。**

PagedAttention 将这一思想搬到 GPU 显存管理上：

| 操作系统概念 | PagedAttention 对应 |
|-------------|-------------------|
| 物理内存页（Page） | KV Cache 块（Block） |
| 页表（Page Table） | 块表（Block Table） |
| 进程逻辑地址空间 | 一个请求的 KV 序列 |

**为什么这样写**：这不是简单的类比，而是**架构级别的复用**。操作系统用分页解决了内存碎片问题，vLLM 用同样的思路解决了 KV Cache 的显存碎片问题。

---

## 第二章：PagedAttention 核心原理——块、块表与按需分配

（个性化提示：学习风格：hands-on）

## 块 58

### 2.1 Block 的定义（③）

PagedAttention 将 KV Cache 切成**固定大小的块（Block）**，默认每个 Block 存储 16 个 token 的 Key 和 Value：

```
一个 Block = 16 个 token 的 K/V 向量

用于"估算容量"的概念形状（不代表单个 Block 的真实内存布局）：
[num_layers, 2(K/V), num_heads, block_size=16, head_dim]
```

**为什么这样写**：`block_size=16` 是部分 vLLM CUDA 版本上常见的默认值，这个大小在 GPU 内存访问对齐和显存粒度之间取得平衡：太小会增加块表管理开销，太大会增加内部碎片。需要注意，**默认值与可选范围可能随引擎版本、硬件平台变化**，这里的 16 只是特定版本/平台的示例，不是永久固定规则；实际请以 `vllm serve --help` 或所装版本文档为准。

> **关于上面的"形状"**：`[num_layers, 2(K/V), num_heads, block_size, head_dim]` 是一个**便于估算 KV Cache 总容量的概念形状**（把各层、K/V 两份、注意力头都乘进去，方便算显存），**并不代表单个物理 Block 在显存里的真实内存布局**。实际实现中 KV Cache 通常**按层分别维护**（每层各自持有若干 Block），K 与 V 是否分开存放、`num_heads` 用的是 Query 头数还是 KV 头数（GQA/MQA 场景）、各维度的先后顺序等，都取决于具体 backend、精度与 vLLM 版本。这里给出该形状只是为了帮助你估算容量，不要据此推断底层张量的排布。

（个性化提示：学习风格：hands-on）

## 块 59

```markdown
# 2.2 Block Table——

## 块 60

### 2.3 内存分配与回收流程（③）

```
[请求到达]
    ↓
[估算所需 Block 数]
    ↓
[从 BlockAllocator 获取空闲 Block] → O(1) 从 free_list 弹出
    ↓
[追加到 block_table]
    ↓
[请求完成 / 生成结束]
    ↓
[释放所有 Block] → 归还到 free_list → O(1)
    ↓
[Block 立刻可被其他请求复用]
```

**为什么这样写**：BlockAllocator 维护一个空闲块链表（free_list），分配和回收都是 O(1) 操作。请求结束后 Block 立刻进池，新请求随时凑块加入，不存在"连续大空间找不到"的问题。

（个性化提示：学习风格：hands-on）

## 块 61

### 2.4 为什么解决了浪费（①）

| 问题 | 传统方案 | PagedAttention |
|------|---------|---------------|
| 内部碎片 | 预留 2048，用 50，浪费 1998 | 按需分配，50 token 只占 4 块，最多浪费最后一块的十几个 token |
| 外部碎片 | 需要连续大块，找不到 | 块固定大小，任何空闲块都能给任何请求用 |
| 显存利用率 | 较低（常见经验值约 20%~40%，示意） | **接近满载**（按需分配，浪费极小） |
| 并发能力 | 低 | 同显存下并发能力可显著提升（具体倍数视模型、序列长度与负载而定） |

**为什么这样写**：PagedAttention 把 KV Cache 显存利用率从"只有几成"提升到接近满载（上文的百分比区间为示意，实际取决于负载），这是 vLLM 吞吐量提升的**第一支柱**。

---

## 第三章：Continuous Batching——动态调度请求

（个性化提示：学习风格：hands-on）

## 块 62

### 3.1 静态批处理的两大浪费

在理解 Continuous Batching 之前，先看传统**静态批处理（Static Batching）**的问题：

```
【静态批处理】
时间 →
Batch 0: [请求A: "Hello world"]         → 请求A已完成，但必须等请求B
         [请求B: "Hi"] + [PAD]×10       → 请求B还在生成，请求A"陪跑"
         ↓ 整批必须等最慢的请求完成才释放

Batch 1: [请求C: "What is AI?"]
         [请求D: "A"] + [PAD]×9
         ↓ 同样的问题重复出现
```

两种浪费（①）：

- **等齐才开始**：凑够一个 batch 才能启动，先到的请求干等
- **等最慢才结束**：整批必须等生成最长的请求跑完，早完成的请求白白占用 GPU 算力和显存

**生活化比喻**：静态批处理就像**拼车必须坐满才发车、所有人到终点才让下车**。而真实场景中，请求陆续到达、生成长度差异巨大。

（个性化提示：学习风格：hands-on）

## 块 63

### 3.2 Continuous Batching 的核心思想

Continuous Batching（也叫 Iteration-level Batching 或 In-flight Batching）取消了"固定 batch"的概念，改为**动态池子**：

> **调度器在每一步生成（每个 token 迭代）都重新审视：谁完成了立刻踢出？有空位立刻把新请求塞进来？**

```
【Continuous Batching】
时间 →

迭代 1: [A_decode] [B_decode] [C_decode]     → 3 个请求同时处理
迭代 2: [A_decode] [D_prefill] [C_decode]     → B 完成退出，D 新加入
迭代 3: [E_prefill] [D_decode] [C_decode]     → A 完成退出，E 新加入
迭代 4: [E_decode] [D_decode] [F_prefill]     → C 完成退出，F 新加入

每一次迭代都保持 GPU 满载
```

**为什么这样写**："Batch"在 Continuous Batching 中不是一个固定的集合，而是**每步都在变化的动态组合**。GPU 几乎不空闲，没有"等齐"也没有"陪跑"的浪费。

（个性化提示：学习风格：hands-on）

## 块 64

### 3.3 Scheduler 调度器的核心逻辑（②③）

vLLM 的 Scheduler 在每一步执行以下决策：

```python
# 简化版核心逻辑（来自 vllm/v1/core/sched/scheduler.py）
class Scheduler:
    def __init__(self):
        self.waiting = deque()   # 新来的请求队列
        self.running = []        # 正在生成的请求列表

    def schedule(self, max_num_batched_tokens: int):
        scheduled = []
        token_budget = max_num_batched_tokens

        # Step 1: 优先继续处理正在运行的请求（decode）
        for seq_group in self.running:
            if token_budget > 0:
                scheduled.append(seq_group)
                token_budget -= 1   # 每个 decode 请求消耗 1 token

        # Step 2: 如果还有预算，从 waiting 中加入新请求（prefill）
        while self.waiting and token_budget > 0:
            new_req = self.waiting.popleft()
            needed = new_req.get_prompt_len()
            if needed <= token_budget:
                scheduled.append(new_req)
                token_budget -= needed
            else:
                break  # 预算不够，可以 chunked prefill（分块预填充）

        return scheduled
```

**为什么这样写**：
- **Decode 优先**：运行中的请求（生成下一个 token）优先级最高，保证低延迟
- **Prefill 填充**：有剩余预算时加入新请求的 prefill（一次性处理整个 prompt）
- **Chunked Prefill**：如果新 prompt 太长，可以拆成多块，分批处理（不阻塞 decode）

（个性化提示：学习风格：hands-on）

## 块 65

### 3.4 效果：GPU 利用率显著提升（②）

| 指标 | 静态批处理 | Continuous Batching |
|------|-----------|-------------------|
| GPU 利用率 | 偏低（常见经验区间约 40%~60%，示意） | **明显更高（可达八九成以上，视负载而定）** |
| 吞吐量提升 | 基准 | **数倍甚至更高**（高并发时收益最明显，具体倍数依模型/负载/对比基线而定） |
| 平均延迟 | 高（等齐+陪跑） | 低（随时进出） |
| 请求响应 | 一批一起返回 | 完成一个返回一个 |

**为什么这样写**：Continuous Batching 是 vLLM 吞吐量提升的**第二支柱**。与 PagedAttention 配合，共同构成了 vLLM 高性能的根基。

---

## 第四章：两大机制如何配合——Paging 让 Batching 成为可能

（个性化提示：学习风格：hands-on）

## 块 66

### 4.1 关键洞察：调度的灵活性建立在显存管理的灵活性之上（①）

这是最容易被忽视、也是面试中最常考的点：

> **Continuous Batching 的"随时进出"，依赖 PagedAttention 的"灵活显存管理"。**

**为什么？**

如果还是传统方案的"预留连续大块显存"：
- 请求 A 走了，留下一个特定大小的连续空洞
- 新请求 B 的 KV Cache 不一定能塞进这个空洞（大小不匹配）
- 外部碎片导致"总空闲够，但塞不进新请求"

PagedAttention 解决了这个问题：
- 请求 A 的 Block 被逐个释放回 free_list
- 每个 Block 大小固定（16 tokens），任何新请求都能凑块使用
- 请求一结束，Block 立刻进池 → 新请求随时加入

**一句话总结**：Paging 管"显存怎么高效分配回收"，Batching 管"请求怎么高效调度进出"，**后者建立在前者之上**。

（个性化提示：学习风格：hands-on）

## 块 67

### 4.2 配合工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│  PagedAttention（显存管理层）                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Block 23   │    │  Block 45   │    │  Block 7    │     │
│  │  (请求A)    │    │  (请求A)    │    │  (请求B)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↑                    ↑                  ↑           │
│    free_list ←── 请求完成 ──→ 释放 Block ──→ 新请求获取    │
└─────────────────────────────────────────────────────────────┘
                            ↓ 支撑
┌─────────────────────────────────────────────────────────────┐
│  Continuous Batching（调度层）                                │
│  迭代1: [A_decode] [B_decode]                                │
│  迭代2: [A_decode] [C_prefill]  ← B完成，Block释放，C加入    │
│  迭代3: [D_prefill] [C_decode]  ← A完成，Block释放，D加入    │
└─────────────────────────────────────────────────────────────┘
```

---

## 第五章：Prefix Caching——分页管理的额外红利（③）

（个性化提示：学习风格：hands-on）

## 块 68

### 5.1 块共享的原理

既然 KV Cache 被切成了固定大小的 Block，那么**多个请求如果有相同的前缀**（如相同的 system prompt），这段前缀的 KV 向量是完全一样的。

传统方案：每个请求各存一份 → N 个请求存 N 份  
PagedAttention：多个请求**共用同一组物理 Block** → N 个请求存 1 份

```
请求1: [System Prompt] + "请翻译Hello"
请求2: [System Prompt] + "请翻译World"
请求3: [System Prompt] + "请翻译AI"

传统方案：System Prompt 的 KV 存 3 份
PagedAttention + Prefix Caching：System Prompt 的 KV 只存 1 份，3 个请求共享
```

**为什么这样写**：Prefix Caching 通过哈希（`hash(token_ids[0:block_size])`）检测是否已有相同前缀的 Block，有则直接引用，O(1) 命中检测。这在 RAG、Chat 等**共享前缀占比高**的多轮对话场景中能带来可观的额外吞吐提升（具体幅度取决于前缀长度占比与命中率，前缀越长、命中越多收益越大，需以实测为准）。

（个性化提示：学习风格：hands-on）

## 块 69

### 5.2 后缀写入与 Copy-on-Write 机制（③）

多个请求共享同一段前缀 Block 后，各自继续生成时会产生不同的后缀 token。这里需要分清两种情况：

- **已缓存的完整前缀 Block 保持只读**，可以被多个请求安全共享，不需要复制。
- **新生成的后缀**通常写入各请求**自己新申请的私有 Block**，天然互不影响，这一步本身并不需要 Copy-on-Write。
- **Copy-on-Write（写时复制）更典型地出现在"确实需要修改某个被共享的 Block"的场景**，例如并行采样（`n>1`）或束搜索中，多条序列共享同一个尚未写满的 Block、却要各自写入不同 token 时。此时才复制该共享 Block、生成私有副本再写入：

```
1. 需要修改某个仍被多个序列共享的 Block 时，才复制它 → 创建私有副本
2. 在私有副本上写入新 token 的 K/V
3. 原共享 Block 的引用计数减 1
```

**为什么这样写**：Prefix Caching 的常态是"只读共享前缀 + 各写各的私有后缀 Block"，并不会每次生成都触发 Copy-on-Write。只有当多条序列需要在**同一个共享 Block**上写入不同内容时，写时复制才登场，用来保证共享只发生在只读部分、写入互不污染。把 COW 说成"共享前缀一进入生成阶段就必然发生"是不准确的。

---

## 第六章：电脑配置批注

本章节涉及 vLLM 推理引擎的核心机制理解，本地和云端场景均需考虑。

（个性化提示：学习风格：hands-on）

## 块 70

### 6.1 本地环境配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 1 张 NVIDIA GPU，显存 **≥16GB** | 1 张 RTX 4090（24GB）/ A100（40GB/80GB） |
| **CPU** | 8 核 | 16 核+ |
| **内存** | 32GB | 64GB+ |
| **磁盘** | 100GB SSD | 500GB+ NVMe SSD |
| **系统** | Linux（Ubuntu 20.04+）或 WSL2 | 原生 Linux（Ubuntu 22.04+） |
| **是否支持 CPU 运行** | ⚠️ 支持但极慢（仅调试） | — |

（个性化提示：学习风格：hands-on）

## 块 71

### 6.2 云端环境配置参考

| 场景 | 推荐实例 | PagedAttention 效果 |
|------|---------|-------------------|
| 低成本实验 | AutoDL RTX 3090（24GB）×1 | 7B 模型，并发 8~16 |
| 标准推理 | AWS g5.xlarge（A10G 24GB） | 7B~13B 模型，并发 16~32 |
| 高并发服务 | AWS g5.12xlarge（4×A10G） | 70B 模型（TP=4），并发 64+ |
| 超长上下文 | AWS p4d.24xlarge（8×A100） | 128K 上下文，Prefix Caching 效果显著 |

（个性化提示：学习风格：hands-on）

## 块 72

### 6.3 关键参数与吞吐关系

在理解vLLM的PagedAttention机制及其与Continuous Batching的协同工作时，我们需要关注几个关键参数及其对模型吞吐量的影响。这些参数的设置直接影响模型的效率和性能，因此需要根据实际情况进行调整。

#### 1. **PagedAttention机制的核心思想**
PagedAttention是一种机制，借鉴自操作系统中的虚拟内存分页机制。它将模型的KV缓存（Key-Value cache）划分为固定大小的块（Block），每个块存储一定数量的token。通过这种机制，模型能够更高效地管理显存资源，减少碎片化的内存占用，从而提高并行处理能力。

- **Block大小（block_size）**：每个块存储的token数量。不同版本的vLLM可能默认设置不同值，通常建议保持默认值（如16），除非有特殊需求。较大的block_size可能导致内存碎片，而过小的block_size则会增加内存管理的开销。

#### 2. **Continuous Batching的作用**
Continuous Batching是一种动态调度机制，能够最大化GPU的利用率。它通过将请求动态地分配到GPU资源上，并在每个批次处理完成后立即释放资源，从而实现GPU的持续满载运行。

- **显存管理**：为了支持高效的Continuous Batching，需要合理设置显存利用率。通常建议将显存使用比例控制在0.85~0.95之间，以避免频繁的preemption（请求被抢占重算）。

- **连续批次处理（Continuous Batching）**：通过将请求动态调度到GPU资源，可以实现资源的高效利用。这不仅提高了吞吐量，还减少了排队等待的时间。

#### 3. **参数调整对吞吐量的影响**
通过调整

## 块 73

1. **把 Paging 和 Batching 当成一对来记**——单独理解任何一个都不完整，两者的配合才是 vLLM 快的原因
2. **面试标准答法**：先讲 PagedAttention 解决显存碎片，再讲 Continuous Batching 解决调度浪费，最后强调"调度灵活性建立在显存管理灵活性之上"
3. **共享前缀场景优先开 Prefix Caching**——RAG 和多轮 Chat 里相同前缀只存一份，能省显存又省计算，但收益取决于前缀占比和命中率，需以实测为准
4. **不要手动凑 batch**——vLLM 的 Scheduler 比你更聪明，把请求丢进去让它自动调度
5. **监控 preemption 次数**——如果频繁出现请求被抢占重算，说明 `gpu_memory_utilization` 设太高了
6. **Block size 一般沿用默认**——部分版本默认 16 已是常见调优结果，默认值随版本/平台可能不同，除非你有明确的性能分析数据再调整

---

## 下一步预告

掌握了 PagedAttention 和 Continuous Batching 的核心机制后，你已经理解了 vLLM 高性能的底层原理。接下来，你将进入 **"能够使用 vLLM 加载模型并配置 gpu_memory_utilization 等关键参数进行推理部署"** 的实战环节——亲手启动 vLLM 服务，配置 `gpu_memory_utilization`、`max_num_seqs`、`tensor_parallel_size` 等关键参数，监控吞吐量和延迟指标，并将模型部署为生产可用的 OpenAI 兼容 API 服务。这是从"懂原理"进化为"能上线"的关键一步，继续加油！

（个性化提示：学习风格：hands-on）

## 块 74

```markdown
# 第一章：vLLM 推理部署的定位——从"能跑"到"能服务"

## 1.1 为什么需要专门的部署配置？

在本地用 `LLM.generate()` 跑几个测试用例很简单，但要把模型部署为**生产可用的推理服务**，你需要考虑：

> **技术要点：**  
> 在生产环境中，模型推理需要满足高吞吐量、低延迟和稳定的显存占用，这些都需要通过特定的配置来实现。

### 1.1.1 本地推理与生产部署的区别
- **本地推理：**  
  - 适合用于测试和小规模部署，资源占用低，运行速度快。
  - 适合使用简单的命令行参数配置，无需复杂的显存管理。

- **生产部署：**  
  - 需要优化模型资源的占用，确保模型在显存中的分配合理，避免内存溢出。
  - 需要配置显存使用策略，如`gpu_memory_utilization`，以控制模型占用的显存比例。
  - 需要

## 块 75

| 问题 | 本地测试无所谓 | 生产服务必须考虑 |
|------|-------------|---------------|
| 显存分配 | 用多少算多少 | 必须预留余量，防止 OOM |
| 并发处理 | 单请求串行 | 多请求同时到达，需要调度 |
| 上下文长度 | 固定短文本 | 用户输入长短不一，需要限制 |
| 多卡扩展 | 单卡够用 | 大模型必须多卡并行 |
| 服务接口 | 直接调 Python API | 需要 RESTful API 供前端调用 |

**为什么这样写**：vLLM 的部署不是"把模型加载进来就行"，而是一套需要精细调参的系统工程。`gpu_memory_utilization`、`max_model_len`、`tensor_parallel_size` 等参数直接决定服务的稳定性、吞吐量和成本。

### 1.2 部署方式概览

vLLM 提供两种主要部署方式（②）：

| 方式 | 适用场景 | 启动方式 |
|------|---------|---------|
| **离线推理（Offline）** | 批量处理数据集、实验评测 | `LLM(model=...)` Python API |
| **在线服务（Online）** | 生产环境 API 服务、Web 应用 | `vllm serve` 命令 |

本章重点讲解**在线服务部署**，这是将模型投入实际应用的核心路径。

---

## 第二章：显存规划——部署前的必修课

（个性化提示：学习风格：hands-on）

## 块 76

### 2.1 显存都去哪儿了？（①）

部署前必须清楚显存的三大消耗者：

```
总显存占用 = 模型权重 + KV Cache + 激活值/CUDA Graph/其他
```

**模型权重**：
- FP16/BF16：参数量 × 2 字节
- AWQ/GPTQ INT4：参数量 × 0.5 字节
- FP8：参数量 × 1 字节

**KV Cache**（PagedAttention 管理）：
- 每 token 的 KV Cache ≈ $2 \times \text{层数} \times \text{KV头数} \times \text{head\_dim} \times \text{精度}$
- 与**序列长度**和**并发请求数**成正比

**激活值与其他**：
- Prefill 阶段峰值激活
- CUDA Graph 预分配显存池
- 系统预留

（个性化提示：学习风格：hands-on）

## 块 77

#

## 块 78

### 2.2.1 KV Cache 怎么估算（用公式，别套固定数字）

不要直接抄"某模型 KV Cache = 固定 X GB"。这类数字往往没有正确体现 **GQA/MQA 的 KV 头数**，很容易偏大或偏小。更稳妥的做法是用公式估算再和实测对照：

```text
单请求 KV Cache 字节数
≈ 2 × 层数(num_hidden_layers)
     × KV头数(num_key_value_heads)
     × head_dim
     × token 数(序列长度)
     × 每元素字节数(FP16=2, FP8=1)
```

- 开头的 `2` 表示 Key 和 Value 各一份。
- **对于 GQA/MQA 模型，必须用 KV 头数 `num_key_value_heads`，而不是 Query 头数 `num_attention_heads`**——很多模型的 KV 头数远小于 Query 头数，用错会把估算放大数倍。
- 参数从模型 `config.json` 读取，乘以计划支持的最大序列长度与并发请求数得到理论值，最后**以 vLLM 启动日志实际分配的 KV Cache 显存为准**修正。

（个性化提示：学习风格：hands-on）

## 块 79

### 2.3 显存诊断命令（①）

```bash
# 查看 GPU 显存总量
nvidia-smi

# 查看 vLLM 启动时的显存分配详情
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9
# 启动日志会自动打印模型加载和 KV Cache 分配信息

# 运行时监控显存
nvidia-smi -l 2  # 每 2 秒刷新

# 查看 vLLM 指标（Prometheus 格式），KV Cache 使用率、吞吐、排队等都在这里
curl http://localhost:8000/metrics
```

> **说明**：运行时的显存/KV Cache/吞吐等指标以 `/metrics`（Prometheus 格式）为准；不同 vLLM 版本暴露的具体端点会有差异，请以你所装版本的官方文档（或 `vllm serve --help` 与启动日志）为准，不要臆造未文档化的调试端点。

---

## 第三章：关键参数详解——gpu_memory_utilization 与 friends

（个性化提示：学习风格：hands-on）

## 块 80

```markdown
# 第3章 关键参数详解——gpu_memory_utilization 与 friends

## 3.1 gpu_memory_utilization：显存使用的"总闸门"（③⑥）

### 3.1.1 理解显存管理的核心参数

在使用vLLM进行模型推理部署时，显存管理是至关重要的一步。`gpu_memory_utilization`参数是控制显存使用的重要指标，它决定了vLLM占用显存的比例。这一参数的取值范围为0到1，直接影响模型的运行效率和性能。

### 3.1.2 显存管理的核心作用

`gpu_memory_utilization`参数

## 块 81

### 3.2 max-model-len：上下文窗口的"硬天花板"（③⑤）

**作用**：限制模型处理的最大序列长度（包括 prompt + 生成的文本）。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192
```

**为什么这样写**：
- 如果不指定，vLLM 使用模型配置中的默认最大长度（如 Llama-3 原生 8K，Qwen2.5 原生 128K）
- **一般不要把它设得超过模型声明的最大上下文长度**：模型没在更长序列上训练过，直接调大不是"精度略降"这么简单，而可能显著劣化甚至输出异常。确需更长上下文时，应使用官方提供的长上下文方案（如对应的长上下文权重或 RoPE scaling 等外推配置），并**单独做质量验证**，而不是随手把 8K 模型设成 32K
- **KV Cache 显存与序列长度成正比**——8K → 32K，KV Cache 约翻 4 倍，并发能力随之大幅下降

**调参建议**：

| 业务场景 | 建议值 | 理由 |
|---------|--------|------|
| 短文本问答 | 2048~4096 | 省显存，支持更多并发 |
| 通用对话 | 8192 | 平衡上下文和并发 |
| 长文档分析 | 32768~131072 | 需要处理长文本，但并发大幅降低 |

（个性化提示：学习风格：hands-on）

## 块 82

### 3.3 tensor-parallel-size：多卡并行的"切分刀"（①③⑤）

**作用**：张量并行度，将模型权重切分到多张 GPU 上。

```bash
# 单卡推理
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1

# 4 卡推理（70B 模型必需）
vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --tensor-parallel-size 4
```

**为什么这样写**：
- 每张 GPU 只保存约 $1/N$ 的模型权重
- TP 每步计算都要跨卡做 AllReduce 同步，**强烈依赖高速卡间互联**——优先单机 NVLink/NVSwitch；用 PCIe 或跨节点并非不能运行，但通信开销更大，是否可接受应以基准测试为准
- `tensor_parallel_size` 必须整除注意力头数（如 32 头可以切 2/4/8，不能切 3）

**多卡部署对照表**（①⑤）：

| 模型 | 单卡 24GB | 2×24GB | 4×24GB | 8×80GB |
|------|----------|--------|--------|--------|
| 7B (FP16) | ⚠️ 紧张 | ✅ | ✅ | ✅ |
| 7B (AWQ) | ✅ 流畅 | ✅ | ✅ | ✅ |
| 14B (FP16) | ❌ | ✅ | ✅ | ✅ |
| 70B (FP16) | ❌ | ❌ | ⚠️ 勉强 | ✅ |
| 70B (AWQ) | ❌ | ✅ | ✅ | ✅ |

（个性化提示：学习风格：hands-on）

## 块 83

### 3.4 max-num-seqs 与 max-num-batched-tokens：并发控制的"双闸"（③）

| 参数 | 作用 | 默认值 | 调参建议 |
|------|------|--------|---------|
| `max_num_seqs` | 单步可并行调度的最大序列数（调度上限） | 随版本而定（曾常见 256，请以 `vllm serve --help` 为准） | 显存紧张时减小（如 8~64） |
| `max_num_batched_tokens` | 每步最多处理的 token 数 | 随版本/是否开启 chunked prefill 而定 | 限制 Prefill 峰值，小值 ITL 更好，大值 TTFT 更好 |

> **说明**：上表默认值仅为示例，vLLM 各版本会调整这些默认行为，**请以 `vllm serve --help` 或所装版本文档为准**，不要把某个具体数字当成永久默认。

**为什么这样写**：
- `max_num_seqs` 是"单步最多同时调度多少条序列"的**上限**，并不意味着一启动就按这个数把 KV Cache 全额预留；但它决定了高并发时 KV Cache 的峰值压力——并发越多、上下文越长，KV Cache 需求越大
- `max_num_batched_tokens` 控制 Chunked Prefill 的块大小。小值（如 2048）让 Decode 更频繁，降低 ITL；大值（如 16384）一次处理更多 Prefill，降低 TTFT

---

## 第四章：启动 vLLM 服务——从最小可用到生产配置

（个性化提示：学习风格：hands-on）

## 块 84

### 4.1 最小可用启动（②）

```bash
vllm serve Qwen/Qwen2.5-1.5B-In

## 块 85

# 启动 vLLM 服务：显存紧张时的降配配置思路

## 前言
在实际部署 vLLM 服务时，可能会遇到显存紧张的问题。此时，我们需要采取降配策略，通过调整模型参数和配置，确保模型能够顺利运行。本文将介绍一个典型的降配配置思路，并提供一些实用的验证任务，帮助你更好地理解和实践。

## 降配配置思路
对于单卡运行70B模型的情况，降配配置的思路可以总结为以下三条：
1. **换预量化权重（AWQ）**：使用预量化权重可以显著减少模型的内存占用。预量化权重采用4bit格式，能够有效降低模型大小。
2. **缩小 KV Cache（限制长度和并发）**：通过限制模型

## 块 86

> **把它变成"可直接运行配置"前，请先实测以下项**（任一未测就不要对外宣称"这套配置可以直接跑 70B"）：
> - **GPU 型号与实际可用显存**（不同 24GB/40GB/48GB 卡差别很大）；
> - **主机内存与 CPU**（`--cpu-offload-gb` 要占用等量的 CPU 内存，内存不足会直接失败或触发系统交换）；
> - **PCIe 带宽/拓扑**（CPU↔GPU 传输是 offload 的瓶颈，带宽低会显著拖慢速度）；
> - **模型仓库的具体 revision**（不同 AWQ 打包的权重体积、`group_size`、是否含额外文件会影响能否加载与占用）；
> - **实测吞吐（tokens/s）与首 token 延迟（TTFT）**，据此再调 `--max-model-len`、`--max-num-seqs`、`--cpu-offload-gb`。
>
> 上表数值只是**起始试探值**；在你自己的机器上完整跑通、量过上述指标之前，请把它当"配置思路"看待，而不是"拿来即用的生产配置"。

---

## 第五章：调用部署好的服务

（个性化提示：学习风格：hands-on）

## 块 87

### 5.1 OpenAI 兼容 API（②）

vLLM 启动后，自动提供**兼容 OpenAI 常用接口**（如 Chat Completions、Completions、Models 等）的 RESTful API：

```bash
# 测试服务是否启动
curl http://localhost:8000/v1/models

# 文本补全
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "my-qwen",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

# 聊天补全
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "my-qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    }'
```

（个性化提示：学习风格：hands-on）

## 块 88

### 5.2 Python 客户端调用（②）

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="my-qwen",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)
print(response.choices[0].message.content)
```

**为什么这样写**：`openai` Python 包可以直接连接 vLLM 服务，常见场景下只改 `base_url`（和把 `api_key` 设为占位值）即可复用原有代码，迁移成本很低。这是 vLLM 的重要优势之一。不过"兼容"针对的是常用的 Chat/Completions 等接口，OpenAI 平台的部分高级特性、专有端点或最新字段不一定 1:1 支持，接入前建议对你实际用到的接口做一次验证。

（个性化提示：学习风格：hands-on）

## 块 89

### 5.3 流式输出（①）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="my-qwen",
    messages=[{"role": "user", "content": "写一首关于AI的诗"}],
    stream=True,  # 开启流式输出
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**为什么这样写**：`stream=True` 使用 Server-Sent Events（SSE）逐 token 返回，Web 端用户体验极佳。

---

## 第六章：电脑配置批注

本章节涉及 vLLM 推理服务的本地部署与云端部署实操。

（个性化提示：学习风格：hands-on）

## 块 90

### 6.1 本地环境配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Linux（Ubuntu 20.04+） | Linux（Ubuntu 22.04+） |
| **GPU** | 1 张 NVIDIA GPU（显存按目标模型而定，7B 示例约 16GB 起，更小模型可用更小显存验证） | 1 张 RTX 4090（24GB）/ A100（40GB/80GB） |
| **CPU** | 8 核 | 16 核+ |
| **内存** | 32GB | 64GB+ |
| **磁盘** | 100GB SSD | 500GB+ NVMe SSD |
| **NVIDIA 驱动** | 驱动支持 CUDA 11.8+ | 驱动支持 CUDA 12.1+ |
| **Python** | 以所装 vLLM 支持范围为准（示例 3.10） | 示例 3.12（仍以发行说明为准） |
| **网络** | 本地回环 | 10Gbps+（多机部署） |
| **是否支持 CPU 运行** | ⚠️ 支持但极慢（仅调试） | — |

> **显存说明**：表中"16GB"是针对本章 **7B 示例**的经验下限，并非"部署任何模型的统一门槛"——显存要求应随模型规模而定，用小模型仅做环境验证时几 GB 也够。

（个性化提示：学习风格：hands-on）

## 块 91

### 6.2 云端环境配置参考

| 场景 | 推荐实例 | 启动命令示例 |
|------|---------|-------------|
| 低成本实验 | AutoDL RTX 3090（24GB） | `vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9` |
| 标准推理 | AWS g5.xlarge（A10G 24GB） | `vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192` |
| 高并发服务 | AWS g5.12xlarge（4×A10G） | `vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 4` |
| 超大模型 | AWS p4d.24xlarge（8×A100） | `vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --tensor-parallel-size 8` |

（个性化提示：学习风格：hands-on）

## 块 92

# 第六章：多卡部署的硬件要求

## 6.3 多卡部署的硬件要求

在进行多卡部署时，硬件配置是确保模型高效推理的重要基础。以下是多卡部署的不同并行策略及其对硬件的要求：

### 1. 单卡部署
- **并行策略**：单卡部署
- **硬件要求**：最低 GPU 数量为 1 卡
- **特点**：无需额外硬件配置，适用于资源有限的环境。

### 2. TP（张量并行）

## 块 93

# 6.3 多卡部署的硬件要求

## 6.3.1 前言

在使用 vLLM 进行模型推理部署时，硬件配置是确保部署稳定性和性能的关键因素。本节将详细讲解多卡部署的硬件要求，并提供实际操作指南，帮助您在实际场景中顺利完成部署。

## 6.3.2 显存预算配置

### 1. 显存预算配置原则

在 vLLM 部署中，显存预算的配置至关重要。显存预算决定了模型在 GPU 中的推理能力，包括权重存储、KV 缓存以及其他运行相关的资源占用。

#### 1.1 基本配置原则

- **默认配置**：vLLM 会根据模型大小自动设置显存预算。默认情况下，显存预算占总显存的 80%。
- **高级配置**：如果需要更高的性能，可以将显存预算比例增加至 90% 或以上。

#### 1.2 显存预算调整

- **调整方法**：使用 `vllm serve` 命令调整显存预算

## 测评题目

### Q-KB15-ST1500-002
vLLM 推理服务通常需要 NVIDIA GPU 及足够的________。

### Q-KB1504-006
模型量化（Quantization）的主要目的是？

### Q-KB15-ST1500-003
安装 vLLM 前，最需要优先检查的是？

### Q-KB1504-009
vLLM 和 TGI（Text Generation Inference）都是用于什么目的的框架？

### Q-KB1504-008
AWQ 和 GPTQ 属于哪种量化方式？

### Q-KB1504-010
vLLM 的核心技术优势是________，它极大提升了显存利用率和吞吐量。

### Q-KB1504-004
请写出启动 vLLM OpenAI 兼容 API 服务器的命令行指令（模型为 `Qwen/Qwen2-7B-Instruct`，端口 8000）。

### Q-KB1504-017
FastAPI 相比 Flask 在大模型 API 开发中的主要优势是？
