# 能够使用vLLM加载模型并配置gpu_memory_utilization等关键参数进行推理部署

> **参考文章链接**
> 
> ① 底稿：《vLLM模型部署》
> 　　网址：https://blog.csdn.net/weixin_57580819/article/details/163075945
> 
> ② 权威文章1：《Quickstart - vLLM》
> 　　网址：https://docs.vllm.ai/en/stable/getting_started/quickstart/
> 
> ③ 权威文章2：《Engine Arguments — vLLM》
> 　　网址：https://docs.vllm.ai/en/v0.6.4/models/engine_args.html
> 
> ④ 权威文章3：《Optimization and Tuning - vLLM》
> 　　网址：https://docs.vllm.ai/en/stable/configuration/optimization/
> 
> ⑤ 参考文章1：《vLLM实战：单机多卡大模型推理部署指南》
> 　　网址：https://blog.csdn.net/qq_40902709/article/details/147086232
> 
> ⑥ 参考文章2：《用 vLLM 跑大模型显存吃紧？一行命令教你把 GPU 内存打下来》
> 　　网址：https://blog.csdn.net/weixin_41544125/article/details/149346036

---

> **本板块目标版本（示例，请以你实际安装的发行为准）**
> - vLLM：v0.6.4（本文正文示例均以该版本的可运行接口为准；旧版写法见文末"历史版本差异"附录，如无则以正文为准）
> - Python：3.10~3.12（以该 vLLM 版本的官方发行说明为准）
> - PyTorch / CUDA wheel：以 `pip install vllm==0.6.4` 实际解析到的依赖及该发行说明为准（如常见 cu121）
> - GPU：示例按常见 NVIDIA GPU（具体型号 / 显存以你的机器为准）
> - 文档审校日期：2026-08（内容为静态审校，未在多 GPU 环境逐条实机验证；部署命令与降配示例的实际吞吐 / 显存请以你的机器实测为准）

---

**适合人群**：已完成 vLLM 环境准备、采样参数配置和 PagedAttention/Continuous Batching 原理学习，希望亲手使用 vLLM 加载模型、配置关键参数并部署为生产可用推理服务的进阶学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够独立使用 `vllm serve` 命令启动推理服务，熟练配置 `gpu_memory_utilization`、`max_model_len`、`tensor_parallel_size` 等关键参数，并掌握显存优化与多卡部署技巧。

**预计学习时长**：约 1.5 小时  
**环境准备**：复用上一章的 vLLM 环境（Python 版本以所装 vLLM 发行说明为准、vLLM 已安装、NVIDIA GPU；具体显存需求随所选模型而定，本章 7B 示例约需 16GB 起）

---

## 第一章：vLLM 推理部署的定位——从"能跑"到"能服务"

### 1.1 为什么需要专门的部署配置？

在本地用 `LLM.generate()` 跑几个测试用例很简单，但要把模型部署为**生产可用的推理服务**，你需要考虑：

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

### 2.2 模型规模与显存对照表（①）

> **说明**：下表只给 **权重的数量级估算**（权重 ≈ 参数量 × 每参数字节数，相对确定）。**KV Cache 单独列在 2.2.1 用公式说明**——它高度依赖 KV 头数（GQA/MQA）、层数、head_dim、缓存精度与序列长度/并发，用一个固定数字很容易误导。所有数值都是量级参考，实际占用请以启动日志中 vLLM 打印的实测分配为准。

| 模型 | 参数量（约） | FP16 权重 | INT4 权重 | FP8 权重 |
|------|----------|----------|----------|---------|
| Qwen2.5-7B | ~7B | ~14 GB | ~3.5 GB | ~7 GB |
| Llama-3.1-8B | ~8B | ~16 GB | ~4 GB | ~8 GB |
| Qwen2.5-14B | ~14B | ~28 GB | ~7 GB | ~14 GB |
| Llama-3.1-70B | ~70B | ~140 GB | ~35 GB | ~70 GB |
| DeepSeek-V3（MoE，总参约 671B） | ~671B | 约 1.3 TB 量级 | 约 0.3 TB 量级 | 约 0.67 TB 量级 |

> **关于 DeepSeek-V3**：它是 MoE 模型，**总参数量约 671B**，每 token 只激活其中一部分专家（激活参数约几十 B）。但部署时**权重需要整体驻留显存/多机显存**，因此显存要按"总参数量"估算：FP16 约 `671B × 2 ≈ 1.3 TB` 量级，需要**多机多卡**才能承载，绝不是几百 GB 单机能放下的。网上一些"约 400GB"的说法多半把"激活参数"当成了"驻留权重"，这是常见误区。它的 KV Cache 因采用 MLA（多头潜在注意力）等结构而相对更省，但这仍是"另算的一项"，不能与权重混为一谈。

**为什么这样写**：部署前必须先做"显存预算"。以 7B 模型 FP16 为例，粗算约需 14GB 权重 + 数 GB KV Cache（按 2.2.1 公式估算）+ 数 GB 其他 ≈ **20GB 上下**，在 24GB 显存上尚有余量；但一旦要支持多并发，KV Cache 会随并发近似线性增长，24GB 就可能不够——所以并发数需结合实测显存来定。

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

### 3.1 gpu_memory_utilization：显存使用的"总闸门"（③⑥）

这是 vLLM 部署中**最关键的显存参数之一**，直接决定能留给 KV Cache 的显存、进而影响最大并发。

**作用**：限制 vLLM 可以使用的 GPU 显存比例，范围 0~1。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9
```

**为什么这样写**：
- 默认值为 `0.9`，即 vLLM 会把**该卡总显存 × 0.9** 作为自己的显存预算目标（用来放模型权重 + KV Cache + 激活/CUDA Graph 等），而不是"在别人占用之外再额外拿 90%"
- 因此，如果启动前这张卡已被其他进程占用了较多显存，vLLM 按 `总显存 × 0.9` 计算出的目标预算可能无法满足（权重都放不下），会直接**启动失败/报显存不足**，而不是"自动只用剩下的一点点"——共享 GPU 时要把这一比例往下调，或先腾出显存
- 在权重占用固定的前提下，**调低这个比例主要是压缩留给 KV Cache 的空间**（进而降低可支持的并发/上下文），而不是"把模型权重变小"

**调参建议**（⑥）：

| 场景 | 建议值 | 理由 |
|------|--------|------|
| 显存充裕、追求高吞吐 | 0.90~0.95 | 给 KV Cache 更多空间，支持更多并发 |
| 显存紧张、防止 OOM / 共享 GPU | 0.70~0.85 | 预留余量应对峰值和碎片，或给同卡其他进程让出显存 |
| 单卡硬塞大模型 | 需配合量化 + CPU offload | 注意：调低本参数只压缩 KV Cache、并不缩小权重；权重放不下要靠 `--cpu-offload-gb`/多卡，而不是把本参数调到很低 |

> **纠一个常见误区**：不要把"单卡跑大模型"简单等价于"把 `gpu_memory_utilization` 调到 0.35~0.5"。这个比例决定的是 vLLM 的显存预算目标，调低它主要是**少给 KV Cache 留空间**；如果权重本身都放不下，调低它只会更早 OOM，真正该做的是量化 + `--cpu-offload-gb` + 必要时多卡。

**为什么这样写**：设太高（如 0.98）容易在峰值时 OOM；设太低则给 KV Cache 的空间过小、吞吐上不去。0.85~0.90 是大多数生产环境（权重能放下的前提下）的甜点区。

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

### 4.1 最小可用启动（②）

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

**为什么这样写**：vLLM 的默认值已经能跑通。默认监听 `0.0.0.0:8000`，自动从 HuggingFace 下载模型。

### 4.2 生产级完整配置（①⑤）

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name my-qwen \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching
  # 仅当模型确实需要自定义建模代码时，再另外追加 --trust-remote-code（见下方说明）
```

**参数解读**：

| 参数 | 说明 |
|------|------|
| `--port 8000` | API 服务监听端口 |
| `--host 0.0.0.0` | 绑定所有网卡，允许外部访问 |
| `--served-model-name my-qwen` | 对外暴露的模型名，客户端请求时用这个名字 |
| `--dtype auto` | 自动检测模型精度（FP16/BF16） |
| `--trust-remote-code` | **可选、且有风险**：仅当模型仓库依赖自定义建模代码（部分模型/自定义结构）时才需要，加上它等于允许执行仓库中的 Python 代码。是否需要因模型而异（很多主流模型已并入 transformers、无需此项），**不要无脑加**；启用前务必确认模型来源可信、代码已审查，避免供应链风险 |
| `--enable-prefix-caching` | 开启自动前缀缓存，在共享 System Prompt 等长公共前缀场景下可明显降低 TTFT（幅度取决于前缀占比与命中率，需实测，非固定值） |

### 4.3 显存紧张时的降配配置思路示例（⑥）

单卡跑不下的大模型，降配的思路是三条同时用：**换预量化权重（缩小权重）+ 缩小 KV Cache（限制长度和并发）+ 把放不下的部分 offload 到 CPU 内存**。下面给一个"单卡 + AWQ 预量化 70B"的**配置思路示例**——它演示的是"该动哪些开关、开关之间怎么配合"的方向，**不是一份未经你本机实测、就能照抄直接跑通的生产配置**：

```bash
# ⚠️ 下面是"思路示例"，各数值需按你的实际硬件实测调整，不保证在任意机器上开箱即用
# 注意：--quantization awq 是"按 AWQ 格式加载权重"，因此 model 必须指向一个
#       已经做过 AWQ 量化的仓库（如 *-AWQ），而不是原始 FP16 权重仓库
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enforce-eager \
  --cpu-offload-gb 40
```

**为什么这样写**（几处容易踩的坑，务必注意）：
- **量化模型要用预量化仓库**：`--quantization awq` 不会把一个普通的 FP16 仓库"临时量化"成 4bit，它期望模型本身就是 AWQ 格式的权重。用原始 `meta-llama/Llama-2-70b-hf` 再加 `--quantization awq` 是自相矛盾的，会加载失败。
- **权重本身要先放得下**：70B 的 AWQ(4bit) 权重约 35GB 量级，再加激活/KV Cache/CUDA Graph，单卡 40GB 通常**放不下**。想在单卡硬跑，必须把一部分权重/缓存 **offload 到 CPU 内存**——这才用 `--cpu-offload-gb`（把约定大小的权重放到 CPU），而不是靠调低 `--gpu-memory-utilization` 把权重"变小"（它压的是 KV Cache，不是权重）。
- **`--swap-space` 不是权重 offload**：`--swap-space` 是给 KV Cache 的 CPU 交换空间（抢占/换出时用），不能理解为"把模型权重放到内存"。把它当成权重 offload 手段是常见误解。
- 上面的具体数值（`--cpu-offload-gb 40`、长度/并发等）都要按你的实际显存/内存反复实测调整；CPU offload 会带来明显的速度下降，属于"跑得起来优先于跑得快"的取舍。参考文章⑥给出过一组"在其特定环境下把占用大幅压下来"的数字，但那是它的环境实测值，**不能照搬当成通用结论**，请以自己环境为准。核心仍是三板斧：**预量化（AWQ）+ 缩小 KV Cache（限制长度和并发）+ CPU offload**。

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

### 6.2 云端环境配置参考

| 场景 | 推荐实例 | 启动命令示例 |
|------|---------|-------------|
| 低成本实验 | AutoDL RTX 3090（24GB） | `vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9` |
| 标准推理 | AWS g5.xlarge（A10G 24GB） | `vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192` |
| 高并发服务 | AWS g5.12xlarge（4×A10G） | `vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 4` |
| 超大模型 | AWS p4d.24xlarge（8×A100） | `vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --tensor-parallel-size 8` |

### 6.3 多卡部署的硬件要求

| 并行策略 | 最低 GPU 数量 | 卡间带宽要求 | 网络要求 |
|---------|-------------|-------------|---------|
| 单卡 | 1 卡 | 无 | 无 |
| TP（张量并行） | 2 卡（需整除 head 数） | **很高**（强烈建议 NVLink/NVSwitch） | 优先单机；PCIe/跨节点并非不可用，但通信开销大，需实测评估 |
| PP（流水线并行） | 2 卡 | 中 | 跨机需 10Gbps+ |
| TP + PP 组合 | 4 卡+ | 组内很高、组间中 | 多机 InfiniBand |

> **说明**：TP 每步都要跨卡同步，对卡间带宽很敏感，因此**优先单机 NVLink/NVSwitch**；在 PCIe 或跨节点上运行 TP 是可行的，只是通信占比会明显上升，是否可接受应以基准测试为准，而非"没有 NVLink 就一定不能用 TP"。

---

## 第七章：常见问题与避坑清单

| 坑点 | 现象 | 解决方案 |
|------|------|---------|
| **OOM 启动失败** | `CUDA out of memory` | 降低 `--gpu-memory-utilization`（0.85→0.7）；减小 `--max-model-len`；使用量化模型（AWQ/GPTQ） |
| **抢占（Preemption）频繁** | 日志出现 `Sequence group X is preempted` | 增大 `--gpu-memory-utilization`；减小 `--max-num-seqs`；增大 `--tensor-parallel-size` |
| **多卡启动报错** | `Total number of attention heads must be divisible by tensor parallel size` | 确保 `tensor_parallel_size` 整除 `num_attention_heads`（如 32 头可切 2/4/8） |
| **模型加载失败** | `KeyError: qweight` 或量化相关报错 | AWQ 模型必须加 `--quantization awq`；检查模型路径是否正确 |
| **API 返回 404** | `Model not found` | 客户端请求的 `model` 名必须与 `--served-model-name` 一致 |
| **WSL2 性能差** | 推理速度比原生 Linux 慢一截（视 I/O、驱动与配置而定） | WSL2 可用但性能有损耗，生产环境建议原生 Linux |
| **CUDA Graph 导致 OOM** | 启动时显存暴涨 | 加 `--enforce-eager` 禁用 CUDA Graph，牺牲少量性能换稳定性 |
| **长 prompt 阻塞** | 大 TTFT（首 token 延迟） | 开启 `--enable-chunked-prefill`；减小 `--max-num-batched-tokens` |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **部署方式** | 离线推理（`LLM` API）vs 在线服务（`vllm serve`） |
| **gpu_memory_utilization** | 按"总显存×比例"定 vLLM 的显存预算目标，默认 0.9；调低主要压缩 KV Cache（不是缩小权重），共享 GPU/防 OOM 时下调 |
| **max_model_len** | 上下文硬天花板，直接影响 KV Cache 大小；短文本设 2048~4096，一般不超过模型声明的最大长度 |
| **tensor_parallel_size** | 张量并行度，必须整除注意力头数，多卡部署大模型必需 |
| **max_num_seqs** | 单步可调度的最大序列数（调度上限），默认随版本而定（曾常见 256），显存紧张时减小 |
| **max_num_batched_tokens** | 每步处理 token 上限，影响 Prefill/Decode 平衡 |
| **生产配置模板** | port + host + served-model-name + dtype + max-model-len + gpu-memory-utilization + TP + prefix-caching |
| **降配三板斧** | 预量化（AWQ 等）+ 缩小 KV Cache（限制长度/并发）+ CPU offload（权重用 `--cpu-offload-gb`，KV Cache 交换用 `--swap-space`） |
| **OpenAI 兼容** | vLLM 自动提供 `/v1/chat/completions` 等常用端点，多数场景只改 `base_url` 即可迁移（高级/专有特性需自行验证） |

---

## 养成的好习惯

1. **部署前先算显存预算**——权重 + KV Cache × 并发 + 其他，确保心里有数
2. **从保守配置开始**——`gpu-memory-utilization=0.85`、`max-model-len=4096`，确认稳定后再上调
3. **三个参数一起调**——`gpu_memory_utilization`、`max_model_len`、`max_num_seqs` 互相影响，需要联动调整
4. **监控 preemption 次数**——频繁 preemption 说明显存不够，需要降并发或扩卡
5. **量化是显存救星，但别夸大**——H100 可用 FP8，消费级常用 AWQ/GPTQ。量化能大幅降低权重占用，但"能不能塞进某张卡"要按实际权重算：例如 70B 即便 4bit 也约 35GB 量级，再加激活/KV Cache，通常**放不进 24GB 单卡**——想在小显存上跑超大模型，往往还得配合多卡或 CPU offload，不能想当然
6. **生产环境优先开 prefix caching**——在共享 System Prompt 等长公共前缀的场景下，可明显降低 TTFT（具体幅度视前缀占比与命中率，需实测）
7. **API 密钥必设**——`--api-key` 防止未授权访问，即使在内网也不要裸奔

---

## 下一步预告

恭喜你！至此，你已经完成了从"Linux 环境准备"到"vLLM 推理部署"的完整知识清单学习。你掌握了：

- ✅ Linux 环境与命令行操作
- ✅ Python 编程基础与数据处理
- ✅ PyTorch 深度学习框架
- ✅ NLP 基础、注意力机制、Tokenizer、Transformer 架构
- ✅ LLM 核心特性与三阶段训练流程（Pretrain → SFT → RLHF）
- ✅ HuggingFace 生态工具链
- ✅ 参数高效微调（LoRA/QLoRA）
- ✅ PyTorch 进阶训练技巧
- ✅ 分布式训练（DeepSpeed + ZeRO）
- ✅ 模型推理部署（vLLM）

接下来，你可以根据实际项目需求，选择深入某个方向（如多模态大模型、Agent 系统、模型量化压缩），或开始动手实践一个完整的"预训练 → 微调 → 部署"项目。你已经具备了从 0 基础到微调+部署推理大模型的完整知识体系和工程能力。继续加油，大模型之路越走越宽！