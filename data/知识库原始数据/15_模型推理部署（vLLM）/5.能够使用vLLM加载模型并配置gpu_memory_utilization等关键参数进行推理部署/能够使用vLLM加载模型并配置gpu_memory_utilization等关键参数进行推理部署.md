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

**适合人群**：已完成 vLLM 环境准备、采样参数配置和 PagedAttention/Continuous Batching 原理学习，希望亲手使用 vLLM 加载模型、配置关键参数并部署为生产可用推理服务的进阶学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够独立使用 `vllm serve` 命令启动推理服务，熟练配置 `gpu_memory_utilization`、`max_model_len`、`tensor_parallel_size` 等关键参数，并掌握显存优化与多卡部署技巧。

**预计学习时长**：约 1.5 小时  
**环境准备**：复用上一章的 vLLM 环境（Python 3.10~3.13、vLLM 已安装、NVIDIA GPU 显存 ≥16GB）

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

> **说明**：下表数值为**数量级估算**（权重按参数量×每参数字节数粗算，KV Cache 与序列长度、batch、KV 头数/GQA、精度等强相关），实际占用随模型结构与部署配置浮动，请以启动日志中 vLLM 打印的实测分配为准。

| 模型 | FP16 权重 | INT4 权重 | FP8 权重 | KV Cache（8K 上下文，单请求，约） |
|------|----------|----------|---------|---------------------------|
| Qwen2.5-7B | ~14 GB | ~3.5 GB | ~7 GB | ~2.4 GB |
| Llama-3.1-8B | ~16 GB | ~4 GB | ~8 GB | ~3 GB |
| Qwen2.5-14B | ~28 GB | ~7 GB | ~14 GB | ~4.8 GB |
| Llama-3.1-70B | ~140 GB | ~35 GB | ~70 GB | ~20 GB |
| DeepSeek-V3 (MoE) | ~400 GB | ~110 GB | ~200 GB | ~0.4 GB（MLA 压缩） |

**为什么这样写**：部署前必须先做"显存预算"。以 7B 模型 FP16 为例，粗算约需 14GB 权重 + 约 2.4GB KV Cache + 数 GB 其他 ≈ **20GB 上下**，在 24GB 显存上尚有余量，但一旦要支持多并发，KV Cache 会随并发近似线性增长，24GB 就可能不够——所以并发数需结合实测显存来定。

### 2.3 显存诊断命令（①）

```bash
# 查看 GPU 显存总量
nvidia-smi

# 查看 vLLM 启动时的显存分配详情
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9
# 启动日志会自动打印模型加载和 KV Cache 分配信息

# 运行时监控显存
nvidia-smi -l 2  # 每 2 秒刷新

# 查看 vLLM 指标（Prometheus 格式）
curl http://localhost:8000/metrics

# 查看模型内部统计
curl http://localhost:8000/v1/internal/stat
```

---

## 第三章：关键参数详解——gpu_memory_utilization 与 friends

### 3.1 gpu_memory_utilization：显存使用的"总闸门"（③⑥）

这是 vLLM 部署中**最关键的显存参数之一**，直接决定能留给 KV Cache 的显存、进而影响最大并发。

**作用**：限制 vLLM 可以使用的 GPU 显存比例，范围 0~1。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9
```

**为什么这样写**：
- 默认值为 `0.9`，即 vLLM 最多使用 90% 的 GPU 显存
- 如果启动前 GPU 已有 50% 显存被占用，且 `--gpu-memory-utilization=0.9`，则 vLLM 只能分配剩余的 40%
- 这个参数直接决定 KV Cache 能分配多少 Block，进而决定**最大并发数**

**调参建议**（⑥）：

| 场景 | 建议值 | 理由 |
|------|--------|------|
| 显存充裕、追求高吞吐 | 0.90~0.95 | 最大化 KV Cache，支持更多并发 |
| 显存紧张、防止 OOM | 0.70~0.85 | 预留余量应对峰值和碎片 |
| 极限压缩（单卡跑大模型） | 0.35~0.50 | 配合量化 + CPU offloading |

**为什么这样写**：设太高（如 0.98）容易在峰值时 OOM；设太低（如 0.5）则浪费显存、吞吐上不去。0.85~0.90 是大多数生产环境的甜点区。

### 3.2 max-model-len：上下文窗口的"硬天花板"（③⑤）

**作用**：限制模型处理的最大序列长度（包括 prompt + 生成的文本）。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192
```

**为什么这样写**：
- 如果不指定，vLLM 使用模型配置中的默认值（如 Llama-3 原生 8K，Qwen2.5 原生 128K）
- 可以**超过**模型原始训练长度（如把 8K 模型设成 32K），但长上下文精度可能下降
- **KV Cache 显存与序列长度成正比**——8K → 32K，KV Cache 翻 4 倍

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
- 每张 GPU 只保存 $1/N$ 的模型权重
- 需要**NVLink 或高速互联**，因为每步计算都要 AllReduce 同步
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
| `max_num_seqs` | 最大并发序列数 | 256 | 显存紧张时减小（如 8~64） |
| `max_num_batched_tokens` | 每步最多处理的 token 数 | 不限制 | 限制 Prefill 峰值，小值 ITL 更好，大值 TTFT 更好 |

**为什么这样写**：
- `max_num_seqs` 直接决定 KV Cache 预留量。256 个并发 × 8K 上下文 = 海量 KV Cache
- `max_num_batched_tokens` 控制 Chunked Prefill 的块大小。小值（2048）让 Decode 更频繁，降低 ITL；大值（16384）一次处理更多 Prefill，降低 TTFT

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
  --enable-prefix-caching \
  --trust-remote-code
```

**参数解读**：

| 参数 | 说明 |
|------|------|
| `--port 8000` | API 服务监听端口 |
| `--host 0.0.0.0` | 绑定所有网卡，允许外部访问 |
| `--served-model-name my-qwen` | 对外暴露的模型名，客户端请求时用这个名字 |
| `--dtype auto` | 自动检测模型精度（FP16/BF16） |
| `--trust-remote-code` | 允许执行模型仓库中的自定义代码（Qwen/DeepSeek 等必须加） |
| `--enable-prefix-caching` | 开启自动前缀缓存，在共享 System Prompt 等长公共前缀场景下可明显降低 TTFT（幅度取决于前缀占比与命中率，需实测，非固定值） |

### 4.3 显存紧张时的降配配置（⑥）

```bash
vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.45 \
  --enable-chunked-prefill \
  --enforce-eager \
  --quantization awq \
  --swap-space 20
```

**为什么这样写**：这是参考文章⑥给出的一组实战配置，据其记载在单卡 A100 40GB 上把 70B 模型的显存占用大幅压下来（该文提到从约 38GB 降到约 19GB，为其特定环境下的实测值，具体数字随模型、量化方案、序列长度/并发与 vLLM 版本而变，请以自己环境实测为准）。核心三板斧：**量化（AWQ）+ 缩小 KV Cache（限制长度和并发）+ CPU offloading（swap-space）**。

---

## 第五章：调用部署好的服务

### 5.1 OpenAI 兼容 API（②）

vLLM 启动后，自动提供与 OpenAI 完全兼容的 RESTful API：

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

**为什么这样写**：`openai` Python 包可以直接连接 vLLM 服务，无需任何修改。这是 vLLM 最大的优势之一——**零成本迁移**。

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
| **GPU** | 1 张 NVIDIA GPU，显存 **≥16GB** | 1 张 RTX 4090（24GB）/ A100（40GB/80GB） |
| **CPU** | 8 核 | 16 核+ |
| **内存** | 32GB | 64GB+ |
| **磁盘** | 100GB SSD | 500GB+ NVMe SSD |
| **NVIDIA 驱动** | 驱动支持 CUDA 11.8+ | 驱动支持 CUDA 12.1+ |
| **Python** | 3.10 | 3.12 |
| **网络** | 本地回环 | 10Gbps+（多机部署） |
| **是否支持 CPU 运行** | ⚠️ 支持但极慢（仅调试） | — |

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
| TP（张量并行） | 2 卡（需整除 head 数） | **极高**（必须 NVLink） | 仅限单机 |
| PP（流水线并行） | 2 卡 | 中 | 跨机需 10Gbps+ |
| TP + PP 组合 | 4 卡+ | 组内极高、组间中 | 多机 InfiniBand |

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
| **gpu_memory_utilization** | 显存总闸门，默认 0.9，紧张时降到 0.7~0.85，极限时 0.35~0.5 |
| **max_model_len** | 上下文硬天花板，直接影响 KV Cache 大小，短文本设 2048~4096 |
| **tensor_parallel_size** | 张量并行度，必须整除注意力头数，多卡部署大模型必需 |
| **max_num_seqs** | 最大并发数，默认 256，显存紧张时减小 |
| **max_num_batched_tokens** | 每步处理 token 上限，影响 Prefill/Decode 平衡 |
| **生产配置模板** | port + host + served-model-name + dtype + max-model-len + gpu-memory-utilization + TP + prefix-caching |
| **降配三板斧** | 量化（AWQ）+ 缩小 KV Cache（限制长度/并发）+ CPU offloading（swap-space） |
| **OpenAI 兼容** | vLLM 自动提供 `/v1/chat/completions` 等端点，零成本迁移 |

---

## 养成的好习惯

1. **部署前先算显存预算**——权重 + KV Cache × 并发 + 其他，确保心里有数
2. **从保守配置开始**——`gpu-memory-utilization=0.85`、`max-model-len=4096`，确认稳定后再上调
3. **三个参数一起调**——`gpu_memory_utilization`、`max_model_len`、`max_num_seqs` 互相影响，需要联动调整
4. **监控 preemption 次数**——频繁 preemption 说明显存不够，需要降并发或扩卡
5. **量化是显存救星**——H100 用 FP8，消费级用 AWQ，70B 模型也能在 24GB 上跑
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