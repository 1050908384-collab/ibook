# 学习包

## 块 1

# 理解vLLM的PagedAttention与Continuous Batching机制

## 第一章：PagedAttention 的定位——为什么需要分页管理 KV Cache？

### 1.1 传统 KV Cache 管理的两大浪费

在大模型自回归推理中，每生成一个新 token，都需要缓存该 token 对应的 Key 和 Value 向量（即 KV Cache）。随着序列增长，KV Cache 不断膨胀，成为推理显存的主要消耗者。

在 vLLM 出现之前，传统推理框架（如 FasterTransformer、ONNX Runtime）采用**静态预分配**策略：

> **为每个请求预先分配一块按最大序列长度计算的连续显存空间。**

例如，模型支持的最大上下文为 2048 tokens，那么即使某个请求实际只生成 50 个 token，也要为其预留能装 2048 tokens 的连续 KV Cache 空间。

这导致了两种严重的显存浪费（①）：

1. **内存碎片**：请求的实际序列长度远小于最大序列长度，但需要预留较大空间，导致内存未被充分利用。
2. **显存浪费**：为每个请求预留的 KV Cache 大小远大于实际需求，导致显存利用率低下。

这种静态预分配策略在模型规模和推理序列长度不断扩展的今天，已经无法满足高性能推理的需求。

### 1.2 PagedAttention 的提出与

**本块学习检查**：先用一个最小示例回顾 理解vLLM的PagedAttention与Continuous Batching机制；确认能解释关键写法后再继续本块。

## 块 2

| 浪费类型 | 说明 | 类比 |
|---------|------|------|
| **内部碎片（Internal Fragmentation）** | 预留了 2048，实际只用了 50，剩余 1998 的空间被白白占用 | 租了 100 平米的办公室，只坐了 5 个人 |
| **外部碎片（External Fragmentation）** | 每个请求需要连续的大块显存，导致显存被切成大小不一的碎片，总空闲够但找不到足够大的连续空间 | 停车场有很多空位，但都是分散的小车位，大巴停不进去 |

**为什么这样写**：在这种静态预分配方案下，KV Cache 的显存利用率往往只能达到**几成左右**（不同来源常给出约 20%~40% 的经验区间，具体数值与序列长度分布、最大长度设置、请求模式强相关，仅为示意），相当一部分显存被碎片浪费。而显存是推理的命根子——可用于 KV Cache 的显存越多，能支持的并发请求数和上下文长度就越长。浪费显存 = 直接限制吞吐量和用户体验。

（个性化提示：学习风格：hands-on）

## 块 3

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

## 块 4

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

## 块 5

```markdown
# 理解vLLM的PagedAttention与Continuous Batching机制

## 2.2 Block Table——逻辑到物理的映射

在vLLM的模型推理过程中，为了高效管理大量的token数据，需要采用分页分配

## 块 6

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

## 块 7

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

## 块 8

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

## 块 9

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

## 块 10

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

## 块 11

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

## 块 12

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

## 块 13

### 4.2 PagedAttention与Continuous Batching的配合工作流程

为了更好地理解PagedAttention和Continuous Batching的工作原理及其配合机制，我们可以通过以下工作流程图进行分析：

#### 工作流程图解析

1. **PagedAttention（显存管理层）**
   - **作用**：负责管理模型的显存资源分配。
   - **工作流程**：
     - **

## 块 14

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

## 块 15

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

## 块 16

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

## 块 17

### 6.2 云端环境配置参考

| 场景 | 推荐实例 | PagedAttention 效果 |
|------|---------|-------------------|
| 低成本实验 | AutoDL RTX 3090（24GB）×1 | 7B 模型，并发 8~16 |
| 标准推理 | AWS g5.xlarge（A10G 24GB） | 7B~13B 模型，并发 16~32 |
| 高并发服务 | AWS g5.12xlarge（4×A10G） | 70B 模型（TP=4），并发 64+ |
| 超长上下文 | AWS p4d.24xlarge（8×A100） | 128K 上下文，Prefix Caching 效果显著 |

（个性化提示：学习风格：hands-on）

## 块 18

# vLLM部署：PagedAttention与Continuous Batching机制

## 6.3 关键参数与吞吐关系

### 6.3.1 关键参数的作用与建议值

在vLLM部署过程中，以下关键参数对模型性能和吞吐量有着重要影响：

1. **`block_size`**  
   - **作用**：每个Block存储的token数量，决定了模型处理的粒细程度。  
   - **建议值**：通常为16，但需根据模型版本和硬件规格进行微调。不同引擎版本可能有不同默认值，建议参考官方文档或通过实验确定最优值。

2. **`gpu_memory_utilization`**  
   - **作用**：控制GPU显存的使用比例，直接影响模型的运行效率。  
   - **建议值**：建议设置为0.85~0.95，避免显存溢出导致的preemption（任务被抢占执行）。

3. **`max_num_seqs`**  
   - **作用**：最大并发处理的序列数，直接影响模型的吞吐量。  
   - **建议值**：根据显存资源调整，建议范围为256~512。若显存资源充足，可适当提高值以提升吞吐量。

4. **`enable_prefix_caching`**  
   - **作用**：启用前缀缓存机制，可减少重复前缀请求

## 块 19

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

## 块 20

# 第一章：vLLM 推理部署的定位——从"能跑"到"能服务"

## 1.1 为什么需要专门的部署配置？

在本地用 `LLM.generate()` 跑几个测试用例很简单，但要把模型部署为**生产可用的推理服务**，你需要考虑：

### 1.1.1 本地推理与生产部署的区别
- **本地推理**：仅用于测试和验证模型性能，通常不考虑实际生产环境的需求。
- **生产部署**：需要满足以下要求：
  1. **性能优化**：提升推理速度和吞吐量。
  2. **稳定性**：确保模型在高负载下正常运行。
  3. **安全性**：保护模型和推理服务免受攻击。
  4. **可扩展性**：支持单机或多卡、多GPU的部署。

### 1.1.2 需要配置的关键参数
在部署过程中，需要配置以下关键参数以优化推理性能：
- `gpu_memory_utilization`：限制GPU内存使用，避免溢出。
- `max_model_len`：设置模型输入的最大长度。
- `tensor_parallel_size`：并行计算的大小，影响模型性能和显存占用。

### 1.1.3 实战示例
假设你已经完成了以下准备工作：
1. vLL

## 块 21

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

## 块 22

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

## 块 23

### 2.2 模型规模与显存对照表（①）

> **说明**：下表只给 **权重的数量级估算**（权重 ≈ 参数量 × 每参数字节数，相对确定）。**KV Cache 单独列在 2.2.1 用公式说明**——它高度依赖 KV 头数（GQA/MQA）、层数、head_dim、缓存精度与序列长度/并发，用一个固定数字很容易误导。所有数值都是量级参考，实际占用请以启动日志中 vLLM 打印的实测分配为准。

| 模型 | 参数量（约） | FP16 权重 | INT

## 块 24

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

## 块 25

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

## 块 26

# 第三章：显存管理参数详解——gpu_memory_utilization

## 3.1 gpu_memory_utilization：显存使用的"总闸门"（③⑥）

在使用vLLM进行模型推理部署时，`gpu_memory_utilization` 是一个关键参数，它决定了模型在GPU显存中的使用比例。简单来说，这个参数控制着vLLM占用显存的多少，范围在0到1之间。

### 作用
`gpu_memory_utilization` 用于限制vLLM占用显存的比例。默认值为0.9，意味着vLLM会将

## 块 27

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

## 块 28

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

## 块 29

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

## 块 30

# 最小可用启动（②）

在使用vLLM部署模型时，最小可用启动配置是最基本的配置方式。以下是详细的实现步骤和参数说明：

## 实战

## 块 31

# 4.3 显存紧张时的降配配置思路示例

在单卡运行大模型时，显存紧张是常见的问题。降配配置的思路是通过三条策略同时优化：**换预量化权重（缩小权重）+ 缩小 KV Cache（限制长度和并发）+ 把放不下的部分 offload 到 CPU 内存**。下面是一个具体的配置示例，帮助你理解如何在实际场景中应用这些策略。

---

## 示例配置思路

1. **预量化（AWQ）**：使用预量化权重可以有效减少模型大小。预量化权重的大小取决于量化格式（如4bit），具体参数由模型提供方确定。
2. **限制 KV Cache长度和并发**：通过设置`--max-model-len`和`--max-num-batched-tokens`，可以控制模型的运行长度和批量处理的大小。
3. **CPU内存卸载**：将无法在显存中加载的部分权重和缓存卸载到CPU内存中，使用`--cpu-offload-gb`参数控制卸载的大小。

---

## 示例配置代码

```bash
# ⚠️ 下面是"思路示例"，各数值需按你的实际硬件实测调整
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1

## 块 32

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

## 块 33

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

## 块 34

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

## 块 35

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

## 块 36

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

## 块 37

### 6.2 云端环境配置参考

| 场景 | 推荐实例 | 启动命令示例 |
|------|---------|-------------|
| 低成本实验 | AutoDL RTX 3090（24GB） | `vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9` |
| 标准推理 | AWS g5.xlarge（A10G 24GB） | `vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192` |
| 高并发服务 | AWS g5.12xlarge（4×A10G） | `vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 4` |
| 超大模型 | AWS p4d.24xlarge（8×A100） | `vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --tensor-parallel-size 8` |

（个性化提示：学习风格：hands-on）

## 块 38

# 第六章：电脑配置批注  
## 6.3 多卡部署的硬件要求  

在进行多卡部署时，硬件配置是确保顺利运行的关键因素。以下是多卡部署的硬件要求及注意事项：  

### 并行策略与硬件要求  

| 并行策略 | 最低 GPU 数量 | 卡间带宽要求 | 网络要求 |  
|---------|-------------|-------------|---------|  
| 单卡     | 1 卡         |

## 块 39

# 多卡部署的硬件要求

## 1. 学习目标
掌握多卡部署的硬件要求，理解多卡环境下vLLM模型推理的配置参数设置，能够顺利完成多卡部署。

## 2. 学习方式
通过实践操作，结合理论知识，完成多卡环境下的vLLM部署配置。

## 3. 学习步骤

### 1. 硬件环境检查
在开始配置之前，请确保以下硬件条件满足：
- 至少拥有两张支持相同架构的显卡（如两块NVIDIA GPU或两块AMD GPU）。
- 每块显卡具有足够的显存，能够支持vLLM的推理任务。

### 2. 基础参数设置
在多卡环境下，vLLM的运行依赖于以下参数的合理配置：
- **gpu_memory_utilization**：建议设置为0.85-0.9，确保显存资源的合理分配。
- **max_model_len**：设置为4096-8192，以适应多卡环境下的模型推理需求。
- **max_num_seqs**：根据显存限制，建议设置为256-512。
- **max_num_batched_tokens**：设置为512-1024，以平衡前缀编码和解码效率。

### 3. 生产配置模板
在多卡环境下，推荐使用以下生产配置模板：
- `--port 11111`
- `--host localhost`
- `--served-model-name your_model`
- `--dtype float16`

## 测评题目

### Q-KB15-ST1503-003
PagedAttention 的主要收益是？

### Q-KB15-ST1503-001
vLLM 的 PagedAttention 用分页方式管理注意力计算所需的 KV________。

### Q-KB1504-006
模型量化（Quantization）的主要目的是？

### Q-KB1504-013
在部署大模型服务时，为什么要关注“首字延迟（TTFT）”和“生成吞吐量（Tokens/s）”这两个指标？

### Q-KB1504-002
请写出使用 vLLM 的 `LLM` 类加载 `meta-llama/Llama-2-7b-chat-hf` 模型并生成文本的代码。

### Q-KB1504-014
Docker 的主要作用是？

### Q-KB15-ST1501-003
与贪心搜索相比，束搜索的典型代价是？

### Q-KB15-ST1500-001
部署 vLLM 前应确保 CUDA、PyTorch 与 vLLM 三者的________兼容。
