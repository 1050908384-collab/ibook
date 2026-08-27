# 学习包

## 块 1

# 第一章：PagedAttention 的定位——为什么需要分页管理 KV Cache？

## 1.1 传统 KV Cache 管理的两大浪费

在大模型自回归推理中，每生成一个新 token，都需要缓存该 token 对应的 Key 和 Value 向量（即 KV Cache）。随着序列增长，KV Cache 不断膨胀，成为推理显存的主要消耗者。

在 vLLM 出现之前，传统推理框架（如 FasterTransformer、ONNX Runtime）采用**静态预分配**策略：

> **为每个请求预先分配一块按最大序列长度计算的连续显存空间。**

例如，模型支持的最大上下文为 2048 tokens，那么即使某个请求实际只生成 50 个 token，也要为其预留能装 2048 tokens 的连续 KV Cache 空间。

这导致了两种严重的显存浪费（①）：

1. **资源浪费**：为每个请求预留了远超实际需求的显存空间，导致显存利用率低下。
2. **性能瓶颈**：在处理短序列请求时，由于预留了大量冗余空间，系统运行效率降低。

这种静态预分配的管理方式，无法有效适应实际推理场景中序列

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

### 2.2 Block Table——逻辑到物理的映射（②③）

每个请求维护一个**块表（Block Table）**，记录该请求的逻辑 token 位置对应到哪个物理 Block：

```
请求当前有 50 个 token（需要约 4 个 Block，50÷16=3.125→向上取整=4）

Block Table（逻辑 Block → 物理 Block ID）：
  逻辑 Block 0 → 物理 Block 23   （存 token 0~15，已满）
  逻辑 Block 1 → 物理 Block 45   （存 token 16~31，已满，且不连续！）
  逻辑 Block 2 → 物理 Block 7    （存 token 32~47，已满）
  逻辑 Block 3 → 物理 Block 12   （存 token 48~49，只用了 2 个槽位，还没满）

当生成第 51、52…个 token 时：
  - Block 3 还有空槽（16 个槽位里只用了 2 个），新 token 继续写入 Block 3
  - 直到 Block 3 写满 16 个 token（即写完逻辑位置 48~63）后，
    才从全局 Free Block Pool 申请一个新的空闲 Block（比如 Block 67），
    并更新 Block Table：逻辑 Block 4 → 物理 Block 67
```

**分配粒度说明**：Block 是"写满 `block_size` 个 token 才换下一块"的分配粒度。50 个 token 时逻辑 Block 3 只占用了 2 个槽位、远未写满，因此后续 token 应继续追加进 Block 3，而不是立刻申请新块——只有当前 Block 的 16 个槽位全部用完，才会向 Free Block Pool 索取新块。这正是分页分配"按块粒度、块内连续填充"的关键。

**为什么这样写**：Block Table 是 PagedAttention 的核心数据结构。查询时只需计算 `block_number = position // block_size` 和 `offset = position % block_size`，即可通过 `block_table[block_number]` 找到物理 Block，时间复杂度 O(1)。

（个性化提示：学习风格：hands-on）

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

# 运行时政策

## 运行时政策

### 3.3 Scheduler 调度器的核心逻辑（②③）

vLLM 的调度器在每一步执行

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

```markdown
# 4.2 PagedAttention与Continuous Batching配合工作流程图解析

## 学习目标
理解PagedAttention和Continuous Batching如何在模型推理中配合工作。

##

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

```markdown
# vLLM部署：PagedAttention与Continuous Batching机制

## 学习目标
- 理解vLLM的PagedAttention机制
- 掌握Continuous Batching的工作原理
- 学习如何通过参数调整提升模型吞吐量

## 学习重点
1. **PagedAttention机制**
   - 借鉴自操作系统虚拟内存分页技术
   - 将KV缓存划分为固定大小的块
   - 实现高效的内存管理
   - 适合大规模模型推理

2. **Continuous Batching机制**
   - 实现GPU上的流水线处理
   - 每次请求动态调度
   - 提升GPU利用率
   - 降低模型调用门槛

3. **关键参数与吞吐关系**
   - `block_size`：每个块存储的token数，默认为16，需根据模型规模调整
   - `gpu_memory_utilization`：GPU显存使用比例，默认控制在0.85~0.95
   - `max_num_seqs`：最大并发序列数，默认根据显存容量设置256~512
   - `enable_prefix_caching`：开启前缀缓存以减少重复计算

## 学习任务
1. **实验验证**
   - 设置`block_size`为16，观察模型运行时的吞吐量
   - 将`gpu_memory_util

## 块 19

## Runtime Policy

- Treat each source chunk as the complete factual boundary. Personalize by selecting, compressing, reordering, re-explaining, and emphasizing source-supported material for the learner.
- Preserve factual traceability. Do not add unsupported technical facts, commands, parameters, or cross-topic claims.

---

## Block Revision: vLLM

## 测评题目

### Q-KB15-ST1503-002
Continuous Batching 的核心是让不同请求在生成过程中________加入或离开批次。

### Q-KB15-ST1503-001
vLLM 的 PagedAttention 用分页方式管理注意力计算所需的 KV________。

### Q-KB15-ST1500-002
vLLM 推理服务通常需要 NVIDIA GPU 及足够的________。

### Q-KB15-ST1502-002
Top-k 采样每一步只在概率最高的 k 个候选________中采样。

### Q-KB1504-007
GGUF 格式是由 ________ 社区主推的模型格式，专为 CPU 和混合 CPU/GPU 推理（如 llama.cpp）设计。

### Q-KB15-ST1502-001
Temperature 增大通常会使采样分布更________。

### Q-KB15-ST1501-003
与贪心搜索相比，束搜索的典型代价是？

### Q-KB1504-006
模型量化（Quantization）的主要目的是？
