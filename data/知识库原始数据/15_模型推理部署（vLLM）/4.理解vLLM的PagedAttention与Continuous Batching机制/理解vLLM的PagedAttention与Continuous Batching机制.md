> **参考文章链接**
> 
> ① 底稿：《【infra之路】阶段三 · 推理线：vLLM 两大法宝原理 — PagedAttention & Continuous Batching》
> 　　网址：https://blog.csdn.net/sweet_ran/article/details/161777519
> 
> ② 参考文章1：《vLLM 底层 PagedAttention（分页注意力）和 Continuous Batching（连续批处理）解释》
> 　　网址：https://blog.csdn.net/shizheng_Li/article/details/160184213
> 
> ③ 参考文章2：《vLLM 核心源码深度解析：PagedAttention 的KV Cache 管理、Scheduler 调度引擎与Continuous Batching 实现全解》
> 　　网址：https://blog.csdn.net/a13662080711/article/details/162737553

---

> **本板块目标版本（示例，请以你实际安装的发行为准）**
> - vLLM：v0.6.4（本文正文示例均以该版本的可运行接口为准；旧版写法见文末"历史版本差异"附录，如无则以正文为准）
> - Python：3.10~3.12（以该 vLLM 版本的官方发行说明为准）
> - PyTorch / CUDA wheel：以 `pip install vllm==0.6.4` 实际解析到的依赖及该发行说明为准（如常见 cu121）
> - GPU：示例按常见 NVIDIA GPU（具体型号 / 显存以你的机器为准）
> - 文档审校日期：2026-08（内容为静态审校，未在多 GPU 环境逐条实机验证）

---

**适合人群**：已掌握 vLLM 环境准备和采样参数配置，希望深入理解 vLLM 两大核心机制——PagedAttention 和 Continuous Batching——的工作原理、配合关系与源码实现，从而真正理解"为什么 vLLM 在高并发下往往比朴素推理快出一个数量级"的进阶学习者。每一步都拆解到"是什么、为什么、怎么配合"，你只需要跟着理解。完成本章后，你将能够从源码级视角解释 PagedAttention 的显存管理、Continuous Batching 的请求调度，以及两者如何协同工作，为后续"使用 vLLM 加载模型并配置 gpu_memory_utilization 等关键参数进行推理部署"打下坚实基础。

**预计学习时长**：约 1.5 小时  
**环境准备**：复用上一章的 vLLM 环境（Python 3.10~3.13、vLLM 已安装、NVIDIA GPU 显存 ≥16GB）

---

## 第一章：PagedAttention 的定位——为什么需要分页管理 KV Cache？

### 1.1 传统 KV Cache 管理的两大浪费

在大模型自回归推理中，每生成一个新 token，都需要缓存该 token 对应的 Key 和 Value 向量（即 KV Cache）。随着序列增长，KV Cache 不断膨胀，成为推理显存的主要消耗者。

在 vLLM 出现之前，传统推理框架（如 FasterTransformer、ONNX Runtime）采用**静态预分配**策略：

> **为每个请求预先分配一块按最大序列长度计算的连续显存空间。**

例如，模型支持的最大上下文为 2048 tokens，那么即使某个请求实际只生成 50 个 token，也要为其预留能装 2048 tokens 的连续 KV Cache 空间。

这导致了两种严重的显存浪费（①）：

| 浪费类型 | 说明 | 类比 |
|---------|------|------|
| **内部碎片（Internal Fragmentation）** | 预留了 2048，实际只用了 50，剩余 1998 的空间被白白占用 | 租了 100 平米的办公室，只坐了 5 个人 |
| **外部碎片（External Fragmentation）** | 每个请求需要连续的大块显存，导致显存被切成大小不一的碎片，总空闲够但找不到足够大的连续空间 | 停车场有很多空位，但都是分散的小车位，大巴停不进去 |

**为什么这样写**：在这种静态预分配方案下，KV Cache 的显存利用率往往只能达到**几成左右**（不同来源常给出约 20%~40% 的经验区间，具体数值与序列长度分布、最大长度设置、请求模式强相关，仅为示意），相当一部分显存被碎片浪费。而显存是推理的命根子——可用于 KV Cache 的显存越多，能支持的并发请求数和上下文长度就越长。浪费显存 = 直接限制吞吐量和用户体验。

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

### 2.1 Block 的定义（③）

PagedAttention 将 KV Cache 切成**固定大小的块（Block）**，默认每个 Block 存储 16 个 token 的 Key 和 Value：

```
一个 Block = 16 个 token 的 K/V 向量

用于"估算容量"的概念形状（不代表单个 Block 的真实内存布局）：
[num_layers, 2(K/V), num_heads, block_size=16, head_dim]
```

**为什么这样写**：`block_size=16` 是部分 vLLM CUDA 版本上常见的默认值，这个大小在 GPU 内存访问对齐和显存粒度之间取得平衡：太小会增加块表管理开销，太大会增加内部碎片。需要注意，**默认值与可选范围可能随引擎版本、硬件平台变化**，这里的 16 只是特定版本/平台的示例，不是永久固定规则；实际请以 `vllm serve --help` 或所装版本文档为准。

> **关于上面的"形状"**：`[num_layers, 2(K/V), num_heads, block_size, head_dim]` 是一个**便于估算 KV Cache 总容量的概念形状**（把各层、K/V 两份、注意力头都乘进去，方便算显存），**并不代表单个物理 Block 在显存里的真实内存布局**。实际实现中 KV Cache 通常**按层分别维护**（每层各自持有若干 Block），K 与 V 是否分开存放、`num_heads` 用的是 Query 头数还是 KV 头数（GQA/MQA 场景）、各维度的先后顺序等，都取决于具体 backend、精度与 vLLM 版本。这里给出该形状只是为了帮助你估算容量，不要据此推断底层张量的排布。

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

### 6.1 本地环境配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 1 张 NVIDIA GPU，显存 **≥16GB** | 1 张 RTX 4090（24GB）/ A100（40GB/80GB） |
| **CPU** | 8 核 | 16 核+ |
| **内存** | 32GB | 64GB+ |
| **磁盘** | 100GB SSD | 500GB+ NVMe SSD |
| **系统** | Linux（Ubuntu 20.04+）或 WSL2 | 原生 Linux（Ubuntu 22.04+） |
| **是否支持 CPU 运行** | ⚠️ 支持但极慢（仅调试） | — |

### 6.2 云端环境配置参考

| 场景 | 推荐实例 | PagedAttention 效果 |
|------|---------|-------------------|
| 低成本实验 | AutoDL RTX 3090（24GB）×1 | 7B 模型，并发 8~16 |
| 标准推理 | AWS g5.xlarge（A10G 24GB） | 7B~13B 模型，并发 16~32 |
| 高并发服务 | AWS g5.12xlarge（4×A10G） | 70B 模型（TP=4），并发 64+ |
| 超长上下文 | AWS p4d.24xlarge（8×A100） | 128K 上下文，Prefix Caching 效果显著 |

### 6.3 关键参数与吞吐关系

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `block_size` | 每个 Block 存储的 token 数 | 特定版本/平台常见默认 16，一般不改；不同引擎版本/硬件可能不同 |
| `gpu_memory_utilization` | GPU 显存使用比例 | 0.85~0.95 |
| `max_num_seqs` | 最大并发序列数 | 根据显存调整，256~512 |
| `enable_prefix_caching` | 前缀缓存开关 | 存在较长的**共享前缀**且命中率较高时（如统一 system prompt、多轮 Chat、RAG）建议开启，并实测 TTFT/吞吐/命中率再决定是否常开 |

---

## 第七章：常见问题与避坑清单

| 坑点 | 现象 | 解决方案 |
|------|------|---------|
| **Block 大小设置不当** | `block_size` 太小导致块表过大，管理开销高；太大导致内部碎片 | 通常保持所装版本的默认值（部分版本为 16），除非有特殊需求并已实测 |
| **Prefix Caching 未开启** | 相同 system prompt 的多个请求各存一份 KV | 设置 `enable_prefix_caching=True` |
| **显存利用率设太高** | OOM 或频繁 preemption（请求被抢占重算） | 降低 `gpu_memory_utilization` 到 0.85 |
| **不理解 Paging 和 Batching 的关系** | 面试答不全 | 记住：**调度的灵活性建立在显存管理的灵活性之上** |
| **静态批处理思维** | 用 vLLM 但按传统方式组织请求 | 让 vLLM 自动调度，不要手动凑 batch |
| **忽略 Chunked Prefill** | 大 prefill 阻塞 decode，ITL 升高 | 确保 vLLM V1 默认开启，或手动启用 |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **PagedAttention 思想** | 借鉴 OS 虚拟内存分页，将 KV Cache 切成固定 Block，按需非连续分配 |
| **Block** | K/V 存储的最小分配粒度；块内可容纳的 token 数由 `block_size` 决定（部分版本默认 16，随版本/平台变化） |
| **Block Table** | 记录逻辑 token 位置到物理 Block ID 的映射，查询 O(1) |
| **解决碎片** | 把内部碎片限制在每条序列**最后一个未写满的 Block**（最多浪费 `block_size-1` 个槽位），并基本消除对**连续物理地址**的要求（外部碎片），显存利用率从"只有几成"提升到接近满载（百分比为示意） |
| **Continuous Batching** | 每步迭代动态调度，完成即释放、新请求即加入，GPU 持续满载 |
| **Scheduler 策略** | Decode 优先 + Prefill 填充 + Chunked Prefill 防阻塞 |
| **两大机制配合** | Paging 管显存回收，Batching 管请求调度，后者建立在前者之上 |
| **Prefix Caching** | 相同前缀请求只读共享 Block，后缀写各自私有 Block；Copy-on-Write 仅在需修改共享 Block（如并行采样/束搜索）时触发 |
| **吞吐提升** | PagedAttention（省显存、提并发）与 Continuous Batching（提 GPU 利用率）叠加，通常能带来数倍到约一个数量级的吞吐提升（具体倍数依模型、序列长度、并发与对比基线而定，需实测，网上流传的"×N"数字多为特定场景示意） |

---

## 养成的好习惯

1. **把 Paging 和 Batching 当成一对来记**——单独理解任何一个都不完整，两者的配合才是 vLLM 快的原因
2. **面试标准答法**：先讲 PagedAttention 解决显存碎片，再讲 Continuous Batching 解决调度浪费，最后强调"调度灵活性建立在显存管理灵活性之上"
3. **共享前缀场景优先开 Prefix Caching**——RAG 和多轮 Chat 里相同前缀只存一份，能省显存又省计算，但收益取决于前缀占比和命中率，需以实测为准
4. **不要手动凑 batch**——vLLM 的 Scheduler 比你更聪明，把请求丢进去让它自动调度
5. **监控 preemption 次数**——如果频繁出现请求被抢占重算，说明 `gpu_memory_utilization` 设太高了
6. **Block size 一般沿用默认**——部分版本默认 16 已是常见调优结果，默认值随版本/平台可能不同，除非你有明确的性能分析数据再调整

---

## 下一步预告

掌握了 PagedAttention 和 Continuous Batching 的核心机制后，你已经理解了 vLLM 高性能的底层原理。接下来，你将进入 **"能够使用 vLLM 加载模型并配置 gpu_memory_utilization 等关键参数进行推理部署"** 的实战环节——亲手启动 vLLM 服务，配置 `gpu_memory_utilization`、`max_num_seqs`、`tensor_parallel_size` 等关键参数，监控吞吐量和延迟指标，并将模型部署为生产可用的 OpenAI 兼容 API 服务。这是从"懂原理"进化为"能上线"的关键一步，继续加油！