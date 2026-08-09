> **参考文章链接**
> 
> ① 底稿：《DeepSpeed：ZeRO-Infinity 和 ZeRO-Offload 的显存优化机制与训练实践》
> 　　网址：https://blog.csdn.net/m0_60388871/article/details/148958601
> 
> ② 权威文章1：《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》
> 　　网址：https://arxiv.org/abs/1910.02054
> 
> ③ 权威文章2：《DeepSpeed ZeRO 官方教程》
> 　　网址：https://www.deepspeed.ai/tutorials/zero/
> 
> ④ 权威文章3：《DeepSpeed ZeRO-3 官方文档》
> 　　网址：https://deepspeed.readthedocs.io/en/latest/zero3.html
> 
> ⑤ 参考文章1：《DeepSpeed ZeRO-3 超大模型分布式训练实战指南：参数/梯度/优化器三重分片全流程解析》
> 　　网址：https://blog.csdn.net/sinat_28461591/article/details/147553324
> 
> ⑥ 参考文章2：《DeepSpeed 与 ZeRO 详解》
> 　　网址：https://blog.csdn.net/zyw2002/article/details/161458255

---

**适合人群**：已完成 DeepSpeed 环境准备与多 GPU 配置，希望深入理解 ZeRO-1/2/3 三种优化策略的原理差异、显存节省机制与配置方法，并能根据模型规模做出合理选型的进阶学习者。每一步都拆解到"是什么、为什么、怎么配"，你只需要跟着理解。完成本章后，你将能够独立编写 `ds_config.json` 中的 `zero_optimization` 配置，清晰判断 ZeRO-1/2/3 的适用场景，为后续"理解数据并行与模型并行的区别"打下坚实基础。

**预计学习时长**：约 1.5 小时  
**环境准备**：复用上一章的 DeepSpeed 环境（Python 3.10+、PyTorch GPU 版、deepspeed 已安装、至少 2 张 NVIDIA GPU）
---

## 第一章：ZeRO 的定位——为什么需要分片优化？

### 1.1 传统数据并行的显存浪费

在标准的 PyTorch DDP（DistributedDataParallel）训练中，每张 GPU 都保存一份**完整的模型副本**。假设你用 Adam 优化器训练一个 7B 参数的模型，FP16 混合精度下，单卡显存占用构成如下：

| 数据对象 | 计算公式 | 显存占用（7B 模型） |
|---------|---------|------------------|
| 模型参数（FP16） | $2 \times \Phi$ | ~14 GB |
| 梯度（FP16） | $2 \times \Phi$ | ~14 GB |
| 优化器状态（FP32） | $12 \times \Phi$（Adam：参数副本+一阶矩+二阶矩） | ~84 GB |
| **总计** | **$16 \times \Phi$** | **~112 GB** |

**为什么这样写**：Adam 优化器需要为每个参数保存 3 份 FP32 数据（参数副本、一阶动量 $m$、二阶动量 $v$），即 $4\Phi \times 3 = 12\Phi$。加上 FP16 的参数和梯度各 $2\Phi$，总共 $16\Phi$。这意味着**7B 模型训练需要约 112GB 显存**——远超单卡 A100（80GB）的容量。

更关键的是，在 DDP 中，**这些冗余数据在每张卡上都完整复制了一份**。4 张卡训练时，4 张卡各自存了 112GB 的等效数据，其中大部分是完全相同的冗余副本。

### 1.2 ZeRO 的核心思想

ZeRO（Zero Redundancy Optimizer）的核心洞察非常简单：

> **既然每张卡上的优化器状态、梯度、参数都是一模一样的副本，为什么不把它们切开，分散到不同 GPU 上？**

ZeRO 通过**分片（Sharding）**技术，将三类模型训练状态按 GPU 数量切分，每张卡只保存自己负责的那一部分。需要使用时，通过高速 NCCL 通信临时聚合。

```
传统 DDP（4卡）              ZeRO（4卡）
┌─────────┐ ┌─────────┐     ┌─────────┐ ┌─────────┐
│ 参数全量 │ │ 参数全量 │     │ 参数25% │ │ 参数25% │
│ 梯度全量 │ │ 梯度全量 │  →  │ 梯度25% │ │ 梯度25% │
│ 优化器全量│ │ 优化器全量│     │ 优化器25%│ │ 优化器25%│
└─────────┘ └─────────┘     └─────────┘ └─────────┘
   每张卡 112GB               每张卡 ~28GB
```

**为什么这样写**：ZeRO 不是模型并行（不需要改模型结构），而是**数据并行的显存优化升级**。它保留了数据并行的计算效率，同时消除了显存冗余。

---

## 第二章：ZeRO-1——只切分优化器状态

### 2.1 原理详解

ZeRO-1 是最轻量级的 ZeRO 阶段，它只分片**优化器状态**（Optimizer States）。

**保留在每张 GPU 上的数据**：
- ✅ 完整的模型参数（FP16）
- ✅ 完整的梯度（FP16）

**切分的数据**：
- 🔪 优化器状态（FP32 参数副本 + 一阶动量 $m$ + 二阶动量 $v$）

**工作流程**：
1. 前向/反向传播：每张卡独立计算，得到完整梯度
2. 梯度规约：使用 **Reduce-Scatter** 规约梯度，让每张卡只拿到"自己负责的那个参数分片"对应的梯度（注意：这一步只是 Reduce-Scatter，**不包含**把梯度再 All-Gather 回全量）
3. 优化器更新：每张卡用上一步拿到的分片梯度，更新自己负责的优化器状态与参数分片
4. 参数分发：更新后各卡通过 **All-Gather** 交换各自更新过的参数分片，恢复出完整参数

> **说明**：ZeRO-1 的通信由"一次梯度 **Reduce-Scatter** + 一次更新参数 **All-Gather**"构成，与环形 AllReduce 的实现（AllReduce ≈ Reduce-Scatter + All-Gather）处于**同一量级**——所以说它"通信量与 DDP 的一次 AllReduce 大致相当"。常见的两个误解要避免：一是把第 2 步的梯度规约说成"完整的 AllReduce"（那等于多算了一次梯度 All-Gather），二是以为"梯度 Reduce-Scatter + 参数 All-Gather"比 DDP 多出一倍通信。正确的图景是：梯度只 Reduce-Scatter 一次、参数只 All-Gather 一次，数据量没有翻倍，区别主要在通信操作的次数/调度方式上。

### 2.2 显存节省效果

以 Adam 优化器为例，优化器状态占总显存的 $\frac{12}{16} = 75\%$。ZeRO-1 将这部分按 GPU 数量 $N$ 分片：

$$\text{单卡显存} \approx 2\Phi + 2\Phi + \frac{12\Phi}{N} = 4\Phi + \frac{12\Phi}{N}$$

当 $N=4$ 时：
- 传统 DDP：$16\Phi$ = 112 GB
- ZeRO-1：$4\Phi + 3\Phi$ = **约 49 GB**（相比 DDP 约节省 2 倍多；只有当优化器状态占绝对主导、忽略其他开销时才逼近理论上界"4×"）

**为什么这样写**：ZeRO-1 的**通信数据量**与标准 DDP 处于同一量级（DDP 的 AllReduce ≈ ZeRO-1 的 Reduce-Scatter + All-Gather），因此额外通信开销很小。这是它最大的优点。

### 2.3 ds_config.json 配置

```json
{
  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 500000000
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

**为什么这样写**：
- `"stage": 1`：启用 ZeRO-1
- `"reduce_bucket_size": 5e8`：梯度归约的桶大小，限制单次 AllReduce 的内存占用。大模型建议 500M，小模型可减小

### 2.4 适用场景

- 模型**刚刚超出单卡显存一点点**（如 7B 模型在 40GB 显存上训练）
- 希望**通信开销最低**，对训练速度敏感
- 优化器状态是显存瓶颈的主要构成（Adam 场景）

---

## 第三章：ZeRO-2——切分优化器状态 + 梯度

### 3.1 原理详解

ZeRO-2 在 ZeRO-1 的基础上，进一步分片**梯度**（Gradients）。

**保留在每张 GPU 上的数据**：
- ✅ 完整的模型参数（FP16）

**切分的数据**：
- 🔪 优化器状态（FP32）
- 🔪 梯度（FP16）

**工作流程**：
1. 反向传播时，每张卡只计算并保留自己负责的那部分梯度
2. 使用 Reduce-Scatter（而非 AllReduce）将梯度规约到对应卡上
3. 优化器更新时，每张卡只更新自己分到的参数

### 3.2 显存节省效果

ZeRO-2 将优化器状态（$12\Phi$）和梯度（$2\Phi$）都分片：

$$\text{单卡显存} \approx 2\Phi + \frac{2\Phi}{N} + \frac{12\Phi}{N} = 2\Phi + \frac{14\Phi}{N}$$

当 $N=4$ 时：
- 单卡显存：$2\Phi + 3.5\Phi$ = **约 38.5 GB**（相比 DDP 进一步下降；理论上界"8×"同样只在忽略参数与其他开销的理想情况下才逼近）

**为什么这样写**：ZeRO-2 用 Reduce-Scatter 替代 AllReduce 来同步梯度。二者的通信量在理论上处于同一量级（AllReduce ≈ Reduce-Scatter + All-Gather），因此 ZeRO-2 的通信开销仍然较低，与 DDP 大致相当。

### 3.3 ds_config.json 配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

**为什么这样写**：
- `"contiguous_gradients": true`：将梯度复制到连续缓冲区，减少反向传播时的内存碎片
- `"overlap_comm": true`：让梯度通信与反向计算重叠，隐藏通信延迟
- `"reduce_scatter": true`：使用 Reduce-Scatter 替代 AllReduce 进行梯度同步，这是 ZeRO-2 的核心

### 3.4 适用场景（②）

- **最推荐的通用场景**，在显存节省和通信效率之间取得最佳平衡
- 中等规模模型（7B~30B），多卡环境（4~8 卡）
- 希望显著降低显存，但不愿承担 ZeRO-3 的额外通信开销

---

## 第四章：ZeRO-3——三重分片 + CPU/NVMe 卸载

### 4.1 原理详解（①②）

ZeRO-3 是 ZeRO 的完全体，对**参数、梯度、优化器状态**进行三重分片。

**保留在每张 GPU 上的数据**：
- ❌ 无（所有三类数据都被分片）

**切分的数据**：
- 🔪 模型参数（FP16）
- 🔪 梯度（FP16）
- 🔪 优化器状态（FP32）

**工作流程**：
1. **模型初始化**：参数立即被分片，每张卡只保存 $\frac{1}{N}$ 的参数
2. **前向传播**：需要某层参数时，通过 All-Gather 临时聚合该层完整参数，计算完成后立即释放
3. **反向传播**：需要某层参数时再次 All-Gather，梯度计算后通过 Reduce-Scatter 分片保存
4. **优化器更新**：每张卡只更新自己分到的参数分片

### 4.2 显存节省效果

$$\text{单卡显存} \approx \frac{2\Phi}{N} + \frac{2\Phi}{N} + \frac{12\Phi}{N} = \frac{16\Phi}{N}$$

当 $N=8$ 时（理论/理想估算）：
- 单卡显存：$\frac{16\Phi}{8}$ = **约 14 GB**（这是仅考虑三类模型状态、忽略激活值与通信缓冲等开销时的理论下界）

**为什么这样写**：ZeRO-3 的三类模型状态显存占用与 GPU 数量近似**线性反比**。上面的"约 14GB/卡"是理论估算，实际训练还要叠加激活值、临时聚合的参数、通信缓冲等开销，真实占用会更高；但趋势成立——分片让**显存较小的 GPU 也有机会参与训练较大的模型**（具体能否跑起来仍需以实测显存为准）。

### 4.3 通信开销

ZeRO-3 的通信量显著高于 ZeRO-1/2：

- 前向传播：每层需要一次 All-Gather 聚合参数
- 反向传播：每层需要一次 All-Gather 聚合参数 + 一次 Reduce-Scatter 分片梯度
- 总通信量通常明显高于普通 DDP（常见量级约为 **1.5~2 倍**，具体比值随实现、通信是否与计算重叠、bucket 大小等配置而变化）

**为什么这样写**：ZeRO-3 用**通信时间换取显存空间**。在 NVLink/InfiniBand 高速互联环境下，通信开销可以被有效隐藏；但在普通 PCIe 环境下，可能成为瓶颈。

### 4.4 ZeRO-Infinity：卸载到 CPU 和 NVMe

ZeRO-3 还可以进一步将数据卸载到 CPU 内存甚至 NVMe 硬盘，形成 **ZeRO-Infinity**：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

更激进的 NVMe 卸载：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/nvme_data",
      "buffer_count": 4,
      "buffer_size": 100000000,
      "pin_memory": true
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/nvme_data",
      "buffer_count": 5,
      "buffer_size": 100000000,
      "pin_memory": true
    }
  }
}
```

**为什么这样写**：
- `"device": "cpu"`：将优化器状态和参数卸载到 CPU 内存，适合 GPU 显存极度紧张但 CPU 内存充裕的场景
- `"device": "nvme"`：进一步卸载到 NVMe SSD，理论上可以训练**万亿参数模型**
- `"pin_memory": true`：使用页锁定内存，通常有助于提高 CPU↔GPU 的异步传输效率，但会增加页锁定内存占用，属于性能优化选项而非必选项，应结合主机内存和实测性能决定是否启用

### 4.5 ds_config.json 完整配置（①）

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,
      "betas": "auto",
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": "auto",
      "warmup_max_lr": 5e-5,
      "warmup_num_steps": 400
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 50000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 10000,
    "stage3_max_live_parameters": 30000000,
    "stage3_max_reuse_distance": 1000000000
  },
  "fp16": {
    "enabled": true,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "gradient_clipping": 1.0,
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  }
}
```

**关键参数解读**：

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `stage3_prefetch_bucket_size` | 参数预取的桶大小 | 50M~100M，与 `reduce_bucket_size` 保持一致 |
| `stage3_param_persistence_threshold` | 小于此阈值的参数不分片 | 10K，小参数（如 LayerNorm）直接保留在 GPU 上，减少通信 |
| `stage3_max_live_parameters` | 单卡上同时保留的最大参数量 | 30M，控制峰值显存 |
| `stage3_max_reuse_distance` | 参数复用距离阈值 | 1B，短期内会复用的参数不释放 |

### 4.6 适用场景（②）

- **超大模型**（70B+ 参数）或只有小显存 GPU 的环境
- 需要**极致显存节省**，愿意用通信开销换取训练可行性
- 多节点高带宽环境（InfiniBand/NVLink），能有效隐藏通信延迟

---

## 第五章：ZeRO-1/2/3 全面对比与选型

### 5.1 核心维度对比表

| 维度 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|--------|--------|--------|
| **分片优化器状态** | ✅ | ✅ | ✅ |
| **分片梯度** | ❌ | ✅ | ✅ |
| **分片模型参数** | ❌ | ❌ | ✅ |
| **显存节省** | 最高约 4×（理论上界，4 卡实际约 2.3×） | 最高约 8×（理论上界，实际更低） | 线性（随 GPU 数 1/N） |
| **通信开销** | 与 DDP 相同（低） | 与 DDP 相同（低） | 高（1.5~2× DDP） |
| **保留的完整数据** | 参数 + 梯度 | 参数 | 无 |
| **优化器状态 Offload（CPU）** | ✅（`offload_optimizer` 对 Stage 1/2/3 均可用，官方多以 Stage 2 讲解 ZeRO-Offload） | ✅（ZeRO-Offload 可在 Stage 2 卸载优化器状态与优化器计算） | ✅ |
| **参数 Offload（CPU/NVMe）** | ❌ | ❌ | ✅（参数卸载、NVMe 卸载主要与 Stage 3 / ZeRO-Infinity 组合） |
| **代码修改** | 无（仅改配置） | 无（仅改配置） | 几乎无（HF 集成时无） |
| **推荐场景** | 小模型、低通信 | **通用首选** | 超大模型、小显存 |

### 5.2 选型决策树

```
模型能否放进单卡显存？
├── 能 → 不需要 ZeRO，直接用 DDP 或单卡训练
└── 不能 → 进入下一步

优化器状态是否占主导（Adam + FP16）？
├── 是，且模型不太大 → ZeRO-1
└── 否 → 进入下一步

模型参数 + 梯度是否也放不下了？
├── 是，但通信带宽充足 → ZeRO-3
└── 否，或希望平衡 → ZeRO-2（推荐）
```

### 5.3 性能趋势参考（①）

以在多卡 A100 环境下训练某个十亿级参数量 GPT-like 模型为例，三种配置呈现出如下**相对趋势**（下表为**定性示意，具体数值随模型结构、序列长度、batch 大小、卡间互联、DeepSpeed 版本等因素波动很大，务必以自己环境的实测为准**）：

| 指标 | ZeRO-2 | ZeRO-3（GPU 内） | ZeRO-3（+CPU Offload） |
|------|--------|-----------------|----------------------|
| 单卡显存占用 | 最高 | 明显下降 | 进一步下降 |
| 训练吞吐 | 基准 | 有机会持平甚至提升 | 通常下降 |
| 每轮训练时间 | 基准 | 有机会缩短 | 通常变长 |

**为什么这样写**：ZeRO-3 纯 GPU 内分片时，显存释放出来后可以适当增大 micro batch，从而**有机会**在吞吐上追平甚至反超 ZeRO-2；但一旦加入 CPU Offload，CPU↔GPU 数据传输往往成为瓶颈，吞吐通常回落。因此需要记住这条**定性结论**：**卸载是"救命"手段，不是性能优化手段**——它用来换取"训练能不能跑起来"，而不是"跑得更快"。

---

## 第六章：电脑配置批注

本章节涉及 ZeRO 各阶段的本地/云端多 GPU 训练配置。

### 6.1 ZeRO-1 配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 2 张 NVIDIA GPU，显存 ≥16GB/卡 | 4~8 张 A100（40GB/80GB） |
| CPU | 8 核 | 16 核+ |
| 内存 | 32GB | 64GB+ |
| 磁盘 | 100GB SSD | 500GB+ NVMe SSD |
| 网络 | 单节点无需特殊网络 | 多节点需 10Gbps+ 以太网 |
| 是否支持 CPU 运行 | ❌ 不支持（需 GPU） | — |

### 6.2 ZeRO-2 配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 2 张 NVIDIA GPU，显存 ≥16GB/卡 | 4~8 张 A100（80GB） |
| CPU | 8 核 | 32 核+ |
| 内存 | 32GB | 128GB+ |
| 磁盘 | 100GB SSD | 500GB+ NVMe SSD |
| 网络 | 单节点无需特殊网络 | 多节点需 InfiniBand |
| 是否支持 CPU 运行 | ❌ 不支持 | — |

### 6.3 ZeRO-3 配置要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 2 张 NVIDIA GPU，显存 ≥12GB/卡 | 8 张 A100（80GB）+ NVLink |
| CPU | 16 核 | 64 核+（CPU Offload 时 CPU 核心数直接影响速度） |
| 内存 | 64GB | 256GB+（CPU Offload 时需要大量 CPU 内存） |
| 磁盘 | 200GB SSD | 1TB+ NVMe SSD（NVMe Offload 时需要） |
| 网络 | 单节点 PCIe 可用 | 多节点必须 InfiniBand |
| 是否支持 CPU 运行 | ❌ 不支持 | — |

### 6.4 云端配置参考

| 场景 | 推荐实例 | ZeRO 阶段 |
|------|---------|----------|
| 低成本实验（7B 模型） | AutoDL 2×RTX 3090 | ZeRO-2 |
| 标准训练（13B~30B） | AWS p4d.24xlarge（8×A100） | ZeRO-2 / ZeRO-3 |
| 超大模型（70B+） | 多节点 A100 集群 | ZeRO-3 + CPU Offload |
| 极限场景（千亿参数） | 多节点 + NVMe 节点 | ZeRO-3 + NVMe Offload |

---

## 第七章：常见问题与避坑清单

| 坑点 | 现象 | 解决方案 |
|------|------|---------|
| **ZeRO-3 启动极慢** | 模型初始化耗时数分钟 | 正常现象（参数分片需要 All-Gather），大模型可用 `deepspeed.zero.Init()` 加速 |
| **CPU Offload 后训练变慢** | 吞吐下降 30%~50% | CPU↔GPU 传输是瓶颈，增加 CPU 核心数、开启 `pin_memory`、使用 NVMe 替代 CPU |
| **`stage3_gather_16bit_weights_on_model_save`** | ZeRO-3 保存的 checkpoint 只有占位符 | 在 `ds_config.json` 中设置 `"stage3_gather_16bit_weights_on_model_save": true`，或用 `deepspeed.save_16bit_model()` |
| **ZeRO-3 与模型并行混用** | 报错或死锁 | ZeRO-3 本身已大幅节省显存，通常不需要再叠加模型并行。如需混用，确保 `mp_size` 与 ZeRO 分片兼容 |
| **小参数被错误分片** | LayerNorm/Bias 层通信频繁 | 调大 `stage3_param_persistence_threshold`（如 1e6），让小参数保留在 GPU 上 |
| **通信超时** | NCCL timeout | 检查网络连通性；多节点时确保 `MASTER_ADDR`/`MASTER_PORT` 正确；设置 `NCCL_IB_DISABLE=1` 回退到以太网 |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **ZeRO 核心思想** | 分片消除冗余：不再每张卡存完整副本，而是将数据切分分散 |
| **ZeRO-1** | 只分片优化器状态，显存节省最高约 4×（理论上界，4 卡实际约 2.3×），通信数据量与 DDP 同量级 |
| **ZeRO-2** | 分片优化器状态 + 梯度，显存节省最高约 8×（理论上界，实际更低），通信开销仍低，**通用首选** |
| **ZeRO-3** | 三重分片（参数+梯度+优化器状态），显存随 GPU 数线性下降，通信开销高 |
| **ZeRO-Infinity** | ZeRO-3 + CPU/NVMe 卸载，可训练万亿参数模型 |
| **关键配置** | `stage`、`contiguous_gradients`、`overlap_comm`、`offload_optimizer/param` |
| **选型原则** | 小模型/低通信 → ZeRO-1；通用场景 → ZeRO-2；超大模型/小显存 → ZeRO-3 |
| **性能权衡** | 卸载是"救命"不是"加速"，纯 GPU 内 ZeRO-3 反而可能因 batch 增大而提速 |

---

## 养成的好习惯

1. **从 ZeRO-2 开始尝试**——不要一上来就用 ZeRO-3，ZeRO-2 在大多数场景下是最佳平衡点
2. **监控显存占用**——使用 `nvidia-smi` 或 `torch.cuda.memory_summary()` 确认每卡显存，验证分片效果
3. **batch size 是调优杠杆**——ZeRO 节省的显存应该用来增大 micro batch，提高 GPU 计算利用率
4. **卸载是最后手段**——优先尝试纯 GPU 内的 ZeRO-3，只有当显存仍然不足时才启用 CPU/NVMe Offload
5. **保存 checkpoint 前检查**——ZeRO-3 默认不保存完整权重，确保 `stage3_gather_16bit_weights_on_model_save` 已开启
6. **小参数不分片**——合理设置 `param_persistence_threshold`，避免 LayerNorm 等小层频繁通信

---

## 下一步预告

掌握了 ZeRO-1/2/3 的优化策略配置后，你已经能够根据模型规模和硬件条件灵活选择分布式训练的显存优化方案。接下来，你将进入 **"理解数据并行与模型并行的区别"** 的学习——从原理层面厘清 DP、MP、PP（流水线并行）、TP（张量并行）四种并行策略的工作机制、通信模式和适用边界，并学会如何在实际项目中组合使用这些策略。这是从"会用 DeepSpeed"进化为"懂分布式训练系统设计"的关键一步，继续加油！