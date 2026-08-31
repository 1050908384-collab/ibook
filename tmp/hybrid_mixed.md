## 「显卡监控命令 nvidia-smi 详解」混合推进学习
#### 讲解 ⇄ 实操交替节拍卡（左列填笔记、右列粘贴每一步的终端输出）
| 小节 | 读到讲解段时，你预判要讲什么？ | 紧接着的代码段实际运行输出（立即粘贴） | 执行后再看下段讲解时修正了哪些理解？ |
|---|---|---|---|
| 第 1 对讲解/代码 |  |  |  |
| 第 2 对 |  |  |  |
| 第 3 对 |  |  |  |
---
# 显卡监控命令 nvidia-smi 详解

在深度学习训练过程中，监控 GPU 状态是定位问题的第一步。**nvidia-smi** 是 NVIDIA 官方提供的命令行工具，用于查看当前系统中 GPU 的实时状态信息。你已经掌握 Linux 基础命令，这里直接上手实操即可。

💡 【先读讲解 →】

## 基础用法

直接执行：

🛠 【→ 实际运行下面示例 →】

```
nvidia-smi
```

💡 【→ 执行完毕再看下一段讲解 →】

会输出一张包含下面 **核心字段** 的总览表：

| 字段 | 含义 |
|---|---|
| Driver Version | 当前已安装的 NVIDIA 驱动版本 |
| CUDA Version | 驱动支持的最高 CUDA 版本 |
| GPU Name | GPU 型号，例如 RTX 4060 / A100 |
| Memory-Usage | 显存占用（已用 / 总量） |
| GPU-Util | GPU 计算核心利用率（0% ~ 100%） |
| Temperature | GPU 温度（摄氏度） |

> 你的实训机是联想拯救者 Y7000P，搭载 RTX 4060 8G 显存。执行 `nvidia-smi` 后，重点看 **GPU Name** 是否显示 RTX 4060、**Memory-Usage** 的 8G 总量是否识别正常。若显存总量不是 8G，说明驱动或硬件识别有问题，需要排查。

💡 【先读讲解 →】

## 常见报错与排查

- **`NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`**
  - 现象：执行 nvidia-smi 时直接报错，没有任何表格输出。
  - 排查 1：确认你机器上有 NVIDIA 显卡并且驱动已安装。
  - 排查 2：重启系统后再试；很多时候驱动升级后未重启导致与内核模块失联。
  - 排查 3：在 Linux 上执行 `lsmod | grep nvidia`，如果没有输出，说明 nvidia 内核模块没加载。

🛠 【→ 实际运行下面示例 →】

```
lsmod | grep nvidia
```

💡 【→ 执行完毕再看下一段讲解 →】

- **显存 OOM（Out of Memory）**
  - 现象：训练脚本抛出 `RuntimeError: CUDA out of memory. Tried to allocate X GiB`。
  - 排查 1：用 nvidia-smi 看 Memory-Usage，找出占用显存最高的进程 PID（表格最右侧 Process Name 列）。
  - 排查 2：对于不再需要的进程执行 `kill <PID>` 或 `kill -9 <PID>` 释放显存。
  - 排查 3：训练代码层面降低 batch_size，或者启用 mixed precision / gradient checkpointing。

> 你的 8G 显存跑大模型时很容易触发 OOM。遇到 `RuntimeError: CUDA out of memory` 时，按上面三步走：先看显存占用找元凶进程 → 杀掉无用进程 → 代码层面降 batch_size 或开混合精度。

💡 【先读讲解 →】

## 进阶参数

- `nvidia-smi dmon -s p`：持续监控功率 / 温度 / SM 占用等多项指标每秒一行。
- `nvidia-smi -l 1`：每 1 秒刷新一次总览表（等同 `watch -n 1 nvidia-smi`）。
- `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits`：CSV 格式导出，便于写入日志。

🛠 【→ 实际运行下面示例 →】

```
nvidia-smi -l 1
```

> 按 `Ctrl+C` 停止刷新。进阶参数里，`-l 1` 适合实时盯训练过程；`--query-gpu` 适合写脚本定时记录 GPU 状态到日志文件，训练完回溯分析。

### 边学边做 · 本块小结
1. 讲解⇄实操交替笔记：围绕第 1 段代码 `nvidia-smi`，写『执行前我从讲解段理解到它能查看 GPU 实时状态 → 实际执行后看到总览表 → 再读讲解修正了哪些理解』三部分短笔记。
2. 概念 ⇄ 现象对应：任取一个 SOURCE 加粗重要术语（如 **Memory-Usage**）+ 你执行输出中的任意 1 个字段（如 1234MiB / 8192MiB）；说明对应关系或分别属于哪个层面。
3. 误区与非预期：写 2 条你在执行或阅读中遇到的误区/非预期结果；说明如何定位解决。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（掌握 Linux 基础命令）的关系；遇到无法解释的术语时记录下来再提问。