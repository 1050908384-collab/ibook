# -*- coding: utf-8 -*-
"""补齐 KB-01 KBS-01-04（nvidia-smi 显存监控）原始 seed 数据缺失的第一章。

问题：原始 LOCAL-KBS-01-04 的 chunks 全部从 "2.1 最简命令"、"二、基础用法" 起步，
缺少 "一、nvidia-smi 是什么 / 定位 / 能解决什么问题" 的开篇章节，导致前端第一屏
直接显示 "## 二、基础用法"，用户以为第一章被删了。

做法：在 knowledge_chunks 表 KB-01 的 nvidia-smi 归属 chunks 之前，插入两个开篇 chunk：
  - LOCAL-KBS-01-04-S01（序章.1）：nvidia-smi 是什么工具？解决什么问题？（定位+能力概览）
  - LOCAL-KBS-01-04-S02（序章.2）：使用前快速检查（驱动预装、基础准入条件）

保持 block_id=KB-01；从同一 doc_id（查当前 C001 的 doc_id 继承）；chunk_index 设为
0.x（在现有 chunk_index=1 之前）。heading_path 前缀与本块一致
("nvidia-smi 显存监控 > 一、工具定位：它是干什么的？ > 1.1 ...")。

新 chunks 的内容按其他子主题（KBS-01-0x）"第一章" 写法构造，符合原始知识库风格：
① 标记为主，正文口语化 + 1 个代码块 + 思考题/提示。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")

CH1_1_1 = {
    # 1.1 开篇总述
    "chunk_id": "LOCAL-KBS-01-04-S01",
    "heading_path": (
        "nvidia-smi 显存监控 > 一、工具定位：它是什么？我什么时候用？（①）"
        " > 1.1 nvidia-smi 是什么？一句话定义（①）"
    ),
    "content": """## 一、工具定位：它是什么？我什么时候用？（①）

### 1.1 nvidia-smi 是什么？一句话定义（①）

`nvidia-smi` = **NVIDIA System Management Interface**，中文直译是「英伟达 系统管理接口」。

你可以把它理解成：**NVIDIA 官方为 GPU 量身定制的一个"命令行任务管理器"**。

- 你在 Windows 上打开「任务管理器」→「性能」→「GPU」那一栏看到的所有指标，
  在 Linux / 服务器环境里，基本都能用这一条命令打印到终端上。
- 它不只是"看看显存"那么简单：GPU 利用率、温度、功耗、当前占用显存的进程列表、
  驱动版本 / CUDA 版本、甚至 GPU 拓扑结构，它都能查。
- 最重要的一点：**它随 NVIDIA 显卡驱动一起安装**，装好驱动就自带，不需要额外 pip
  或 apt 装任何东西，零配置成本。

> 为什么我们这门课把它放在"命令行操作"的学习目标里？
> 因为它是**你在训练大模型 / 跑推理服务时第一个会被要求"先跑一下看看"的命令**。
> 老师说"你先贴个 nvidia-smi 的截图"，意思就是"先把你当前 GPU 的体检单给我看"——
> 这是 AI 开发 / 运维场景里的"通行问候语"。

**一个最朴素的直觉**：任何涉及"显卡是不是被占满了 / 训练为什么这么慢 / 为什么报 OOM / 有没有真的在用 GPU 跑"的问题，第一步都是敲一下它。
""",
}

CH1_1_2 = {
    # 1.2 准入条件 + 快速验证
    "chunk_id": "LOCAL-KBS-01-04-S02",
    "heading_path": (
        "nvidia-smi 显存监控 > 一、工具定位：它是什么？我什么时候用？（①）"
        " > 1.2 使用前的快速检查（①）：你需要满足什么前提才能看到有效输出？"
    ),
    "content": """### 1.2 使用前的快速检查（①）：你需要满足什么前提才能看到有效输出？

`nvidia-smi` 不是任何机器都能跑通的。在正式学它的用法之前，先做 30 秒准入检查：

**必要条件（三条缺一不可）**：

1. 机器里必须有 **NVIDIA 品牌** 的独立显卡（AMD / 核显 / Apple Silicon Mac 都不行）；
2. 已经安装好了 **对应版本的 NVIDIA 显卡驱动**（不是 CUDA Toolkit，是驱动本身）；
3. 如果是在**云服务器 / WSL2 / 虚拟机**里，还必须提前把 GPU"透传"或"挂载"到当前环境。

**验证步骤（①）**：打开终端，敲：

```bash
nvidia-smi
```

**会出现三种情况**：

- ✅ **情况 A（正常）**：终端输出一张包含 "NVIDIA-SMI / Driver Version / CUDA Version"
  表头、下面有 GPU 编号和名称的大表格。说明环境已就绪，本小节后续内容直接学。
- ❌ **情况 B（驱动未装 / 显卡不识别）**：终端提示
  `nvidia-smi: command not found` 或
  `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`。
  说明驱动没装 / 内核模块没加载，先退回去装驱动，装好在回来。
- ❌ **情况 C（无 GPU 环境）**：你的机器就是纯 CPU、或者 AMD 显卡。这种情况下
  `nvidia-smi` 无法使用——这很正常，后续不带 GPU 的基础命令学习不受影响；
  但本小节所有 GPU 显存相关数值你看不到表格，把它当阅读材料过一遍即可，
  等有 GPU 环境再实操。

> **小提示**：如果你是 Windows 本机 + WSL2（Ubuntu），在 Windows 的 PowerShell 里
> 先跑一次 `nvidia-smi` 确认能通，再到 WSL2 里跑——两边都能通，说明驱动共享正常。
> 如果你在 Windows 侧就跑不通，那问题不在 Linux，去 NVIDIA 官网下载对应显卡型号
> 的最新驱动安装。
""",
}

NEW_CHUNKS = [CH1_1_1, CH1_1_2]


def main():
    with engine.connect() as conn:
        # 1. 查 C001 的 doc_id，继承
        ref = conn.execute(text("""
            SELECT doc_id, block_id FROM knowledge_chunks WHERE chunk_id='LOCAL-KBS-01-04-C001'
        """)).first()
        if not ref:
            print("[ERR] LOCAL-KBS-01-04-C001 不存在，DB 里 seed 数据可能已不同。退出。")
            return 2
        doc_id, block_id = ref[0], ref[1]

        # 2. 检查是否已插入过（幂等）
        existing = conn.execute(text("""
            SELECT chunk_id FROM knowledge_chunks WHERE chunk_id IN (:a,:b)
        """), {"a": NEW_CHUNKS[0]["chunk_id"], "b": NEW_CHUNKS[1]["chunk_id"]}).fetchall()
        if existing:
            print(f"[SKIP] 开篇 chunks 已存在：{[e[0] for e in existing]}，无需重复插入。")
            return 0

        # 3. 构造 chunk_index：现有 nvidia-smi chunks 的 chunk_index 最小是 1（见诊断）
        #    插入序章用 index = -1 / 0 保持在最前（用小数 0.1, 0.2 避免和后续冲突）
        #    但 chunk_index 是 INT 列？先查表结构
        col_rows = conn.execute(text("PRAGMA table_info(knowledge_chunks)")).fetchall()
        dtype = {r[1]: r[2] for r in col_rows}
        print(f"[schema] chunk_index dtype = {dtype.get('chunk_index')}")
        # 查现有 chunk_index 最小值
        mn = conn.execute(text("""
            SELECT COALESCE(MIN(chunk_index), 0)
            FROM knowledge_chunks WHERE block_id=:bid AND heading_path LIKE '%nvidia-smi%显存监控%'
        """), {"bid": block_id}).scalar()
        print(f"[existing] KBS-01-04 min chunk_index = {mn}")
        base = mn if mn is not None else 1
        # 如果 base=1，序章 chunk_index=0 和 -1
        indices = [base - 2, base - 1]

        # 4. 插入
        for c, cidx in zip(NEW_CHUNKS, indices):
            conn.execute(text("""
                INSERT INTO knowledge_chunks
                    (chunk_id, doc_id, block_id, chunk_index, heading_path,
                     content, chunk_type, char_start, char_end,
                     heading_level, source_line_start, source_line_end, chunk_strategy)
                VALUES
                    (:chunk_id, :doc_id, :block_id, :chunk_index, :heading_path,
                     :content, :chunk_type, :char_start, :char_end,
                     :heading_level, :source_line_start, :source_line_end, :chunk_strategy)
            """), {
                "chunk_id": c["chunk_id"],
                "doc_id": doc_id,
                "block_id": block_id,
                "chunk_index": cidx,
                "heading_path": c["heading_path"],
                "content": c["content"],
                "chunk_type": "text",
                "char_start": None,
                "char_end": None,
                "heading_level": 3,
                "source_line_start": 0,
                "source_line_end": 0,
                "chunk_strategy": "补齐缺失序章_seed_patch",
            })
            print(f"[OK] 插入 {c['chunk_id']}  chunk_index={cidx}  "
                  f"leaf={c['heading_path'].split('>')[-1].strip()[:50]}")
        conn.commit()
        print("\n[DONE] KBS-01-04 缺失的第一章（序章两个 chunk）已补入 knowledge_chunks。")
        print("       ⚠️  注意：已生成的旧 session learning_package 还保持旧 content_blocks，")
        print("            需要删除旧的 Linux 学习目标，用一键启动重新生成，才能看到第一章。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
