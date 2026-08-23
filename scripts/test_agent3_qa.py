# -*- coding: utf-8 -*-
"""
个性化学习资源生成 Agent 问答检索效果测试脚本（交互式）

流程：
  1. 调用 学情诊断与阶段规划 Agent 生成一个学习阶段
  2. 打印阶段的板块(block_id)和知识点(subtopic_id/name)
  3. 用户自己在终端输入问题，脚本实时输出命中的板块和知识点
  4. 输入 q 退出

运行方式：
  cd <项目根目录>
  python scripts\test_agent3_qa.py
"""
import os
import sys
import asyncio
import sqlite3

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BACKEND_DIR)

from backend.orchestrator.harness import Harness
from backend.tools.agent3_tools.vector_search_tool import vector_search, _get_db_path


def lookup_subtopic_info(chunk_ids: list[str]) -> dict:
    """用 chunk_id 反查 block_name + subtopic_name。"""
    if not chunk_ids:
        return {}
    conn = sqlite3.connect(f"file:{_get_db_path()}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(f"""
            SELECT kc.chunk_id,
                   kc.block_id,
                   kb.block_name,
                   kd.subtopic_id,
                   ks.subtopic_name
            FROM knowledge_chunks kc
            LEFT JOIN knowledge_blocks kb ON kc.block_id = kb.block_id
            LEFT JOIN knowledge_docs   kd ON kc.doc_id   = kd.doc_id
            LEFT JOIN knowledge_subtopics ks ON kd.subtopic_id = ks.subtopic_id
            WHERE kc.chunk_id IN ({placeholders})
        """, chunk_ids).fetchall()
        return {
            r[0]: {"block_id": r[1], "block_name": r[2],
                   "subtopic_id": r[3], "subtopic_name": r[4]}
            for r in rows
        }
    finally:
        conn.close()


def do_search(question: str, block_ids: list[str], subtopic_ids: list[str],
              top_k: int = 5):
    """执行一次检索并打印结果"""
    print(f"\n{'─'*60}")
    print(f"  问题: {question}")
    print(f"  过滤: block_ids={block_ids}, subtopic_ids={subtopic_ids}")
    print(f"{'─'*60}")

    result = vector_search({
        "query": question,
        "block_ids": block_ids,
        "subtopic_ids": subtopic_ids,
        "top_k": top_k,
    }, config={"use_mock": False})

    if result["status"] != "success":
        print(f"  ❌ 检索失败: {result.get('message', '')}")
        return

    results = result["data"]["results"]
    db_available = result["data"]["db_available"]
    source = "Chroma" if db_available else "SQLite FTS5"

    if not results:
        print(f"  ⚠️ 无检索结果（来源: {source}）")
        return

    chunk_ids = [r["chunk_id"] for r in results]
    info_map = lookup_subtopic_info(chunk_ids)

    print(f"  检索来源: {source}，命中 {len(results)} 条：\n")
    for i, r in enumerate(results, 1):
        info = info_map.get(r["chunk_id"], {})
        block_name = info.get("block_name") or r.get("title", "")[:30]
        subtopic_name = info.get("subtopic_name") or "(未知知识点)"
        subtopic_id = info.get("subtopic_id") or "(无)"
        print(f"  [{i}] similarity={r['similarity']:.4f}")
        print(f"      板块: {info.get('block_id', '?')} - {block_name}")
        print(f"      知识点: {subtopic_id} - {subtopic_name}")
        print(f"      chunk_id: {r['chunk_id']}")
        print(f"      内容摘要: {r['content'][:100].replace(chr(10), ' ')}...")
        print()


async def run_test():
    print("=" * 60)
    print("  个性化学习资源生成 Agent 问答检索效果测试（交互式）")
    print("  Embedding: qwen3-embedding:0.6b")
    print("=" * 60)

    harness = Harness()

    # === Step 1: 创建 session ===
    session_id = harness.state_manager.create_session(None)["session_id"]
    print(f"\n✅ Session 创建成功: {session_id}")

    # === Step 2: 调用 学情诊断与阶段规划 Agent 生成阶段 ===
    user_input = """
    我的学习背景：研究生，有PyTorch基础，用过transformer库
    我的学习目标：掌握LoRA微调，能独立部署到生产环境
    我每天能学：2小时
    我喜欢怎么学：偏爱动手实操，不太喜欢纯理论讲解
    """
    print("\n⏳ 正在调用 学情诊断与阶段规划 Agent 生成学习阶段（需要 LLM，请耐心等待）...")
    result = await harness.process_initial_input(session_id, user_input)

    if result["status"] != "ready":
        print(f"⚠️ 阶段生成未就绪: {result['status']}")
        if "questions" in result:
            print(f"  需要补充信息: {result['questions']}")
        return

    stage = result["stage_info"]

    # === Step 3: 打印阶段信息 ===
    print("\n" + "=" * 60)
    print("  📋 阶段信息")
    print("=" * 60)
    print(f"  阶段编号: {stage['stage_num']}")
    print(f"  主板块: {stage['block_id']} - {stage['block_name']}")
    print(f"  多板块: {stage.get('block_ids', [])}")
    print(f"  知识点ID: {stage.get('subtopic_ids', [])}")
    print(f"  知识点名称:")
    subtopic_names = stage.get("subtopic_names") or (
        [stage["subtopic_name"]] if stage.get("subtopic_name") else []
    )
    for i, name in enumerate(subtopic_names, 1):
        print(f"    {i}. {name}")
    if not subtopic_names:
        print(f"    (fallback) {stage.get('subtopic_name', '(无)')}")
    print(f"  预计学时: {stage['estimated_hours']}h")
    print(f"  难度: {stage.get('difficulty', '?')}")

    # 收集过滤条件
    block_ids = stage.get("block_ids") or ([stage["block_id"]] if stage.get("block_id") else [])
    subtopic_ids = stage.get("subtopic_ids") or (
        [stage["subtopic_id"]] if stage.get("subtopic_id") else []
    )

    # === Step 4: 交互式问答 ===
    print("\n\n" + "=" * 60)
    print("  💬 交互式问答（输入问题，看命中的板块和知识点）")
    print("=" * 60)
    print("  操作说明：")
    print("    - 输入问题后回车，查看本阶段范围内的检索结果")
    print("    - 输入 'all' 切换为全库检索（不加板块/知识点过滤）")
    print("    - 输入 'stage' 切换回本阶段检索")
    print("    - 输入 'q' 退出")
    print()

    search_mode = "stage"  # 'stage' 或 'all'

    while True:
        try:
            print(f"\n[当前模式: {'本阶段检索' if search_mode == 'stage' else '全库检索'}]")
            question = input("📝 你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() == "q":
            print("再见！")
            break
        if question.lower() == "all":
            search_mode = "all"
            print("  ✅ 已切换为全库检索")
            continue
        if question.lower() == "stage":
            search_mode = "stage"
            print("  ✅ 已切换为本阶段检索")
            continue

        # 执行检索
        if search_mode == "stage":
            do_search(question, block_ids, subtopic_ids)
        else:
            do_search(question, [], [])


if __name__ == "__main__":
    asyncio.run(run_test())
