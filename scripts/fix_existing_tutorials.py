# -*- coding: utf-8 -*-
"""
修复 DB 中现存 session 的 tutorial 排版：
- 用 harness._aggregate_blocks 的新逻辑，基于已有的 content_blocks 重算 tutorial
- 只改 tutorial 字段，content_blocks / sources / claims 等不动
- 保留旧数据备份到 tmp/backup_session_states_{timestamp}.json
"""
import json
import sys
import os
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text  # noqa: E402


def recompute_tutorial(state: dict) -> str:
    """按新聚合逻辑，用 state['learning_package']['content_blocks'] 重算 tutorial。
    逻辑与 harness._aggregate_blocks 对齐（Python 侧独立实现，避免 FastAPI 依赖链）。
    """
    import re as _re

    def _clean_internal_hr(text: str) -> str:
        if not text:
            return text
        lines = text.splitlines()
        result: list[str] = []
        _HR_RE = _re.compile(r"^\s*([-*_])\s*(?:\1\s*){2,}$")
        for ln in lines:
            stripped = ln.strip()
            is_hr = bool(_HR_RE.match(stripped)) and len(stripped) >= 3 and stripped[0] in "-*_"
            if is_hr:
                if result and result[-1].strip() != "":
                    result.append("")
                continue
            result.append(ln)
        out: list[str] = []
        blank_count = 0
        for ln in result:
            if ln.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    out.append(ln)
            else:
                blank_count = 0
                out.append(ln)
        return "\n".join(out)

    lp = state.get("learning_package") or {}
    stage_name = lp.get("block_name") or lp.get("stage_name") or "学习内容"
    blocks = lp.get("content_blocks") or []

    # block_index 排序
    def _idx(b):
        if isinstance(b, dict):
            return b.get("block_index", 0) or 0
        return getattr(b, "block_index", 0) or 0

    ordered = sorted(blocks, key=_idx)

    parts: list[str] = [f"# {stage_name}"]
    evidence = lp.get("learning_evidence") or {}
    preflight = evidence.get("preflight_review") if isinstance(evidence, dict) else None
    if isinstance(preflight, dict) and preflight.get("markdown"):
        parts.append(str(preflight["markdown"]))

    for idx, blk in enumerate(ordered):
        if idx > 0:
            parts.append("\n---\n")
        heading = ""
        pc_raw = ""
        if isinstance(blk, dict):
            heading = blk.get("heading_path") or blk.get("block_name") or ""
            pc_raw = blk.get("personalized_content") or ""
        else:
            heading = getattr(blk, "heading_path", None) or getattr(blk, "block_name", None) or ""
            pc_raw = getattr(blk, "personalized_content", None) or ""
        pc = _clean_internal_hr(pc_raw)
        # 同步写回 content_blocks 里的 personalized_content（与 tutorial 一致）
        if pc != pc_raw and isinstance(blk, dict):
            blk["personalized_content"] = pc

        has_internal_heading = bool(pc) and any(
            ln.lstrip().startswith(prefix)
            for prefix in ("## ", "### ", "#### ")
            for ln in pc.splitlines()
        )
        if heading and not has_internal_heading:
            leaf = heading
            if ">" in heading:
                leaf = heading.split(">")[-1].strip()
            parts.append(f"## {leaf}")
        if pc:
            parts.append(pc)

    return "\n\n".join(p for p in parts if p is not None and p != "")


def main():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    backups: dict = {}
    updated_count = 0
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT session_id, state_json FROM session_states")).fetchall()
        for sid, sj in rows:
            try:
                state = json.loads(sj)
            except Exception:
                continue
            lp = state.get("learning_package")
            if not lp or not lp.get("content_blocks"):
                continue
            old_tutorial = lp.get("tutorial") or ""
            new_tutorial = recompute_tutorial(state)
            if old_tutorial == new_tutorial:
                print(f"[{sid[:12]}] tutorial 已是新版，跳过")
                continue
            backups[sid] = {"old_tutorial": old_tutorial, "old_state_json": sj}
            lp["tutorial"] = new_tutorial
            state["learning_package"] = lp
            new_sj = json.dumps(state, ensure_ascii=False)
            conn.execute(text("""
                UPDATE session_states SET state_json = :sj, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = :sid
            """), {"sj": new_sj, "sid": sid})
            updated_count += 1
            # 打印前后对比（前25行）
            print(f"\n[{sid[:12]}] tutorial 已更新（len: {len(old_tutorial)} -> {len(new_tutorial)}）")
            print("  ---- 新版 tutorial 前 25 行 ----")
            for i, ln in enumerate(new_tutorial.splitlines()[:25]):
                print(f"    L{i+1:2d}: {ln[:120]}")
        conn.commit()

    # 备份
    if backups:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bk_path = PROJECT_ROOT / "tmp" / f"backup_session_states_{ts}.json"
        bk_path.parent.mkdir(exist_ok=True)
        with open(bk_path, "w", encoding="utf-8") as f:
            json.dump(backups, f, ensure_ascii=False, indent=2)
        print(f"\n备份文件：{bk_path}")

    print(f"\n共更新 {updated_count} 个 session 的 tutorial。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
