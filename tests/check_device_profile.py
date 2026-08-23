# -*- coding: utf-8 -*-
"""
校验「用户画像设备配置」保存 + 设备型号自动补全（个性化学习资源生成 Agent）
================================================================
覆盖用户提出的两个问题：
  ① 没有保存用户画像中设备配置相关内容
  ② 无法通过设备型号自动填写用户画像的设备配置内容（如只报 "RTX 4090"）

断言分三段：
  D1 device_spec 查表/归一化：型号→显存/架构自动补全，显式显存优先，非 GPU 兜底。
  D2 PracticeProfile：from_user_input / from_hardware_text 产出结构化字段。
  D3 端到端持久化：StateManager.update_practice_profile 落库 learner_profiles，
     get/refresh 能读回结构化设备字段；只报型号也能补出显存（②）；
     历史行（仅有 hardware 文本）读回时自动补全（无需数据回填）。

运行：
  python tests/check_device_profile.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from sqlalchemy import text  # noqa: E402

from db.sqlite_client import DEFAULT_USER_ID, SQLiteClient  # noqa: E402
from models.device_spec import resolve_device  # noqa: E402
from models.profile import PracticeProfile  # noqa: E402
from orchestrator.state_manager import StateManager  # noqa: E402


class _Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, cond: bool, name: str, detail: str = "") -> None:
        if cond:
            self.passed += 1
            print(f"  ✅ PASS  {name}" + (f" | {detail}" if detail else ""))
        else:
            self.failed += 1
            print(f"  ❌ FAIL  {name}" + (f" | {detail}" if detail else ""))


def main() -> int:
    checker = _Checker()

    # ---------- D1 device_spec 查表/归一化 ----------
    print("\n[D1] device_spec 型号→显存/架构 查表与归一化")
    spec = resolve_device("我的显卡是 RTX 4090")
    checker.check(spec.vram_gb == 24, "D1-1 只报型号也能补出显存（RTX 4090→24GB）", f"vram={spec.vram_gb}")
    checker.check(spec.gpu_architecture == "Ada Lovelace", "D1-1 补出架构", spec.gpu_architecture)
    checker.check(spec.gpu_model == "RTX 4090", "D1-1 归一化型号", spec.gpu_model)
    checker.check(spec.device_type == "gpu", "D1-1 设备类型判定为 gpu", spec.device_type)

    spec2 = resolve_device("RTX 4090 with 16GB VRAM")
    checker.check(spec2.vram_gb == 16, "D1-2 用户显式显存优先于查表默认值（16 覆盖 24）", f"vram={spec2.vram_gb}")

    spec3 = resolve_device("云端 A100")
    checker.check(
        spec3.device_type == "cloud" and spec3.vram_gb == 80,
        "D1-3 云端 + 数据中心卡（A100→80GB, cloud）",
        f"type={spec3.device_type}, vram={spec3.vram_gb}",
    )

    spec4 = resolve_device("i7-12700 纯 CPU")
    checker.check(
        spec4.device_type == "cpu" and spec4.vram_gb is None,
        "D1-4 非 GPU 不臆造显存（CPU→vram=None）",
        f"type={spec4.device_type}, vram={spec4.vram_gb}",
    )

    spec5 = resolve_device("nvidia rtx 4070 ti super")
    checker.check(
        spec5.gpu_model == "RTX 4070 Ti Super" and spec5.vram_gb == 16,
        "D1-5 带后缀/厂商前缀型号归一化（RTX 4070 Ti Super→16GB）",
        f"model={spec5.gpu_model}, vram={spec5.vram_gb}",
    )

    # ---------- D2 PracticeProfile 结构化字段 ----------
    print("\n[D2] PracticeProfile 结构化设备字段")
    pp = PracticeProfile.from_user_input("我有一块 RTX 3080 用来跑模型")
    checker.check(pp.gpu_model == "RTX 3080", "D2-1 from_user_input 识别无标签型号", pp.gpu_model)
    checker.check(pp.vram_gb == 10, "D2-1 自动补显存（RTX 3080→10GB）", f"vram={pp.vram_gb}")

    pp2 = PracticeProfile.from_hardware_text("RTX 4060")
    checker.check(
        pp2.hardware == "RTX 4060" and pp2.vram_gb == 8,
        "D2-2 from_hardware_text 保留原文并补显存",
        f"hardware={pp2.hardware}, vram={pp2.vram_gb}",
    )

    pp3 = PracticeProfile.from_user_input("设备条件是 RTX 4090 24GB")
    checker.check(pp3.vram_gb == 24, "D2-3 带标签+显式显存解析", f"vram={pp3.vram_gb}")

    # ---------- D3 端到端持久化 ----------
    print("\n[D3] StateManager 落库/读回 结构化设备配置")
    with tempfile.TemporaryDirectory() as tmp:
        database = Path(tmp) / "device_profile.db"
        client = SQLiteClient(f"sqlite:///{database.as_posix()}")
        try:
            sm = StateManager(client)
            sm.create_session("sess-device", user_id=DEFAULT_USER_ID)
            sm.update_profile("sess-device", {
                "session_id": "sess-device",
                "user_id": DEFAULT_USER_ID,
                "goal": "微调大模型",
                "daily_hours": 2.0,
                "learning_style": "practice",
            })

            # ② 只报型号 → update_practice_profile 应自动补全并落库
            sm.update_practice_profile("sess-device", {"hardware": "RTX 4090"})

            # ① 直接查库确认设备配置已保存（不是只留在内存）
            row = client.get_user_profile(DEFAULT_USER_ID) or {}
            checker.check(row.get("hardware") == "RTX 4090", "D3-1 hardware 文本已落库（问题①）", row.get("hardware"))
            checker.check(row.get("vram_gb") == 24, "D3-2 显存已自动补全并落库（问题②）", f"vram={row.get('vram_gb')}")
            checker.check(row.get("gpu_model") == "RTX 4090", "D3-2 型号已落库", row.get("gpu_model"))
            checker.check(row.get("gpu_architecture") == "Ada Lovelace", "D3-2 架构已落库", row.get("gpu_architecture"))
            checker.check(row.get("device_type") == "gpu", "D3-2 设备类型已落库", row.get("device_type"))

            # get_practice_profile 读回结构化字段
            got = sm.get_practice_profile("sess-device")
            checker.check(
                got.get("vram_gb") == 24 and got.get("gpu_model") == "RTX 4090",
                "D3-3 get_practice_profile 读回结构化设备字段",
                f"vram={got.get('vram_gb')}, model={got.get('gpu_model')}",
            )

            # 显式显存优先：PATCH 类更新写入 "RTX 4090 16GB" 应覆盖为 16
            sm.update_practice_profile("sess-device", {"hardware": "RTX 4090 16GB"})
            row2 = client.get_user_profile(DEFAULT_USER_ID) or {}
            checker.check(row2.get("vram_gb") == 16, "D3-4 显式显存覆盖查表默认值并落库", f"vram={row2.get('vram_gb')}")

            # 历史行兼容：手动只写 hardware 文本（模拟旧数据），读回自动补全
            client.upsert_user_profile(DEFAULT_USER_ID, {
                "hardware": "RTX 3070",
                "device_type": "",
                "gpu_model": "",
                "vram_gb": None,
                "gpu_architecture": "",
            })
            legacy = sm.get_practice_profile("sess-device")
            checker.check(
                legacy.get("vram_gb") == 8 and legacy.get("gpu_model") == "RTX 3070",
                "D3-5 历史行仅有文本时读回自动补全（无需数据回填）",
                f"vram={legacy.get('vram_gb')}, model={legacy.get('gpu_model')}",
            )
        finally:
            client.close()

    print("\n" + "#" * 60)
    print(f"# 设备画像断言汇总：{checker.passed} 通过 / {checker.failed} 失败")
    print("#" * 60)
    return 1 if checker.failed else 0


if __name__ == "__main__":
    sys.exit(main())
