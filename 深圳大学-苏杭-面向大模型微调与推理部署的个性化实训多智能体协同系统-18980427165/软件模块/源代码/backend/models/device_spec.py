# -*- coding: utf-8 -*-
"""
device_spec.py —— 设备型号 → 结构化规格（显存/架构）查表与归一化
================================================================
用户画像里的"设备配置"过去只存一条自由文本（如 "RTX 4090 24GB"），
既没有结构化字段，也无法在用户只报型号（如 "RTX 4090"）时自动补出显存。

本模块提供：
  - 一张 NVIDIA/AMD 常见消费级/工作站显卡的型号→(显存 GB, 架构) 查表；
  - ``resolve_device(text)``：从自由文本里识别 GPU 型号、显存、架构、设备类型，
    在缺失显存时用查表自动补全，返回结构化 ``DeviceSpec``。

设计原则：
  - 只补全"能可靠查到"的信息；查不到就保持空，绝不臆造显存数字。
  - 用户自己写明的显存（"RTX 4090 24GB" / "24G 显存"）优先于查表值，
    因为同一型号可能有非公版/移动版差异。
"""
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

# 型号关键字（小写、去空格后匹配）→ (显存 GB, 架构)。
# 键使用"紧凑型号"形式：去掉空格，如 "rtx4090"。移动版/Ti 单列以便精确匹配。
_GPU_SPEC_TABLE: Dict[str, Tuple[int, str]] = {
    # --- NVIDIA RTX 40 系（Ada Lovelace）---
    "rtx4090": (24, "Ada Lovelace"),
    "rtx4080super": (16, "Ada Lovelace"),
    "rtx4080": (16, "Ada Lovelace"),
    "rtx4070tisuper": (16, "Ada Lovelace"),
    "rtx4070ti": (12, "Ada Lovelace"),
    "rtx4070super": (12, "Ada Lovelace"),
    "rtx4070": (12, "Ada Lovelace"),
    "rtx4060ti": (8, "Ada Lovelace"),
    "rtx4060": (8, "Ada Lovelace"),
    # --- NVIDIA RTX 30 系（Ampere）---
    "rtx3090ti": (24, "Ampere"),
    "rtx3090": (24, "Ampere"),
    "rtx3080ti": (12, "Ampere"),
    "rtx3080": (10, "Ampere"),
    "rtx3070ti": (8, "Ampere"),
    "rtx3070": (8, "Ampere"),
    "rtx3060ti": (8, "Ampere"),
    "rtx3060": (12, "Ampere"),
    "rtx3050": (8, "Ampere"),
    # --- NVIDIA RTX 20 系（Turing）---
    "rtx2080ti": (11, "Turing"),
    "rtx2080super": (8, "Turing"),
    "rtx2080": (8, "Turing"),
    "rtx2070super": (8, "Turing"),
    "rtx2070": (8, "Turing"),
    "rtx2060super": (8, "Turing"),
    "rtx2060": (6, "Turing"),
    # --- NVIDIA GTX 16/10 系（Turing/Pascal）---
    "gtx1660ti": (6, "Turing"),
    "gtx1660super": (6, "Turing"),
    "gtx1660": (6, "Turing"),
    "gtx1650": (4, "Turing"),
    "gtx1080ti": (11, "Pascal"),
    "gtx1080": (8, "Pascal"),
    "gtx1070ti": (8, "Pascal"),
    "gtx1070": (8, "Pascal"),
    "gtx1060": (6, "Pascal"),
    # --- NVIDIA 数据中心/工作站 ---
    "a100": (80, "Ampere"),
    "h100": (80, "Hopper"),
    "v100": (32, "Volta"),
    "a6000": (48, "Ampere"),
    "a5000": (24, "Ampere"),
    "a4000": (16, "Ampere"),
    "l40s": (48, "Ada Lovelace"),
    "l40": (48, "Ada Lovelace"),
    "l4": (24, "Ada Lovelace"),
    "t4": (16, "Turing"),
    # --- AMD Radeon RX 7000/6000 系 ---
    "rx7900xtx": (24, "RDNA 3"),
    "rx7900xt": (20, "RDNA 3"),
    "rx7800xt": (16, "RDNA 3"),
    "rx7700xt": (12, "RDNA 3"),
    "rx6950xt": (16, "RDNA 2"),
    "rx6900xt": (16, "RDNA 2"),
    "rx6800xt": (16, "RDNA 2"),
    "rx6800": (16, "RDNA 2"),
    "rx6700xt": (12, "RDNA 2"),
    "rx6600xt": (8, "RDNA 2"),
    "rx6600": (8, "RDNA 2"),
}

# 归一化别名：把常见写法收敛成查表键片段（在紧凑化之前替换）。
_ALIAS_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"geforce", ""),
    (r"nvidia", ""),
    (r"radeon", ""),
    (r"laptop|mobile|移动版|笔记本版", ""),
)

# 识别显式型号的正则（保留原文用于展示）。
_MODEL_PATTERN = re.compile(
    r"(?:nvidia\s+|geforce\s+|amd\s+|radeon\s+)?"
    r"(?:rtx|gtx|rx)\s*\d{3,4}\s*(?:ti\s*super|super|ti|xtx|xt)?"
    r"|(?:a100|h100|v100|a6000|a5000|a4000|l40s|l40|l4|t4)",
    re.IGNORECASE,
)

# 识别用户显式写出的显存（"24GB" / "24 G" / "24G 显存" / "显存 24GB"）。
_VRAM_PATTERN = re.compile(
    r"(\d{1,3})\s*g\s*b?(?:\s*(?:vram|显存|内存|memory))?"
    r"|(?:显存|vram)\s*[:：]?\s*(\d{1,3})\s*g",
    re.IGNORECASE,
)

_CPU_MARKERS = ("cpu", "核", "i3", "i5", "i7", "i9", "ryzen", "xeon", "arm", "m1", "m2", "m3")
_CLOUD_MARKERS = ("cloud", "云", "colab", "kaggle", "autodl", "aws", "aliyun", "阿里云", "腾讯云")


@dataclass
class DeviceSpec:
    """设备配置的结构化表示。所有字段可空——查不到就留空，不臆造。"""
    raw: str = ""                       # 原始/归一化后的展示文本
    device_type: str = ""               # gpu / cpu / cloud / ""(未知)
    gpu_model: str = ""                 # 归一化型号，如 "RTX 4090"
    vram_gb: Optional[int] = None       # 显存 GB
    gpu_architecture: str = ""          # 架构，如 "Ada Lovelace"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def summary(self) -> str:
        """人类可读摘要，供提示/展示使用。"""
        if not self.raw and not self.gpu_model:
            return ""
        parts = []
        if self.gpu_model:
            parts.append(self.gpu_model)
        elif self.raw:
            parts.append(self.raw)
        if self.vram_gb:
            parts.append(f"{self.vram_gb}GB 显存")
        if self.gpu_architecture:
            parts.append(f"{self.gpu_architecture} 架构")
        return "，".join(parts)


def _compact(model_text: str) -> str:
    """把型号文本收敛成查表键：小写、去别名、去空格。"""
    s = model_text.lower()
    for pattern, repl in _ALIAS_PATTERNS:
        s = re.sub(pattern, repl, s)
    return re.sub(r"\s+", "", s)


def _pretty_model(model_text: str) -> str:
    """把匹配到的原始型号整理成规范展示形式，如 "RTX 4090"。"""
    s = model_text.strip()
    for pattern, _ in _ALIAS_PATTERNS[:3]:  # 去掉厂商前缀，保留后缀（ti/super/xt）
        s = re.sub(pattern, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    # 统一大小写：型号前缀大写，后缀小写化为 Ti/Super/XT/XTX
    m = re.match(
        r"(rtx|gtx|rx)\s*(\d{3,4})\s*(ti\s*super|super|ti|xtx|xt)?",
        s,
        re.IGNORECASE,
    )
    if m:
        prefix = m.group(1).upper()
        number = m.group(2)
        suffix = (m.group(3) or "").replace(" ", "").lower()
        suffix_label = {
            "tisuper": " Ti Super",
            "super": " Super",
            "ti": " Ti",
            "xtx": " XTX",
            "xt": " XT",
        }.get(suffix, "")
        return f"{prefix} {number}{suffix_label}"
    # 数据中心卡（a100/h100...）直接大写
    if re.fullmatch(r"[a-z]\d{1,3}[a-z]*", s, re.IGNORECASE):
        return s.upper()
    return s


def lookup_gpu(model_text: str) -> Optional[Tuple[int, str]]:
    """按型号查 (显存 GB, 架构)；查不到返回 None。"""
    return _GPU_SPEC_TABLE.get(_compact(model_text))


def _extract_vram(text: str) -> Optional[int]:
    """从文本里抠出用户显式写的显存 GB；识别不到返回 None。"""
    for match in _VRAM_PATTERN.finditer(text):
        value = match.group(1) or match.group(2)
        if not value:
            continue
        vram = int(value)
        # 过滤明显不是显存的数字（型号里的 4090 已被 \b 边界避免，
        # 但仍防御性约束到合理显存范围）。
        if 1 <= vram <= 200:
            return vram
    return None


def _detect_device_type(text: str, has_gpu: bool) -> str:
    normalized = text.casefold()
    if any(marker in normalized for marker in _CLOUD_MARKERS):
        return "cloud"
    if has_gpu:
        return "gpu"
    if any(marker in normalized for marker in _CPU_MARKERS):
        return "cpu"
    return ""


def resolve_device(text: str) -> DeviceSpec:
    """从自由文本解析结构化设备配置，缺失显存时按型号查表自动补全。

    示例：
      resolve_device("我的显卡是 RTX 4090")      -> vram_gb=24, architecture="Ada Lovelace"
      resolve_device("RTX 4090 with 16GB")       -> vram_gb=16（用户显式值优先于查表 24）
      resolve_device("云端 A100")                 -> device_type="cloud", vram_gb=80
    """
    text = (text or "").strip()
    if not text:
        return DeviceSpec()

    model_match = _MODEL_PATTERN.search(text)
    gpu_model = ""
    vram_gb: Optional[int] = None
    architecture = ""

    if model_match:
        gpu_model = _pretty_model(model_match.group(0))
        table_hit = lookup_gpu(model_match.group(0))
        if table_hit:
            vram_gb, architecture = table_hit

    # 用户显式写出的显存优先（覆盖查表默认值，兼容非公版/移动版）。
    explicit_vram = _extract_vram(text)
    if explicit_vram is not None:
        vram_gb = explicit_vram

    device_type = _detect_device_type(text, has_gpu=bool(gpu_model))

    return DeviceSpec(
        raw=text,
        device_type=device_type,
        gpu_model=gpu_model,
        vram_gb=vram_gb,
        gpu_architecture=architecture,
    )
