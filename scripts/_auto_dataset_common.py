#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_FOLDER_NAME = "color"
MASK_FOLDER_NAME = "mask"
IGNORE_LABEL = 255


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_dir: Path
    image_dir: Path


def discover_samples(dataset_root: Path, image_folder: str = IMAGE_FOLDER_NAME) -> list[Sample]:
    samples: list[Sample] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        image_dir = child / image_folder
        if image_dir.is_dir():
            samples.append(Sample(sample_id=child.name, source_dir=child, image_dir=image_dir))
    return samples


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(source: Path, target: Path, link_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return

    if link_mode == "none":
        return

    if link_mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass

    if link_mode in {"hardlink", "copy"}:
        shutil.copy2(source, target)
        return

    raise ValueError(f"Unsupported link mode: {link_mode}")


def mirror_sample_assets(source_dir: Path, target_dir: Path, mask_folder: str, link_mode: str) -> None:
    ensure_clean_dir(target_dir)
    for item in sorted(source_dir.iterdir()):
        if item.name == mask_folder:
            continue
        if item.is_dir():
            for file_path in sorted(item.rglob("*")):
                if not file_path.is_file():
                    continue
                relative = file_path.relative_to(source_dir)
                link_or_copy_file(file_path, target_dir / relative, link_mode)
        elif item.is_file():
            link_or_copy_file(item, target_dir / item.name, link_mode)


def write_mask(mask, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)


def write_class_file(output_root: Path, classes: Iterable[str], ignore_label: int = IGNORE_LABEL) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    class_file = output_root / "labels.txt"
    lines = [f"{idx}: {name}" for idx, name in enumerate(classes)]
    lines.append(f"{ignore_label}: ignore")
    class_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_meta_file(output_root: Path, payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
