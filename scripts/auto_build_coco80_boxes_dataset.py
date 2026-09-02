#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_LABEL = 255
LABEL_FOLDER_NAME = "labels"

COCO80_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_dir: Path
    image_dir: Path


def log(message: str) -> None:
    print(message, flush=True)


def discover_samples(dataset_root: Path, image_folder: str = "color") -> list[Sample]:
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(source: Path, target: Path, link_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or link_mode == "none":
        return
    if link_mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def mirror_sample_assets(source_dir: Path, target_dir: Path, label_folder: str, link_mode: str) -> None:
    ensure_dir(target_dir)
    for item in sorted(source_dir.iterdir()):
        if item.name == label_folder:
            continue
        if item.is_dir():
            for file_path in sorted(item.rglob("*")):
                if not file_path.is_file():
                    continue
                relative = file_path.relative_to(source_dir)
                link_or_copy_file(file_path, target_dir / relative, link_mode)
        elif item.is_file():
            link_or_copy_file(item, target_dir / item.name, link_mode)


def write_label_file(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_class_file(output_root: Path, classes: list[str], ignore_label: int = IGNORE_LABEL) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    lines = [f"{idx}: {name}" for idx, name in enumerate(classes)]
    lines.append(f"{ignore_label}: ignore")
    (output_root / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_meta_file(output_root: Path, payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a COCO-80 object-detection dataset from color images with progress output."
    )
    parser.add_argument("input_root", type=Path, help="Source dataset root, e.g. /data/lmz/train_data")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output dataset root. Default: RGBDT500_80_det",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local Ultralytics checkpoint path, e.g. /data/lmz/code8.31/yolov8x-seg.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument(
        "--label-folder",
        default=LABEL_FOLDER_NAME,
        help=f"Label folder name inside each sample. Default: {LABEL_FOLDER_NAME}",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "none"],
        default="hardlink",
        help="How to materialize original files into the new dataset root.",
    )
    parser.add_argument("--device", default=None, help="Inference device such as cpu, cuda:0, or empty for auto.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip label generation when the output txt already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root.with_name("RGBDT500_80_det")).resolve()

    if not args.model:
        raise SystemExit("Please pass a local checkpoint with --model.")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model checkpoint not found locally: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model_names = [model.names[idx] for idx in sorted(model.names)]
    if len(model_names) != 80:
        raise SystemExit(f"Expected a COCO-80 model, but got {len(model_names)} classes.")

    samples = discover_samples(input_root)
    if not samples:
        raise SystemExit(f"No sample folders with `color/` were found in {input_root}")
    sample_images = [(sample, collect_images(sample.image_dir)) for sample in samples]
    total_images = sum(len(images) for _, images in sample_images)

    log(f"[DET80] output={output_root}")
    log(f"[DET80] samples={len(sample_images)} images={total_images}")

    write_class_file(output_root, COCO80_CLASSES)
    write_meta_file(
        output_root,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(input_root),
            "generator": "auto_build_coco80_boxes_dataset.py",
            "model": str(model_path),
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "classes": COCO80_CLASSES,
        },
    )

    done_images = 0
    for sample_idx, (sample, image_paths) in enumerate(sample_images, start=1):
        log(f"[DET80] sample {sample_idx}/{len(sample_images)} {sample.sample_id} ({len(image_paths)} images)")
        target_sample_dir = output_root / sample.sample_id
        mirror_sample_assets(sample.source_dir, target_sample_dir, args.label_folder, args.link_mode)

        for image_idx, image_path in enumerate(image_paths, start=1):
            output_txt = target_sample_dir / args.label_folder / f"{image_path.stem}.txt"
            if args.skip_existing and output_txt.exists():
                done_images += 1
                log(f"[DET80]   ({done_images}/{total_images}) skip {sample.sample_id}/{image_idx}/{len(image_paths)} {image_path.name}")
                continue

            with Image.open(image_path) as image:
                width, height = image.size

            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
                device=args.device,
            )
            result = results[0]
            lines: list[str] = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy().astype(float)
                for (x1, y1, x2, y2), class_id, score in zip(boxes, classes, scores):
                    cx, cy, bw, bh = xyxy_to_yolo(float(x1), float(y1), float(x2), float(y2), width, height)
                    cx = min(max(cx, 0.0), 1.0)
                    cy = min(max(cy, 0.0), 1.0)
                    bw = min(max(bw, 0.0), 1.0)
                    bh = min(max(bh, 0.0), 1.0)
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            write_label_file(lines, output_txt)
            done_images += 1
            log(f"[DET80]   ({done_images}/{total_images}) done {sample.sample_id}/{image_idx}/{len(image_paths)} {image_path.name} boxes={len(lines)}")

    print(f"Finished COCO-80 detection dataset build: {output_root}")
    print(f"Samples: {len(sample_images)}")
    print(f"Images processed: {done_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
