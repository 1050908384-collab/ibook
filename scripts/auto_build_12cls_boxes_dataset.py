#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

CLASS_NAMES_12 = [
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage_can",
    "uav",
    "tricycle",
]

# COCO class id -> 12-class id
COCO_TO_12 = {
    0: 0,    # person
    8: 1,    # boat
    14: 2,   # bird -> animal
    15: 2,   # cat -> animal
    16: 2,   # dog -> animal
    17: 2,   # horse -> animal
    18: 2,   # sheep -> animal
    19: 2,   # cow -> animal
    20: 2,   # elephant -> animal
    21: 2,   # bear -> animal
    22: 2,   # zebra -> animal
    23: 2,   # giraffe -> animal
    13: 3,   # bench -> seat
    56: 3,   # chair -> seat
    57: 3,   # couch -> seat
    11: 4,   # stop sign -> sign
    1: 5,    # bicycle
    3: 5,    # motorcycle -> bicycle
    2: 6,    # car
    5: 6,    # bus -> car
    7: 6,    # truck -> car
    32: 7,   # sports ball
    9: 8,    # traffic light -> light
}

COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_dir: Path
    image_dir: Path


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    script_dir = Path(__file__).resolve().parent
    cand = script_dir / path
    if cand.exists():
        return cand.resolve()
    return (Path.cwd() / path).resolve()


def discover_samples(dataset_root: Path, image_folder: str = "color") -> list[Sample]:
    samples: list[Sample] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        image_dir = child / image_folder
        if image_dir.is_dir():
            samples.append(Sample(child.name, child, image_dir))
    return samples


def collect_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def class_name(cls_id: int) -> str:
    return CLASS_NAMES_12[cls_id] if 0 <= cls_id < len(CLASS_NAMES_12) else str(cls_id)


def class_color(cls_id: int) -> tuple[int, int, int]:
    h = (cls_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    y = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (0, 0, 0) if y > 160 else (255, 255, 255)


def load_font(size: int = 18):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size=size)
    return ImageFont.load_default()


def draw_boxes(image: Image.Image, result, line_width: int = 3):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = load_font(max(14, line_width * 5))

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return img, 0

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy().astype(float)
    count = 0

    for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
        if cls_id not in COCO_TO_12:
            continue
        mapped = COCO_TO_12[cls_id]
        color = class_color(mapped)
        label = f"{class_name(mapped)} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 3

        tx = max(0, int(x1))
        ty = max(0, int(y1) - th - pad * 2)
        if ty < 0:
            ty = int(y1) + 1

        draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad * 2], fill=color)
        draw.text((tx + pad, ty + pad), label, fill=text_color(color), font=font)
        count += 1

    return img, count


def write_label_file(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and visualize 12-class box labels with progress output.")
    parser.add_argument("input_root", type=Path, help="e.g. /data/lmz/train_data")
    parser.add_argument("--model", required=True, help="local model path, e.g. /data/lmz/code8.31/yolov8x-seg.pt")
    parser.add_argument("--output-root", type=Path, default=None, help="default: RGBDT500_12_box")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--save-labels", action="store_true", help="also save YOLO txt labels")
    parser.add_argument("--label-folder", default="labels", help="folder name for txt labels")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root.with_name("RGBDT500_12_box")).resolve()
    model_path = resolve_path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    samples = discover_samples(input_root)
    if not samples:
        raise SystemExit(f"No sample folders with `color/` found in {input_root}")
    sample_images = [(s, collect_images(s.image_dir)) for s in samples]
    total = sum(len(imgs) for _, imgs in sample_images)

    log(f"[VIS12] output={output_root}")
    log(f"[VIS12] samples={len(sample_images)} images={total}")
    log(f"[VIS12] model={model_path}")

    done = 0
    for si, (sample, image_paths) in enumerate(sample_images, start=1):
        log(f"[VIS12] sample {si}/{len(sample_images)} {sample.sample_id} ({len(image_paths)} images)")
        out_dir = output_root / sample.sample_id
        ensure_dir(out_dir)
        if args.save_labels:
            ensure_dir(out_dir / args.label_folder)

        for ii, image_path in enumerate(image_paths, start=1):
            out_img = out_dir / image_path.name
            out_txt = out_dir / args.label_folder / f"{image_path.stem}.txt"
            if args.skip_existing and out_img.exists():
                done += 1
                log(f"[VIS12]   ({done}/{total}) skip {sample.sample_id}/{ii}/{len(image_paths)} {image_path.name}")
                continue

            with Image.open(image_path) as im:
                result = model.predict(
                    source=str(image_path),
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    verbose=False,
                    device=args.device,
                )[0]
                annotated, n_boxes = draw_boxes(im, result, line_width=args.line_width)
                annotated.save(out_img)

                if args.save_labels:
                    boxes = result.boxes
                    lines: list[str] = []
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        confs = boxes.conf.cpu().numpy().astype(float)
                        w, h = im.size
                        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                            if cls_id not in COCO_TO_12:
                                continue
                            mapped = COCO_TO_12[cls_id]
                            cx = ((x1 + x2) / 2.0) / w
                            cy = ((y1 + y2) / 2.0) / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            lines.append(f"{mapped} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    write_label_file(lines, out_txt)

            done += 1
            log(f"[VIS12]   ({done}/{total}) done {sample.sample_id}/{ii}/{len(image_paths)} {image_path.name} boxes={n_boxes}")

    print(f"Finished: {output_root}")
    print(f"Images processed: {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
