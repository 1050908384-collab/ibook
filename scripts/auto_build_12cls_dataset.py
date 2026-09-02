#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_LABEL = 255
MASK_FOLDER_NAME = "mask"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_dir: Path
    image_dir: Path


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


def ensure_clean_dir(path: Path) -> None:
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


def write_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)


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


def log(message: str) -> None:
    print(message, flush=True)

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

CLASS_PROMPTS = {
    0: ["person", "pedestrian"],
    1: ["boat", "ship"],
    2: ["animal", "dog", "cat", "bird", "horse", "cow"],
    3: ["seat", "chair", "bench", "sofa"],
    4: ["sign", "traffic sign", "road sign", "signboard", "banner"],
    5: ["bicycle", "motorcycle", "electric bicycle", "two-wheeled vehicle"],
    6: ["car", "bus", "truck", "four-wheeled vehicle"],
    7: ["ball", "sports ball"],
    8: ["light", "traffic light", "street light", "lamp", "ceiling light"],
    9: ["garbage can", "trash can", "waste bin"],
    10: ["uav", "drone", "quadcopter"],
    11: ["tricycle", "three-wheeled vehicle"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically build the custom 12-class semantic segmentation dataset from color images."
    )
    parser.add_argument("input_root", type=Path, help="Source dataset root, e.g. /data/lmz/train_data")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output dataset root. Default: RGBDT500_12",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local SAM 3 checkpoint path, e.g. /data/lmz/weights/sam3.pt",
    )
    parser.add_argument(
        "--mask-folder",
        default=MASK_FOLDER_NAME,
        help=f"Mask folder name inside each sample. Default: {MASK_FOLDER_NAME}",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "none"],
        default="hardlink",
        help="How to materialize original files into the new dataset root.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Concept confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--quantize", type=int, default=16, help="SAM 3 quantization level.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip mask generation when the output mask already exists.",
    )
    return parser.parse_args()


def _to_numpy_masks(masks) -> np.ndarray:
    if masks is None:
        return np.empty((0, 0, 0), dtype=np.uint8)
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return masks.astype(np.uint8)


def _extract_masks_from_results(results) -> np.ndarray:
    if results is None:
        return np.empty((0, 0, 0), dtype=np.uint8)

    if isinstance(results, tuple) and results:
        return _to_numpy_masks(results[0])

    if isinstance(results, list) and results:
        first = results[0]
        if hasattr(first, "masks") and first.masks is not None and hasattr(first.masks, "data"):
            return _to_numpy_masks(first.masks.data)

    if hasattr(results, "masks") and results.masks is not None and hasattr(results.masks, "data"):
        return _to_numpy_masks(results.masks.data)

    raise RuntimeError(
        "Could not extract masks from SAM 3 output. Check that ultralytics>=8.3.237 is installed."
    )


def build_mask_for_image(predictor, image_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    predictor.set_image(str(image_path))
    predictor.reset_prompts()
    mask = np.full(image_shape, IGNORE_LABEL, dtype=np.uint8)
    overlays: list[tuple[int, int, np.ndarray]] = []

    for class_id, prompts in CLASS_PROMPTS.items():
        for prompt in prompts:
            results = predictor.inference(predictor.batch[1][0], text=[prompt])
            masks = _extract_masks_from_results(results)
            if masks.size == 0:
                continue
            for instance_mask in masks:
                binary_mask = instance_mask.astype(bool)
                if binary_mask.shape != image_shape:
                    binary_mask = np.array(
                        Image.fromarray(binary_mask.astype(np.uint8) * 255).resize(
                            (image_shape[1], image_shape[0]),
                            resample=Image.NEAREST,
                        )
                    ) > 0
                area = int(binary_mask.sum())
                if area == 0:
                    continue
                overlays.append((area, class_id, binary_mask))

    # Let smaller masks overwrite larger ones so tiny objects are less likely to disappear.
    overlays.sort(key=lambda item: item[0], reverse=True)
    for _, class_id, binary_mask in overlays:
        mask[binary_mask] = class_id
    return mask


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root.with_name("RGBDT500_12")).resolve()

    if not args.model:
        raise SystemExit("Please pass a local SAM 3 checkpoint with --model.")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"SAM 3 checkpoint not found locally: {model_path}")

    from ultralytics.models.sam import SAM3SemanticPredictor

    predictor = SAM3SemanticPredictor(
        overrides={
            "conf": args.conf,
            "task": "segment",
            "mode": "predict",
            "model": str(model_path),
            "imgsz": args.imgsz,
            "quantize": args.quantize,
            "verbose": False,
        }
    )

    samples = discover_samples(input_root)
    if not samples:
        raise SystemExit(f"No sample folders with `color/` were found in {input_root}")
    sample_images = [(sample, collect_images(sample.image_dir)) for sample in samples]
    total_images = sum(len(images) for _, images in sample_images)

    log(f"[12CLS] output={output_root}")
    log(f"[12CLS] samples={len(sample_images)} images={total_images}")

    write_class_file(output_root, CLASS_NAMES_12)
    write_meta_file(
        output_root,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(input_root),
            "generator": "auto_build_12cls_dataset.py",
            "model": str(model_path),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "quantize": args.quantize,
            "ignore_label": IGNORE_LABEL,
            "classes": CLASS_NAMES_12,
            "prompts": CLASS_PROMPTS,
        },
    )

    done_images = 0
    for sample_idx, (sample, image_paths) in enumerate(sample_images, start=1):
        log(f"[12CLS] sample {sample_idx}/{len(sample_images)} {sample.sample_id} ({len(image_paths)} images)")
        target_sample_dir = output_root / sample.sample_id
        mirror_sample_assets(sample.source_dir, target_sample_dir, args.mask_folder, args.link_mode)

        for image_idx, image_path in enumerate(image_paths, start=1):
            output_mask = target_sample_dir / args.mask_folder / image_path.name
            if args.skip_existing and output_mask.exists():
                done_images += 1
                log(f"[12CLS]   ({done_images}/{total_images}) skip {sample.sample_id}/{image_idx}/{len(image_paths)} {image_path.name}")
                continue

            with Image.open(image_path) as image:
                width, height = image.size
            semantic_mask = build_mask_for_image(predictor, image_path, (height, width))
            write_mask(semantic_mask, output_mask)
            done_images += 1
            log(f"[12CLS]   ({done_images}/{total_images}) done {sample.sample_id}/{image_idx}/{len(image_paths)} {image_path.name}")

    print(f"Finished 12-class dataset build: {output_root}")
    print(f"Samples: {len(sample_images)}")
    print(f"Images processed: {done_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
