"""Randomly display fire detection samples with COCO bounding boxes."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_coco_annotations(coco_json: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    with coco_json.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[annotation["image_id"]].append(annotation)

    return coco.get("images", []), annotations_by_image


def image_path(images_dir: Path, file_name: str) -> Path:
    path = images_dir / file_name
    if path.exists():
        return path

    stem = Path(file_name).stem
    matches = [candidate for candidate in images_dir.iterdir() if candidate.stem == stem]
    if matches:
        return matches[0]

    return path


def draw_sample(ax, path: Path, annotations: list[dict]) -> None:
    with Image.open(path) as image:
        ax.imshow(image.convert("RGB"))

    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        box = Rectangle(
            (x, y),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(box)

    fire_count = len(annotations)
    title = "no fire" if fire_count == 0 else f"{fire_count} fire(s)"
    ax.set_title(f"{path.name}\n{title}", fontsize=9)
    ax.axis("off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly show images with fire bounding boxes from COCO labels."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("images/train"))
    parser.add_argument("--coco-json", type=Path, default=Path("train_coco.json"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the sampled grid before showing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    images, annotations_by_image = load_coco_annotations(args.coco_json)
    available_images = [
        image
        for image in images
        if image_path(args.images_dir, image["file_name"]).suffix.lower() in IMAGE_EXTENSIONS
        and image_path(args.images_dir, image["file_name"]).exists()
    ]
    if not available_images:
        raise FileNotFoundError(f"No labeled images found in {args.images_dir}")

    rng = random.Random(args.seed)
    sample_count = min(args.count, len(available_images))
    samples = rng.sample(available_images, sample_count)

    columns = min(5, sample_count)
    rows = math.ceil(sample_count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3.4))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, image in zip(axes_list, samples):
        path = image_path(args.images_dir, image["file_name"])
        annotations = annotations_by_image.get(image["id"], [])
        draw_sample(ax, path, annotations)

    for ax in axes_list[sample_count:]:
        ax.axis("off")

    fig.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
    plt.show()


if __name__ == "__main__":
    main()

