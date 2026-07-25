"""Convert the fire COCO annotations to labels consumed by Ultralytics YOLO."""

import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COCO_JSON = ROOT / "train_coco.json"
IMAGE_DIR = ROOT / "images" / "train"
LABEL_DIR = ROOT / "labels" / "train"
TRAIN_MANIFEST = ROOT / "coco_train_dataset.txt"
VAL_MANIFEST = ROOT / "coco_val_dataset.txt"
VALIDATION_FRACTION = 0.15
SPLIT_SEED = 0


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_yolo_labels(coco: dict) -> int:
    """Recreate ``labels/train`` directly from the project's COCO annotations."""
    if LABEL_DIR.exists():
        shutil.rmtree(LABEL_DIR)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    images = {image["id"]: image for image in coco["images"]}
    class_ids = {category["id"]: index for index, category in enumerate(coco["categories"])}
    labels: dict[Path, list[tuple[float, float, float, float, float]]] = {}

    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", False):
            continue
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            continue

        box = (
            class_ids[annotation["category_id"]],
            (x + width / 2) / image["width"],
            (y + height / 2) / image["height"],
            width / image["width"],
            height / image["height"],
        )
        label_path = LABEL_DIR / Path(image["file_name"]).with_suffix(".txt")
        image_labels = labels.setdefault(label_path, [])
        if box not in image_labels:
            image_labels.append(box)

    for label_path, boxes in labels.items():
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with label_path.open("w", encoding="utf-8", newline="\n") as file:
            for box in boxes:
                file.write(" ".join(f"{value:g}" for value in box) + "\n")

    return sum(len(boxes) for boxes in labels.values())


def source_sequence(image: dict) -> str:
    """Use the filename prefix before its trailing frame number as source ID."""
    return Path(image["file_name"]).stem.rsplit("_", 1)[0]


def split_images(images: list[dict], positive_image_ids: set[int]) -> tuple[list[dict], list[dict]]:
    """Randomly hold out each collection's positive and negative images with a fixed seed."""
    strata: dict[tuple[str, bool], list[dict]] = {}
    for image in images:
        key = (source_sequence(image), image["id"] in positive_image_ids)
        strata.setdefault(key, []).append(image)

    random_generator = random.Random(SPLIT_SEED)
    val_image_ids: set[int] = set()
    for images_in_stratum in strata.values():
        random_generator.shuffle(images_in_stratum)
        val_count = round(len(images_in_stratum) * VALIDATION_FRACTION)
        if len(images_in_stratum) > 1:
            val_count = min(max(val_count, 1), len(images_in_stratum) - 1)
        val_image_ids.update(image["id"] for image in images_in_stratum[:val_count])

    val_images = [image for image in images if image["id"] in val_image_ids]
    train_images = [image for image in images if image["id"] not in val_image_ids]
    return train_images, val_images


def write_manifest(images: list[dict], destination: Path) -> None:
    missing = [image["file_name"] for image in images if not (IMAGE_DIR / image["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} COCO images are missing from {IMAGE_DIR}: {missing[:5]}")

    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for image in images:
            # The './' makes Ultralytics resolve each entry relative to this
            # manifest, preserving the /images/ -> /labels/ path mapping.
            file.write(f"./images/train/{image['file_name']}\n")


def main() -> None:
    coco = load_json(COCO_JSON)
    annotation_count = write_yolo_labels(coco)
    positive_image_ids = {annotation["image_id"] for annotation in coco["annotations"]}
    train_images, val_images = split_images(coco["images"], positive_image_ids)
    write_manifest(train_images, TRAIN_MANIFEST)
    write_manifest(val_images, VAL_MANIFEST)

    print(f"Created {TRAIN_MANIFEST.name} with {len(train_images)} images.")
    print(
        f"Created {VAL_MANIFEST.name} with {len(val_images)} images "
        f"({len(val_images) / len(coco['images']):.1%}, random seed {SPLIT_SEED})."
    )
    print(f"Created {LABEL_DIR.relative_to(ROOT)} with {annotation_count} fire boxes in {len(list(LABEL_DIR.glob('*.txt')))} files.")


if __name__ == "__main__":
    main()
