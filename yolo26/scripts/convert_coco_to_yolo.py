"""Build a leakage-free YOLO train/validation dataset from COCO annotations."""
import random
import shutil
from pathlib import Path

from utils import COCO_JSON, IMAGE_DIR, LABEL_DIR, load_json


VALIDATION_FRACTION = 0.2


def write_yolo_labels(coco: dict, images: list[dict], label_directory: Path) -> int:
    """Write YOLO labels for ``images`` into a cleared ``label_directory``.

    Args:
        coco: COCO annotation data containing image metadata and bounding boxes.
        images: The split-specific COCO image entries to label.
        label_directory: Destination directory for the generated ``.txt`` labels.
    """
    if label_directory.exists():
        shutil.rmtree(label_directory)
    label_directory.mkdir(parents=True, exist_ok=True)

    images_by_id = {image["id"]: image for image in images}
    class_ids = {category["id"]: index for index, category in enumerate(coco["categories"])}
    labels: dict[Path, list[tuple[float, float, float, float, float]]] = {
        label_directory / Path(image["file_name"]).with_suffix(".txt"): []
        for image in images
    }

    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", False):
            continue
        image = images_by_id.get(annotation["image_id"])
        if image is None:
            continue

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
        label_path = label_directory / Path(image["file_name"]).with_suffix(".txt")
        if box not in labels[label_path]:
            labels[label_path].append(box)

    for label_path, boxes in labels.items():
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with label_path.open("w", encoding="utf-8", newline="\n") as file:
            for box in boxes:
                file.write(" ".join(f"{value:g}" for value in box) + "\n")

    return sum(len(boxes) for boxes in labels.values())


def split_images(images: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hold out complete source sequences without exceeding the validation target.

    Args:
        images: All COCO image entries to partition by source sequence.
    """
    remaining = round(len(images) * VALIDATION_FRACTION)
    rng = random.Random(42)  # use a random source
    train_images = images  # use all images as the train images
    val_images = rng.sample(images, remaining)
    return train_images, val_images


def copy_images(images: list[dict], source_directory: Path, destination: Path) -> None:
    """Copy a split's images into a cleared destination directory.

    Args:
        images: COCO image entries included in this split.
        source_directory: Directory containing the unsplit source image files.
        destination: Directory where this split's image files will be copied.
    """
    missing = [image["file_name"] for image in images if not (source_directory / image["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} COCO images are missing from {source_directory}: {missing[:5]}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for image in images:
        source = source_directory / image["file_name"]
        target = destination / image["file_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    """Create paired train/validation image and label directories."""
    coco = load_json(COCO_JSON)
    source_image_dir = IMAGE_DIR / "all-images"
    train_images, val_images = split_images(coco["images"])

    copy_images(train_images, source_image_dir, IMAGE_DIR / "train")
    copy_images(val_images, source_image_dir, IMAGE_DIR / "val")
    train_box_count = write_yolo_labels(coco, train_images, LABEL_DIR / "train")
    val_box_count = write_yolo_labels(coco, val_images, LABEL_DIR / "val")

    print(f"Created images/train with {len(train_images)} images and {train_box_count} fire boxes.")
    print(
        f"Created images/val with {len(val_images)} images and {val_box_count} fire boxes "
        f"({len(val_images) / len(coco['images']):.1%} validation)."
    )


if __name__ == "__main__":
    main()
