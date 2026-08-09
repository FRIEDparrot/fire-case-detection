"""Validate a fire detector and save confidence-thresholded diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def validate_model(
    model_path: Path,
    data_path: Path,
    output_directory: Path,
    confidence: float,
    *,
    image_size: int = 800,
    batch_size: int = 8,
    device: str = "0",
) -> dict[str, float]:
    """Run Ultralytics validation and save plots in ``output_directory``.

    Args:
        model_path: Path to the detector checkpoint to evaluate.
        data_path: Path to the Ultralytics dataset YAML file.
        output_directory: Destination directory for validation plots and logs.
        confidence: Detection confidence threshold used by the confusion matrix.
        image_size: Inference image size in pixels.
        batch_size: Number of images evaluated per batch.
        device: CUDA device index or ``cpu``.

    Returns:
        The scalar validation metrics reported by Ultralytics.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = YOLO(str(model_path)).val(
        data=str(data_path),
        split="val",
        imgsz=image_size,
        batch=batch_size,
        workers=0,
        conf=confidence,
        iou=0.7,
        device=device,
        plots=True,
        project=str(output_directory.parent),
        name=output_directory.name,
        exist_ok=True,
        verbose=True,
    )
    return {key: float(value) for key, value in metrics.results_dict.items()}


def main() -> None:
    """Parse command-line arguments and validate one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint (.pt) to validate.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML containing the val split.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated plots.")
    parser.add_argument("--conf", type=float, default=0.33, help="Confidence threshold for the matrix.")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu.")
    args = parser.parse_args()
    print(validate_model(args.model, args.data, args.output_dir, args.conf, image_size=args.imgsz, batch_size=args.batch, device=args.device))


if __name__ == "__main__":
    main()

