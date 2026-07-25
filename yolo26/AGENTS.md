## Project Task

This project is a fire object detection dataset. Each image can contain zero, one, or many visible fire regions.

Data layout:

- `images/` contains the image files.
- `train_image.json` maps each image filename to an image-level label: `1` means fire is present, `0` means no fire.
- `train_coco.json` is the localization ground truth in COCO-style format with exactly three top-level keys: `categories`, `images`, and `annotations`.
- `categories` currently contains one class: `fire` with `id: 0`.
- `images` records image metadata such as `id`, `file_name`, `width`, and `height`.
- `annotations` records each fire instance. Match annotations to images with `annotation.image_id == image.id`.
- Bounding boxes use COCO format: `[x, y, width, height]` in pixel coordinates.

What to do:

- Treat this as object detection, not only image classification. The detector must return every fire bounding box and also handle images with no fire.
- Use `train_coco.json` as the source of truth for training and evaluating localization.
- Use `train_image.json` for sanity checks, dataset summaries, or image-level fire/no-fire analysis.
- Preserve original filenames, image IDs, category IDs, and COCO bbox format unless an explicit conversion script is being written.
- Keep validation/test splits leakage-free. Frames from the same video/source sequence should stay in the same split when possible.
- Evaluate with detection metrics such as mAP@50, mAP@50-95, precision, recall, and false positives on no-fire images.
- Pay special attention to small fires and confusing hard negatives such as sunsets, lights, smoke, reflections, and orange objects.
