from utils import load_json, IMAGE_DIR, COCO_JSON
from PIL import Image, ImageDraw

coco = load_json(COCO_JSON)

# data augmentation
target_dir = IMAGE_DIR / "all-labeled"
target_dir.mkdir(parents=True, exist_ok=True)

bbox_annotations: list = coco["annotations"]
for img in coco["images"]:
    fn = img["file_name"]
    fid = img["id"]
    bboxes = [ano['bbox'] for ano in bbox_annotations if ano['image_id'] == fid]

    image = Image.open(IMAGE_DIR / "all-images" / fn)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x, y, w, h = bbox
        # COCO bbox format: [x, y, width, height]
        draw.rectangle([x, y, x + w, y + h], outline="yellow", width=2)

    image.save(target_dir / fn)
