from utils import load_json, IMAGE_DIR, COCO_JSON
from PIL import Image, ImageDraw

coco = load_json(COCO_JSON)

# data augmentation
target_dir = IMAGE_DIR / "all-labeled"
target_dir.mkdir(parents=True, exist_ok=True)

bbox_annotations: list = coco["annotations"]

max_width = 0
max_height = 0
m_file = ""
max_bh = 0
max_bw = 0
for img in coco["images"]:
    fn = img["file_name"]
    fid = img["id"]
    bboxes = [ano['bbox'] for ano in bbox_annotations if ano['image_id'] == fid]

    image = Image.open(IMAGE_DIR / "all-images" / fn)
    img_width, img_height = image.size
    if img_width > max_width:
        max_width = img_width
        max_height = img_height
        m_file = fn

    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x, y, w, h = bbox
        if h > max_bh:
            max_bh = h
        if w > max_bw:
            max_bw = w

        # COCO bbox format: [x, y, width, height]
        draw.rectangle([x, y, x + w, y + h], outline="yellow", width=2)
    image.save(target_dir / fn)

print(f"Max width: {max_width}, Max height: {max_height}")
print(f"file: {m_file}")

print("max bbox width: ", max_bw)
print("max bbox height: ", max_bh)
