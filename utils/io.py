from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"
COCO_JSON = ROOT / "train_coco.json"

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


