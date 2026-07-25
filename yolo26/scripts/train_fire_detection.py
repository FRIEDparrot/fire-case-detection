import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT / "dataset.yaml"

os.chdir(ROOT)

def main() -> None:
    config = YAML.load(DATASET_YAML)
    train_args = dict(config["training"])
    model_name = train_args.pop("model")
    train_args["data"] = str(DATASET_YAML)
    YOLO(model_name).train(**train_args)


if __name__ == "__main__":
    main()
