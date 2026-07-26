import os
from ultralytics import YOLO
from ultralytics.utils import YAML
from utils import ROOT

DATASET_YAML = ROOT / "yolo26" / "dataset.yaml"

def main() -> None:
    config = YAML.load(DATASET_YAML)
    train_args = dict(config["training"])
    model_name = train_args.pop("model")
    train_args["data"] = str(DATASET_YAML)
    YOLO(model_name).train(**train_args)


if __name__ == "__main__":
    main()
