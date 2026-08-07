import os
from ultralytics import YOLO
from utils import ROOT
from ultralytics.utils import YAML

DATASET_YAML = ROOT / "yolo26" / "dataset.yaml"

def main() -> None:
    model = YOLO("yolo26l.pt")
    config = YAML.load(DATASET_YAML)
    train_args = dict(config["training"])
    model.train(
        data=DATASET_YAML,
        **train_args,
    )  # model.train(data=str(DATASET_YAML), **train_args)
    print("Train args:", train_args)

if __name__ == "__main__":
    main()
