import os
from ultralytics import YOLO
from utils import ROOT
from ultralytics.utils import YAML

DATASET_YAML = ROOT / "yolo26" / "dataset.yaml"

def main() -> None:
    model = YOLO("yolo26m.pt")
    model.train(data=DATASET_YAML)  # model.train(data=str(DATASET_YAML), **train_args)
    config = YAML.load(DATASET_YAML)
    train_args = dict(config["training"])
    print("Train args:", train_args)


if __name__ == "__main__":
    main()
