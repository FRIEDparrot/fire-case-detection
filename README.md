# fire-case-detection
the detection of fire dataset  

### 1. How to use 

1. copy all the dataset train images to `images/all-images`
2. run `yolo26/scripts/convert_coco_to_yolo.py`, this will generate `train` and `val` folder under the images/all-images
3. run `yolo26/scripts/train_fire_detection.py` to train the model
