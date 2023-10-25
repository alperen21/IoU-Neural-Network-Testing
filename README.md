# Testing Module for CellFace

## Features
* IoU (intersection over union) testing for given models and images in terms of object detection
* Performance testing to see how many seconds the inference has took
* Importing detection results with bounding boxes and confidence scores
* Class detection test

## Installation
After cloning this repository run 
```
pip install -r requirements.txt
```
Using Python 3.9.6 is recommended

## Usage
You will need to change the config.py file depending on which model weights and images to use:
- models_path: the path were the model weights are located
- model_weights: list of .pt (PyTorch) files that contain the model weights
- iou_threshold: instances where IoU values are lower than this value will be treated as fail
- images: each image is basically a dictionary consisting of three entries:
    - img_url: url of the image
    - boxes: list of the ground truth boxes of the image
    - classes: list of which objects are present in the object

After configuring the config.py file, run the test using:
```
python test.py
```
After the test execution is complete, you will see the log of the execution under logs subdirectory and output images including the bounding boxes under output_images subdirectory.