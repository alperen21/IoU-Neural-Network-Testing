from ultralytics import YOLO
from iou import Image

model_tuples = [(weights, YOLO(weights)) for weights in [
    "yolov5s.pt",
]]

images = [
    Image(
            img_url = "bus.jpg", 
            boxes = [[12, 12, 12, 12], [12, 12, 12, 12]], 
            classes = ["bus", "person"]
        ),
]

iou_threshold = 0.5