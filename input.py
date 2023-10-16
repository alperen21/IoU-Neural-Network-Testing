from ultralytics import YOLO
from iou import Image

models = [(weights, YOLO(weights)) for weights in [
    "yolov5s.pt",
]]

images = [
    Image("bus.jpg", [[12, 12, 12, 12], [12, 12, 12, 12]]),
]