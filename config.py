models_path = "./models"
model_weights = [
    "yolov5s.pt",
]
iou_threshold = 0.5
images = [
    {
        "img_url": "bus.jpg", 
        "boxes": [[12, 12, 12, 12], [12, 12, 12, 12]], 
        "classes": ["bus", "person"]
    }
]