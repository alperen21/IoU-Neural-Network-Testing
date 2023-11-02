models_path = "./models"
model_weights = [
    "yolov5s.pt",
]
iou_threshold = 0.5
images = [
    {
        "img_url": "test2.jpg", 
        "boxes": [
            [0.0607, 0.3732, 0.3028, 0.8375],
            [2.1884e-04, 5.0712e-01, 9.2044e-02, 8.0601e-01],
            [0.0065, 0.2143, 0.9934, 0.6849],
            [0.8268, 0.3634, 0.9993, 0.8157],
            [0.2732, 0.3762, 0.4264, 0.7948]
            ], 
        "classes": ["bus", "person"]
    }
]