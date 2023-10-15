from ultralytics import YOLO
from iou import bbox_iou
from input import models, images
import statistics
import time


class Test_IoU:
    def __init__(self, models, images) -> None:
        self.models = models
        self.images = images
    
    def test_iou(self, model):
        iou_results = list()
        time_results = list()
        
        for image in images:
            start_time = time.time()
            results = model.predict(image)
            end_time = time.time()

            boxes = results[0].boxes.xyxy[0]        

            iou_results.append(bbox_iou(boxes, boxes))
            time_results.append(end_time - start_time)
            
    
        average_iou = statistics.mean(iou_results)
        average_time = statistics.mean(time_results)

        print(f"Average time for {model.name} is {average_time}")
        print(f"Average IoU for {model.name} is {average_iou}")
    
    def run(self):
        for model in models:
            self.test_iou(model)


if __name__ == "__main__":
    test = Test_IoU(models, images)
    test.run()