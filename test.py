from ultralytics import YOLO
from iou import bbox_iou, draw_predictions
from input import models, images
import statistics
import time
import os
import cv2
from logger import setup_logger, get_filename


class Test_IoU:
    def __init__(self, models, images) -> None:
        self.models = models
        self.images = images
        self.logger = setup_logger(os.path.join(".", "logs", get_filename()))
    
    def test_iou(self, model_tuple):
        iou_results = [0]
        time_results = list()

        model_name, model = model_tuple
        
        for image in images:

            start_time = time.time()
            results = model.predict(image.img_url)
            end_time = time.time()

            for result in results:
                
                for box in result.boxes.xyxy:
                    time_result = end_time - start_time
                    iou_result = bbox_iou(box.unsqueeze(0), box.unsqueeze(0)).item()
                    
                    iou_results.append(iou_result)
                    time_results.append(time_result)
                    
                    self.logger.info(f"{model_name}, {image.img_url}, iou: {iou_result}, time: {time_result}, box: {box}")
    
        self.logger.info("*"*20)
        print("average results")
        average_iou = statistics.mean(iou_results)
        average_time = statistics.mean(time_results)

        self.logger.info(f"Average time for {model_name} is {average_time}")
        self.logger.info(f"Average IoU for {model_name} is {average_iou}")

        
    
    def test_class(self, model_tuple):
        model_name, model = model_tuple

        for image in images:
            img = cv2.imread(image.img_url)
            results = model.predict(img)
            boxes_object = results[0].boxes
            labels = boxes_object.cls  # Use 'cls' for class labels (e.g., 0 for 'person', 5 for 'bus', etc.)
            names = results[0].names
            class_names = [names[int(i)] for i in labels]

            try:
                for class_ in image.classes:
                    assert class_ in class_names, f"Model: {model_name} could not detect class: {class_} in image {image.img_url}"
                #yanlış index verirse falan hangisinin olduğunu yazdır, iou mu yanlış yoksa class mı yanlış differentiate
                for class_ in class_names:
                    assert class_ in image.classes, f"Model: {model_name} detected an extra class: {class_} in image {image.img_url}"
            except Exception as e:
                self.logger.error(e)
                raise e

    def print_predictions(self, model_tuple):
        model_name, model = model_tuple

        for img in images:
            results = model.predict(img.img_url)

            for index, result in enumerate(results):
                boxes_object = result.boxes
                bounding_boxes = boxes_object.xyxy  # This gives [x1, y1, x2, y2] for each detection
                labels = boxes_object.cls  # Use 'cls' for class labels (e.g., 0 for 'person', 5 for 'bus', etc.)
                names = result.names
                confidence_scores = boxes_object.conf  # This gives confidence scores for each detection

                class_names = [names[int(i)] for i in labels]

                img_with_boxes = draw_predictions(result.orig_img, bounding_boxes, class_names, confidence_scores)

                output_folder = os.path.join('output_images', model_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                output_path = os.path.join(output_folder, f'prediction_{img.img_url}_{index}.jpg')
                cv2.imwrite(output_path, img_with_boxes)
                self.logger.info(f"Image saved to {output_path}")

            # her bounding box için yap

    def run(self):
        for model in models:
            self.test_class(model)
            self.test_iou(model)
            self.print_predictions(model)


if __name__ == "__main__":
    test = Test_IoU(models, images)
    test.run()