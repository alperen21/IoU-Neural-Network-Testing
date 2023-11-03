from gather_input import gather_models, gather_images
import os
import torch
from config import config
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from logger.logger import setup_logger, get_filename

def calculate_iou(box1, box2):
    # Make sure that the boxes are 1-D tensors
    assert box1.dim() == 1 and box2.dim() == 1, "Box tensors must be 1-D"
    assert box1.numel() == 4 and box2.numel() == 4, "Box tensors must have 4 elements"

    # Calculate intersection
    x_left = max(box1[0], box2[0]).item()
    y_top = max(box1[1], box2[1]).item()
    x_right = min(box1[2], box2[2]).item()
    y_bottom = min(box1[3], box2[3]).item()

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return float(iou)        

class TestDetection:
    def __init__(self) -> None:
        self.img_objects = gather_images()
        self.model_tuples = gather_models()
        self.logger = setup_logger(os.path.join("logs",get_filename()))

    def draw_bounding_boxes(self, image_path, objects):
        # Read the image
        img = cv2.imread(image_path)
        
        if img is None:
            print("Error: The image could not be read.")
            return None

        img_height, img_width = img.shape[:2]

        # Define the font for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        for obj in objects:
            message = ""

            # Convert normalized coordinates to absolute coordinates and apply a small adjustment
            x_min, y_min, box_width, box_height = obj.bounding_box
            x_min_abs = int((x_min * img_width) - 5)  # add 0.5 for rounding
            y_min_abs = int((y_min * img_height) - 5) # add 0.5 for rounding
            box_width_abs = int((box_width * img_width) - 5) 
            box_height_abs = int((box_height * img_height) - 5)

            # Calculate the bottom right corner from the top left corner and width and height
            x_max_abs = x_min_abs + box_width_abs
            y_max_abs = y_min_abs + box_height_abs

            # Correction for potential offset
            offset_correction = 1  # You might need to adjust this value
            x_max_abs -= offset_correction
            y_max_abs -= offset_correction

            # Determine color based on the flags
            color = (0, 255, 0)  # Green
            if not obj.passed_iou_threshold:
                color = (0, 255, 255)  # Yellow
                message += "iou "
            if not obj.correct_classification:
                color = (0, 0, 255)  # Red
                message += "class "

            # Draw the bounding box
            cv2.rectangle(img, (x_min_abs, y_min_abs), (x_max_abs, y_max_abs), color, 2)

            # Draw label
            label = f"{obj.object_class} {message}".strip()

            # Get the width and height of the text box
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size
            text_x1, text_y1 = x_min_abs, y_min_abs - 10

            # Draw a filled rectangle behind the text for better visibility
            cv2.rectangle(img, (text_x1, text_y1 - text_height - 5), (text_x1 + text_width, text_y1), color, cv2.FILLED)

            # Draw the text on top of the filled rectangle
            cv2.putText(img, label, (text_x1, text_y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

        # Show the image
        # cv2.imshow('Image with Bounding Boxes', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Save or return the image if you need
        # cv2.imwrite('output_image_path.jpg', img)
        return img


    
    def test(self):
        test_passed = True
        for model_tuple in self.model_tuples:
            model = model_tuple.model
            
            self.logger.info(f"testing: {model_tuple.model_weight_file}")

            for img_object in self.img_objects:
                self.logger.info(f"using: {img_object.img_url}")

                prediction = model.predict(os.path.join(".", img_object.img_url))[0] #this is fine since we only give one image as input and not an array of images
                predicted_object_classes = prediction.boxes.cls
                predicted_boxes = prediction.boxes.xyxyn

                for predicted_object_class, predicted_box in zip(predicted_object_classes, predicted_boxes):
                    predicted_object_class = int(predicted_object_class)
                    max_iou = float('-inf')
                    true_class = -1
                    true_object= None

                    for object in img_object.objects:
                        iou = calculate_iou(torch.tensor(object.bounding_box).reshape(-1), predicted_box.reshape(-1))
                        
                        if iou > max_iou:
                            max_iou = iou
                            true_class = int(object.object_class)
                            true_object = object
                    
            
                    passed_iou_threshold = None
                    if max_iou < config.threshold:
                        self.logger.error(f"iou test failed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}")
                        test_passed=False
                        passed_iou_threshold = False
                    else:
                        self.logger.info(f"iou test passed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}")
                        passed_iou_threshold = True

                    
                    correct_classification = None
                    if true_class == predicted_object_class:
                        self.logger.info(f"class test passed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}")
                        correct_classification = True
                    else:
                        self.logger.error(f"class test failed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}")
                        test_passed=False
                        correct_classification = False

                
                img = self.draw_bounding_boxes(
                    image_path = img_object.img_url,
                    objects = img_object.objects,
                )

                filepath, file_extension = os.path.splitext(img_object.img_url)
                filename = filepath.split(os.sep)[-1]

                # img.save(os.path.join("output",filename+file_extension))
                cv2.imwrite(os.path.join("output",filename+file_extension), img)

                if test_passed:
                    self.logger.info("test passed")
                else:
                    self.logger.error("test failed")

if __name__ == "__main__":
    TestDetection().test()