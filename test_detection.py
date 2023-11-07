from gather_input import gather_models, gather_images
import os
import torch
from config import config
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from logger.logger import setup_logger, get_filename

def calculate_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-9): 
     # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4 
    box2 = box2.T 
  
     # Get the coordinates of bounding boxes 
    if x1y1x2y2:  # x1, y1, x2, y2 = box1 
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3] 
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3] 
    else:  # transform from xywh to xyxy 
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2 
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2 
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2 
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2 
  
     # Intersection area 
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0) 
  
     # Union Area 
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps 
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps 
    union = w1 * h1 + w2 * h2 - inter + eps 
  
    iou = inter / union 
    if GIoU or DIoU or CIoU: 
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width 
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height 
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1 
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared 
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + 
                     (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared 
            if DIoU: 
                return iou - rho2 / c2  # DIoU 
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47 
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2) 
                with torch.no_grad(): 
                    alpha = v / ((1 + eps) - iou + v) 
                return iou - (rho2 / c2 + v * alpha)  # CIoU 
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf 
            c_area = cw * ch + eps  # convex area 
            return iou - (c_area - union) / c_area  # GIoU 
    else: 
        return iou  # IoU 

class TestDetection:
    def __init__(self) -> None:
        self.img_objects = gather_images()
        self.model_tuples = gather_models()
        self.logger = setup_logger(os.path.join("logs",get_filename()))

    def draw_bounding_boxes(self, image_path, objects, names):
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
            wh_rounding = 8
            coordinate_rounding = 15
            x_min_abs = int(round((x_min * img_width))) - coordinate_rounding  # add 0.5 for rounding
            y_min_abs = int(round((y_min * img_height)))  - coordinate_rounding # add 0.5 for rounding
            box_width_abs = int((box_width * img_width) + wh_rounding ) 
            box_height_abs = int((box_height * img_height) + wh_rounding )

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
            
            object_class = int(obj.object_class)
            object_class_name = names[object_class]
            # Draw label
            label = f"{object_class_name} {message}".strip()

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
                        if iou >= max_iou:
                            max_iou = iou
                            true_class = int(object.object_class)
                            true_object = object
                    
            
                    passed_iou_threshold = None
                    if max_iou < config.threshold:
                        self.logger.error(f"iou test failed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}, iou: {max_iou}")
                        test_passed=False
                        passed_iou_threshold = False
                    else:
                        self.logger.info(f"iou test passed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}, iou: {max_iou}")
                        passed_iou_threshold = True
                        true_object.passed_iou_threshold = True

                    
                    correct_classification = None
                    if true_class == predicted_object_class:
                        self.logger.info(f"class test passed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}, predicted: {predicted_object_class}, true: {true_class} ")
                        correct_classification = True
                        true_object.correct_classification = True
                    else:
                        self.logger.error(f"class test failed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}, predicted: {predicted_object_class}, true: {true_class} ")
                        test_passed=False
                        correct_classification = False

                
                names = prediction.names
                img = self.draw_bounding_boxes(
                    image_path = img_object.img_url,
                    objects = img_object.objects,
                    names = names
                )

              

                filepath, file_extension = os.path.splitext(img_object.img_url)
                filename = filepath.split(os.sep)[-1]

                # img.save(os.path.join("output",filename+file_extension))
                cv2.imwrite(os.path.join("output",filename+file_extension), img)

                #if test_passed:
                #    self.logger.info("test passed")
                #else:
                #    self.logger.error("test failed")

if __name__ == "__main__":
    TestDetection().test()