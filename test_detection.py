from gather_input import gather_models, gather_images, delete_files_in_directory, delete_specific_files
import os
import torch
from config import config
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from logger.logger import setup_logger, get_filename
from multiprocessing import Pool

PARALLEL = True

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
        self.logger = setup_logger(os.path.join("logs",get_filename()))
        self.logger.info("deleting files from the previous test in output directory")
        delete_files_in_directory("output")
        self.img_objects = gather_images()
        self.model_tuples = gather_models()

        self.passed_class_count = 0
        self.failed_class_count = 0
        self.passed_iou_count = 0
        self.failed_iou_count = 0

        
        if not os.path.exists(os.path.join(".", "output")):
            os.makedirs(os.path.join(".", "output"))

    def draw_square(self, img, label, index, x, y, w, h, thickness=1):
        color_map = {
            '0': (0, 0, 255),
            '1': (255, 0, 0),
            '2': (0, 255, 0),
            # Additional labels can be added here if necessary
        }
        color = color_map.get(label, (255, 0, 0))  # Default color is white

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(img, str(index), (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


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

        h,w,c = img.shape
        for obj in objects:
            message = ""

            # Convert normalized coordinates to absolute coordinates and apply a small adjustment
            p0,p1,p2,p3 = obj.bounding_box

            x_max = int((w / 2) * (2 * p0 + p2))
            y_max = int((h / 2) * (2 * p1 + p3))
            w_box = int(p2 * w)
            h_box = int(p3 * h)
            x_min = int(x_max - w_box)
            y_min = int(y_max - h_box)

            self.draw_square(img, "label", 1, x_min, y_min, w_box, h_box, thickness=1)

            
        # Show the image
        # cv2.imshow('Image with Bounding Boxes', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Save or return the image if you need
        # cv2.imwrite('output_image_path.jpg', img)
        img_original = cv2.imread(image_path)
        combined_image = cv2.hconcat([img_original, img])

        return combined_image

    def test_img_parallel(self, arguments):
        img_object, model_tuple = arguments
        return self.test_img(img_object, model_tuple)


    def test_img(self, img_object, model_tuple):
        model = model_tuple.model
        self.logger.info(f"using: {img_object.img_url}")

        prediction = model.predict(os.path.join(".", img_object.img_url))[0] #this is fine since we only give one image as input and not an array of images
        predicted_object_classes = prediction.boxes.cls
        predicted_boxes = prediction.boxes.xyxyn

        for predicted_object_class, predicted_box in zip(predicted_object_classes, predicted_boxes):
            predicted_object_class = int(predicted_object_class)
            max_iou = float('-inf')
            true_class = -1
            true_object= None

            if len(img_object.objects) == 0:
                self.logger.error(f"no objects found in image: {img_object.img_url}")
                test_passed=False
                continue

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
                self.failed_iou_count += 1
            else:
                self.logger.info(f"iou test passed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}, iou: {max_iou}")
                passed_iou_threshold = True
                true_object.passed_iou_threshold = True
                self.passed_iou_count += 1

            
            correct_classification = None
            if true_class == predicted_object_class:
                self.logger.info(f"class test passed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}, predicted: {predicted_object_class}, true: {true_class} ")
                correct_classification = True
                true_object.correct_classification = True
                self.passed_class_count += 1
            else:
                self.logger.error(f"class test failed for:, {prediction.names[int(true_object.object_class)]}, using model:, {model_tuple.model_weight_file}, predicted: {predicted_object_class}, true: {true_class} ")
                test_passed=False
                correct_classification = False
                self.failed_class_count += 1

        
        names = prediction.names
        img = self.draw_bounding_boxes(
            image_path = img_object.img_url,
            objects = img_object.objects,
            names = names
        )

        filepath, file_extension = os.path.splitext(img_object.img_url)
        filename = filepath.split(os.sep)[-1]
        cv2.imwrite(os.path.join("output",filename+file_extension), img)

        return {
            "passed_iou_count" : self.passed_iou_count,
            "failed_iou_count" : self.failed_iou_count,
            "passed_class_count" : self.passed_class_count,
            "failed_class_count" : self.failed_class_count
        }
    
    def test(self):
        test_passed = True
        for model_tuple in self.model_tuples:
            model = model_tuple.model
            
            self.logger.info(f"testing: {model_tuple.model_weight_file}")

            if PARALLEL:
                results = []
                with Pool(os.cpu_count()) as p:
                    results.extend(p.map(self.test_img_parallel, [(img_object, model_tuple) for img_object in self.img_objects]))
                
                for result in results:
                    self.passed_iou_count += result["passed_iou_count"]
                    self.failed_iou_count += result["failed_iou_count"]
                    self.passed_class_count += result["passed_class_count"]
                    self.failed_class_count += result["failed_class_count"]
            else:
                for img_object in self.img_objects:
                    self.test_img(img_object, model_tuple)
                
    def print_results(self):
        self.logger.info(f"passed_iou_count: {self.passed_iou_count}")
        self.logger.info(f"failed_iou_count: {self.failed_iou_count}")
        self.logger.info(f"passed_class_count: {self.passed_class_count}")
        self.logger.info(f"failed_class_count: {self.failed_class_count}")

    def clean(self):
        self.logger.info("cleaning up...")
        delete_specific_files('.', '._')

if __name__ == "__main__":
    testObject = TestDetection()
    testObject.test()
    testObject.clean()
    testObject.print_results()