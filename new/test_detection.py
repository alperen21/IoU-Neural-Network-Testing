from gather_input import gather_models, gather_images
import os
import torch
from config import config
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

class TestResultObject:
    def __init__(self, objectClass, objectBoundingBox, correct_class, passed_iou_value) -> None:
        self.objectClass = objectClass
        self.objectBoundingBox = objectBoundingBox
        self.correct_class = correct_class
        self.passed_iou_value = passed_iou_value
        

class TestDetection:
    def __init__(self) -> None:
        self.img_objects = gather_images()
        self.model_tuples = gather_models()

    def draw_bounding_boxes(self, image_path, test_result_objects):
        # Load the image
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size

            # Load a font
            font = ImageFont.load_default()

            for obj in test_result_objects:
                message = ""
                # Convert normalized coordinates to absolute coordinates
                x1, y1, x2, y2 = obj.objectBoundingBox
                abs_box = [
                    x1 * img_width,
                    y1 * img_height,
                    x2 * img_width,
                    y2 * img_height,
                ]

                # Determine color based on the flags
                color = 'green'
                if not obj.passed_iou_value:
                    color = 'yellow'
                    message += "iou "

                if not obj.correct_class:
                    color = 'red'
                    message += "class "
                # Draw the bounding box
                draw.rectangle(abs_box, outline=color, width=2)

                # Draw label
                label = f"{obj.objectClass} {message}" 
                # Approximate the text size based on the length of the label and font size
                font_size = 12  # Assuming a default font size
                text_length = len(label) * font_size
                text_height = font_size  # Approximate height for most fonts
                text_position = (abs_box[0], abs_box[1] - text_height)

                # Draw a filled rectangle behind the text for better visibility
                draw.rectangle((text_position[0], text_position[1], text_position[0] + text_length, text_position[1] + text_height), fill=color)
                # Draw the text on top of the filled rectangle
                draw.text(text_position, label, fill='white', font=font)

            # Save or display the image
            img.show()
            img.save("annotated_image.png")



    
    def test(self):
        test_passed = True
        for model_tuple in self.model_tuples:
            model = model_tuple.model
            
            print(f"testing: {model_tuple.model_weight_file}")

            for img_object in self.img_objects:
                testResultObjects = list()
                print(f"using: {img_object.img_url}")

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
                    
            
                    passed_iou_value = None
                    if max_iou < config.threshold:
                        print("iou test failed for:", true_object, "using model:", model_tuple)
                        test_passed=False
                        passed_iou_value = False
                    else:
                        print("iou test passed for:", true_object, "using model:", model_tuple)
                        passed_iou_value = True

                    
                    correct_class = None
                    if true_class == predicted_object_class:
                        print("class test passed for:", true_object, "using model:", model_tuple)
                        correct_class = True
                    else:
                        print("class test failed for:", true_object, "using model:", model_tuple)
                        test_passed=False
                        correct_class = False

                    testResultObjects.append(
                            TestResultObject(
                                    objectClass = true_class, 
                                    objectBoundingBox = true_object.bounding_box,
                                    passed_iou_value = passed_iou_value,
                                    correct_class = correct_class
                                )
                        )

                self.draw_bounding_boxes(
                    image_path = img_object.img_url,
                    test_result_objects = testResultObjects
                ) 


                    

                    


                # print(box)
                # print([prediction.names[int(elem)] for elem in object_class])




TestDetection().test()
