from gather_input import gather_models, gather_images
import os
import torch

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
    
    def test(self):
        for model_tuple in self.model_tuples:
            model = model_tuple.model
            
            print(f"testing: {model_tuple.model_weight_file}")

            for img_object in self.img_objects:
                print(f"using: {img_object.img_url}")

                prediction = model.predict(os.path.join(".", img_object.img_url))[0] #this is fine since we only give one image as input and not an array of images
                predicted_object_classes = prediction.boxes.cls
                predicted_boxes = prediction.boxes.xyxyn

                for predicted_object_class, predicted_box in zip(predicted_object_classes, predicted_boxes):
                    predicted_object_class = int(predicted_object_class)
                    max_iou = float('-inf')
                    true_class = -1

                    for object in img_object.objects:
                        iou = calculate_iou(torch.tensor(object.bounding_box).reshape(-1), predicted_box.reshape(-1))
                        
                        if iou > max_iou:
                            max_iou = iou
                            true_class = int(object.object_class)
                    
                    

                    
                    print("check")
                    print("predicted class:", prediction.names[predicted_object_class], ", true class: ", prediction.names[true_class], "iou value:", max_iou)
                    print("check")
                    

                    


                # print(box)
                # print([prediction.names[int(elem)] for elem in object_class])




TestDetection().test()
