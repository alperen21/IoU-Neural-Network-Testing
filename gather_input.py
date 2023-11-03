import cv2 
import os
from config import config
from ultralytics import YOLO

class Object:
    """
    Object that are to be detected by the model:
        - Image class : int
        - Bounding box: list of floats (normalized coordinates) 
    """
    def __init__(self, object_class : int, bounding_box : list[float]):
        self.object_class = object_class
        self.bounding_box = bounding_box
    
    def __repr__(self) -> str:
        return f"""
        object class : {self.object_class}
        bounding_box: {self.bounding_box}
        """


class ImageObject:
    """
    This class encapsulates the following information:
        - img url : string
        - objects : list of Object
    """
    def __init__(self, img_url : str, objects : list[Object]):
        self.img_url = img_url
        self.objects = objects
    
    def __repr__(self) -> str:
        return f"""
        img_url : {self.img_url}
        objects : {self.objects}
        """

class ModelTuple():
    def __init__(self, model : YOLO, model_weight_file : str):
        self.model = model
        self.model_weight_file = model_weight_file
    
    def __repr__(self) -> str:
        return f"<model tuple with: {self.model_weight_file}>"


def isImage(path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return path.lower().endswith(valid_extensions)

def gather_images():
    img_objects = []
    image_urls = []
    box_files = []

    for root, _, files in os.walk(os.path.join("input", "images")):
        for file in files:
            if isImage(file):
                image_urls.append(os.path.join(root, file))
    
    for root, _, files in os.walk(os.path.join("input", "boxes")):
        for file in files:
            if file.lower().endswith("txt"):
                box_files.append(os.path.join(root, file))
    
    image_urls = sorted(image_urls)
    box_files = sorted(box_files)

    for image_url, box_file in zip(image_urls, box_files):

        objects = list()
        with open(box_file, "r") as file:
            for line in file.readlines():
                line = line.strip().split(config.delimiter)
                object_class = line[0]
                bounding_box = [float(elem) for elem in line[1:]]

                object = Object(
                    object_class = object_class,
                    bounding_box = bounding_box
                )

                objects.append(object)
    
        image_object = ImageObject(
            img_url = image_url,
            objects = objects
        )

        

        img_objects.append(image_object)

    return img_objects


def gather_models():
    model_tuples = list()
    weights = list()
    models = list()

    for root, _, files in os.walk("models"):
        for file in files:
            weights.append(os.path.join(root, file))
    
    for weight in weights:
        models.append(YOLO(weight))

    for weight, model in zip(weights, models):
        modelTuple = ModelTuple(
            model_weight_file = weight,
            model = model
        )

        model_tuples.append(modelTuple)

    return model_tuples
    
def main():
    print(gather_images())
    print(gather_models())


if __name__ == "__main__":
    main()

    

    
