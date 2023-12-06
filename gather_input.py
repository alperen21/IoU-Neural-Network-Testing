import cv2 
import os
from config import config
from ultralytics import YOLO
import pathlib

class Object:
    """
    Object that are to be detected by the model:
        - Image class : int
        - Bounding box: list of floats (normalized coordinates) 
    """
    def __init__(self, object_class : int, bounding_box : list[float], correct_classification = False, passed_iou_threshold = False):
        self.object_class = object_class
        self.bounding_box = bounding_box
        self.correct_classification = correct_classification
        self.passed_iou_threshold = passed_iou_threshold
        self.detected = False
    
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

def delete_files_in_directory(directory_path):
    """
    Deletes all files in the specified directory.

    Args:
    directory_path (str): The path to the directory from which to delete files.
    """
    # Convert the directory path to a pathlib Path object for easy manipulation
    dir_path = pathlib.Path(directory_path)

    # Check if the directory exists
    if not dir_path.is_dir():
        print(f"The directory {directory_path} does not exist.")
        return

    # Iterate over each item in the directory
    for item in dir_path.iterdir():
        # Check if the item is a file and delete it
        if item.is_file():
            try:
                item.unlink()  # unlink is a method to delete files
                print(f"Deleted file: {item}")
            except Exception as e:
                print(f"Error deleting file {item}: {e}")

def delete_specific_files(directory_path, prefix):
    """
    Deletes all files with a specific prefix in the specified directory and its subdirectories.

    Args:
    directory_path (str): The path to the directory to search.
    prefix (str): The prefix of the files to be deleted.
    """
    # Convert the directory path to a pathlib Path object for easy manipulation
    dir_path = pathlib.Path(directory_path)

    # Check if the directory exists
    if not dir_path.is_dir():
        print(f"The directory {directory_path} does not exist.")
        return

    # Walk through all directories and files
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.startswith(prefix):
                file_path = pathlib.Path(root) / file
                try:
                    file_path.unlink()  # Delete the file
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")


def clean():
    print("cleaning up...")
    delete_files_in_directory("output")
    delete_specific_files('.', '._')


def gather_models():
    model_tuples = list()
    weights = list()
    models = list()

    for root, _, files in os.walk("models"):
        for file in files:
            if file.endswith(".pt"):
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
    clean()


if __name__ == "__main__":
    main()

    

    
