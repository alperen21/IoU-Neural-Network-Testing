from ultralytics import YOLO
import os
from config import model_weights


def get_model_tuples(model_weights):
    return [(weights, YOLO(os.path.join(".","models",weights))) for weights in model_weights]