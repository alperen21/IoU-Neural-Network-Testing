from ultralytics import YOLO
import cv2

# Initialize YOLO model
model = YOLO('yolov5s.pt')  # Here 'yolov5s' represents the small variant. You can also use 'yolov5m', 'yolov5l', 'yolov5x' for medium, large, and extra-large variants.

# Load an image
img = cv2.imread("bus.jpg")

# Get predictions
results = model.predict(img)

# Extracting bounding boxes and class labels from the results
boxes_object = results[0].boxes
bounding_boxes = boxes_object.xyxy  # This gives [x1, y1, x2, y2] for each detection
labels = boxes_object.cls  # Use 'cls' for class labels (e.g., 0 for 'person', 5 for 'bus', etc.)
confidence_scores = boxes_object.conf  # This gives confidence scores for each detection

# Mapping the labels to their names
names = results[0].names
class_names = [names[int(i)] for i in labels]

print("Bounding boxes:", bounding_boxes)
print("Predicted classes:", class_names)
print("Confidence scores:", confidence_scores)
