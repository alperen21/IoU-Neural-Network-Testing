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

import cv2
import os

def draw_predictions(img, boxes, class_names, confidences):
    for box, label, conf in zip(boxes, class_names, confidences):
        # Extract coordinates
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display label and confidence score
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img

# Draw predictions on the image
img_with_boxes = draw_predictions(results[0].orig_img, bounding_boxes, class_names, confidence_scores)

# Specify output folder and save the image
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_path = os.path.join(output_folder, 'result_image.jpg')
cv2.imwrite(output_path, img_with_boxes)
print(f"Image saved to {output_path}")

