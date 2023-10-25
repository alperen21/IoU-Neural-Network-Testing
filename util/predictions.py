import cv2

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