# ---------------------------- YOLOv8 Visualization Script ----------------------------
import os
import cv2
import glob
from ultralytics import YOLO
from utils_yolo import read_gt_labels, visualize_boxes

# Load trained model
model = YOLO("runs/detect/yolov8m_b16/weights/best.pt")

# Define paths
test_image_dir = "data_yolo/test/images"
test_label_dir = "data_yolo/test/labels"
save_dir = "vis_outputs"
os.makedirs(save_dir, exist_ok=True)

# Get image paths
image_paths = sorted(glob.glob(os.path.join(test_image_dir, "*.jpg")))

# Iterate through images
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Get ground truth
    label_path = os.path.join(test_label_dir, image_name.replace(".jpg", ".txt"))
    gt_boxes = read_gt_labels(label_path, img_width, img_height)

    # Get predictions
    results = model(image_path, iou=0.3, conf=0.1)
    pred_boxes = []
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            pred_boxes.append((int(x1), int(y1), int(x2), int(y2), int(cls), float(conf)))

    # Draw boxes
    vis_img = visualize_boxes(img.copy(), gt_boxes, pred_boxes)

    # Save image
    save_path = os.path.join(save_dir, image_name)
    cv2.imwrite(save_path, vis_img)

print(f"Visualized outputs saved to: {save_dir}")
