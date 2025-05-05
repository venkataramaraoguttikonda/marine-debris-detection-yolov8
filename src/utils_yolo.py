# ---------------------------- Utility Functions ----------------------------
import os
import cv2

def read_gt_labels(label_path, img_width, img_height):
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
    with open(label_path, "r") as f:
        for line in f:
            cls, x_c, y_c, w, h = map(float, line.strip().split()[:5])
            x1 = int((x_c - w / 2) * img_width)
            y1 = int((y_c - h / 2) * img_height)
            x2 = int((x_c + w / 2) * img_width)
            y2 = int((y_c + h / 2) * img_height)
            gt_boxes.append((x1, y1, x2, y2, int(cls)))
    return gt_boxes

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def visualize_boxes(image, gt_boxes, pred_boxes):
    for (x1, y1, x2, y2, cls) in gt_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green for GT
        cv2.putText(image, f"GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    for (x1, y1, x2, y2, cls, conf) in pred_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for predictions
        cv2.putText(image, f"Pred ({conf:.2f})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return image