# ---------------------------- YOLOv8 Evaluation Script ----------------------------
import os
import cv2
import glob
from ultralytics import YOLO
from utils_yolo import read_gt_labels, compute_iou

model = YOLO("runs/detect/yolov8m_b16/weights/best.pt")

test_image_dir = "data_yolo/test/images"
test_label_dir = "data_yolo/test/labels"
image_paths = sorted(glob.glob(os.path.join(test_image_dir, "*.jpg")))

TP, FP, FN = 0, 0, 0

for image_path in image_paths:
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    label_path = os.path.join(test_label_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
    gt_boxes = read_gt_labels(label_path, img_width, img_height)
    results = model(image_path, iou=0.3, conf=0.1)

    pred_boxes = []
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            pred_boxes.append((int(x1), int(y1), int(x2), int(y2), int(cls), float(conf)))

    matched_gt = set()
    for pb in pred_boxes:
        px1, py1, px2, py2, pcls, pconf = pb
        best_iou, best_gt_idx = 0, -1

        for i, gb in enumerate(gt_boxes):
            gx1, gy1, gx2, gy2, gcls = gb
            if pcls == gcls:
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, i

        if best_iou > 0.3 and best_gt_idx not in matched_gt:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    FN += len(gt_boxes) - len(matched_gt)

accuracy = TP / (TP + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
