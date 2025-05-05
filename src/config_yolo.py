# ---------------------------- YOLOv8 Configuration ----------------------------
TRAIN_CONFIG = {
    "pretrained_model": "yolov8m.pt",
    "train_args": {
        "data": "config_yolo2.yaml",
        "epochs": 100,
        "imgsz": 320,
        "batch": 16,
        "lr0": 0.0001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.001,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "scale": 0.5,
        "translate": 0.1,
        "mosaic": 0.5,
        "cos_lr": False,
        "resume": False,
        "val": True,
        "device": 0
    }
}
