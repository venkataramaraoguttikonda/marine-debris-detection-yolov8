# ---------------------------- YOLOv8 Training Script ----------------------------
from ultralytics import YOLO
from config_yolo import TRAIN_CONFIG

def train():
    model = YOLO(TRAIN_CONFIG["pretrained_model"])
    model.train(**TRAIN_CONFIG["train_args"])

if __name__ == "__main__":
    train()