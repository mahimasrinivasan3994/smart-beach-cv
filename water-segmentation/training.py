from ultralytics import YOLO
import numpy as np

if __name__ == '__main__':
    YOLO('base/yolov8s-seg.pt').train(
    data='data.yaml',
    imgsz=1280,
    epochs=100,
    batch=10,
    patience=100
)