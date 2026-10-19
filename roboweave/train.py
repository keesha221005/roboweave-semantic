"""Fine-tune YOLOv8 on the simulator dataset.

    python -m roboweave.train                 # 25 epochs, yolov8n
    python -m roboweave.train --epochs 40
Prints the path of the best weights when done.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, workers=2,
                name="roboweave", exist_ok=True, plots=False)
    print("\nBEST WEIGHTS:", model.trainer.best)


if __name__ == "__main__":
    main()
