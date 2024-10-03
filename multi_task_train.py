import torch

from ultralytics.models.yolo.multi import MultiTaskTrainer

args = dict(
    model="ultralytics/cfg/models/v8/yolov8m-multi.yaml",
    data="ultralytics/cfg/datasets/green-onion-multi.yaml",
    project="multi-task-yolov8",
    epochs=2,
    imgsz=640,
    name="multi-task",
    workers=8,
    batch=8,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
)

trainer = MultiTaskTrainer(overrides=args)

trainer.train()
