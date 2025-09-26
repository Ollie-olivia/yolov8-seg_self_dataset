from ultralytics import YOLO

# Load an official or custom model
model = YOLO(r"E:\master files\yellow_flower_detection\code\yolov8-seg_for_yellow\best.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model

# Perform tracking with the model
results = model.track(source=r"E:\master files\yellow_flower_detection\code\yolov8-seg_for_yellow\track\shift_rotate_daylily.mp4", show=True)