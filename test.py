
from ultralytics import YOLO
import cv2
import os

model_path = "runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

import glob
import matplotlib.pyplot as plt
import cv2

# Path to the folder containing test images
image_paths = glob.glob('images/train/IMG-20240921-WA0149.jpg')

# Run inference on each image and count the bounding boxes
for img_path in image_paths:
    # Read the image
    img = cv2.imread(img_path)

    # Run inference on the image
    results = model(img_path)

    # Count the number of vehicles (bounding boxes)
    num_boxes = len(results[0].boxes)
    print(f"Image: {img_path}, Number of vehicles detected: {num_boxes}")