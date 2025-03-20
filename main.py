from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List

app = FastAPI()

# 📌 Load your trained YOLOv8 model
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image from request
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 📌 Perform inference with YOLOv8
        results = model(img)

        # 📌 Extract detections
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })

        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}

# Run FastAPI using: uvicorn filename:app --host 0.0.0.0 --port 5001 --reload
