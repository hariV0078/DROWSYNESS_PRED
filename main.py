from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from ultralytics import YOLO  # Import YOLOv8 library

app = Flask(__name__)

# 📌 Load your trained YOLOv8 model (best.pt)
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read image from request
        file = request.files['image'].read()
        np_img = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 📌 Perform inference with YOLOv8
        results = model(img)

        # 📌 Extract detections (bounding boxes, class names, confidence)
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

        return jsonify(detections)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
