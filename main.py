import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import time
from sort import *
import requests

# Load video
cap = cv2.VideoCapture("D:/Mini_Project/Videos/cars.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Define class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Sinric Pro Credentials
SINRIC_PRO_URL = "https://api.sinric.pro/api/v1/devices/{device_id}/state"
APP_KEY = "a535c0c2-f7ea-43c7-9c0b-9138bbc1be03"
APP_SECRET = "2ea2add5-ef26-4138-99af-3db145f7f2bb-26e4d9e2-886e-4fb8-bb65-7d89f0661c8f"
DEVICE_ID = "67c5dde6c8ff966556a1d8e9"

def send_signal_to_esp8266(state="On"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APP_SECRET}",
    }
    data = {
        "state": {
            "powerState": state.capitalize()  # Sinric expects "On" / "Off"
        }
    }

    try:
        response = requests.put(SINRIC_PRO_URL.format(device_id=DEVICE_ID), json=data, headers=headers, timeout=5)
        response_json = response.json()

        if response.status_code == 200 and response_json.get("success"):
            print("✅ Signal sent successfully!")
        else:
            print(f"❌ Failed to send signal: {response.status_code} - {response_json}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending signal: {e}")

# Load mask
mask = cv2.imread(r"D:\Mini_Project\Images\mask_cars.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []

if mask is None:
    print("Error: Mask image not found or unreadable.")
    exit()

prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Error: Could not read frame.")
        break

    new_frame_time = time.time()

    # Resize mask to match frame size
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Convert mask to 3 channels if grayscale
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    height, width, _ = img.shape
    y_offset = 113
    x_offset = 57
    y_line = min(height - 1, (height // 2) + y_offset)
    line_length = 300
    x_start = max(0, (width // 2) - (line_length // 2) + x_offset)
    x_end = min(width, x_start + line_length)
    cv2.line(img, (x_start, y_line), (x_end, y_line), (0, 0, 255), 3)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=0.5, thickness=1, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count Logic
        if x_start < cx < x_end and y_line - 15 < cy < y_line + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                if len(totalCount) >= 1:
                    print("⚠️ Threshold crossed! Sending signal...")
                    send_signal_to_esp8266("On")
                cv2.line(img, (x_start, y_line), (x_end, y_line), (0, 255, 0), 3)

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
