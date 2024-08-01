from ultralytics import YOLO

import cv2
import math 
# start webcam
cap = cv2.VideoCapture(-1)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolov8n.pt")
import os
if os.path.exists('yolov8n.onnx'):
    model = YOLO('yolov8n.onnx')
else:
    model.export(format='onnx', imgsz=320, simplify=True, half=True)
    model = YOLO('yolov8n.onnx')
class_names = list(model.names.values())


while True:
    success, img = cap.read()
    if not success:
        break
    results = model(img, imgsz=320, stream=True)
    
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # class name
            cls = int(box.cls[0])
            print("Class name = ", class_names[cls])

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence = ",confidence)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, class_names[cls], org, font, fontScale, color, thickness)
    

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()