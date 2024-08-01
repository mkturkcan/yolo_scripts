from ultralytics import YOLO
from ultralytics import YOLOWorld

import cv2
import math 
import onnxruntime as ort
import numpy as np
x = np.zeros((1,128,128,3))
ort_sess = ort.InferenceSession("depth_large.onnx")
cap = cv2.VideoCapture(0)
cap.set(3, 128)
cap.set(4, 128)

while True:
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    
    x[0,:,:,:] = img / 255.
    outputs = ort_sess.run(None, {'x': x.astype(np.float32)})
    out = np.ascontiguousarray(outputs[0][0])
    # out = out / np.max(out)
    #out = np.clip(out, 0.1, np.log(300.))
    #out = np.exp(out) / 300.
    #print(np.max(out))
    out = out / np.log(20.)
    print(np.max(out))
    cv2.imshow('Webcam', out)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()