

from FaceChecker.FaceDetector import FaceDetector
import cv2
import time
import numpy as np
model = FaceDetector('FaceDetector/pretrain/yolov2_tiny-face.h5')

cap = cv2.VideoCapture('test.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#Detection
while True:
    #Face Detection
    ret, frame = cap.read() #BGR

    #frame = cv2.imread("images/dress3.jpg")
    start_time = time.time()
    img=frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    list_boxes = model.detect_faces(img)
    for box in list_boxes:
        x, y, w, h = box['coordinates']
        cv2.rectangle(frame, (x-w, y-h), (x+w, y+h), (0, 255, 0), 4)

    print('detect face: '+str(time.time()-start_time)+' (s)')


    cv2.imshow('image', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()