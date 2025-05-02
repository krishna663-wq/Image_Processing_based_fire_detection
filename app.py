import cv2
import numpy as np
import time
import os

v = cv2.VideoCapture(0)
time.sleep(2)

fdc = cv2.CascadeClassifier('fire_detection.xml')  # <- Update path here

while True:
    d, i = v.read()
    print(i.shape)

    fire = fdc.detectMultiScale(i, 1.3, 9)
    print(fire)

    if len(fire) >= 1:
        for x, y, w, h in fire:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(i, 'FIRE DETECTED', (x - w, y - h), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(i, (x, y), (x + w, y + h), (0, 0, 255), 2)
            time.sleep(0.2)
            os.system("say 'Fire detected'")  # <- Mac sound alert

        print('FIRE DETECTED')

    cv2.imshow('image', i)
    k = cv2.waitKey(5)
    if k == ord('q'):
        cv2.destroyAllWindows()
        break





