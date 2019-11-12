import os
import os.path as osp
import sys
import cv2
import datetime

dst = './recording'
if not osp.exists(dst):
    os.mkdir(dst)

cap = cv2.VideoCapture(1)

ret, frame = cap.read()
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Detection Results", frame)
    time = datetime.datetime.now().strftime("CAM1-%Y-%m-%d_%H-%M-%S")    
    fname = osp.join(dst, "{}.jpg".format(time))
    cv2.imwrite(fname, frame)

    key = cv2.waitKey(1)
    if key == 27:
        break