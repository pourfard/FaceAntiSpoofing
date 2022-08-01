import time

import cv2

from detector.cv_face_detector.model import CVFaceDetector
from models.m1.model import M1FaceAntiSpoofing
from models.m2.model import M2FaceAntiSpoofing
from models.m3.model import M3FaceAntiSpoofing
from models.m4.model import M4FaceAntiSpoofing
from models.m5.model import M5FaceAntiSpoofing

import os

face_detector = CVFaceDetector()
spoof_detector = M5FaceAntiSpoofing()

skip_frame = 3
cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.2.111:554")
count = 0
while True:
    ret, bgr = cap.read()
    if not ret:
        break
    count += 1
    if count % skip_frame != 0:
        continue
    face_bboxes = face_detector.get_face_bboxes(bgr)

    for bbox in face_bboxes:
        real_score = spoof_detector.get_real_score(bgr, bbox)

        print(real_score)
        if real_score > 0.5:
            cv2.rectangle(bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0))
        else:
            cv2.rectangle(bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))

    cv2.imshow("test", bgr)
    cv2.waitKey(1)