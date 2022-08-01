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
spoof_detectors = [M4FaceAntiSpoofing()]

benchmark_dir = "benchmarks"
for spoof_detector in spoof_detectors:
    print("Start ----------------------------- ", type(spoof_detector))
    total_time = 0
    all_count = 0
    correct_count = 0
    errors = []
    for class_name in ["fake", "real"]:
        class_path = os.path.join(benchmark_dir, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            bgr = cv2.imread(image_path)

            face_bboxes = face_detector.get_face_bboxes(bgr)

            for bbox in face_bboxes:
                start_time = time.time()
                real_score = spoof_detector.get_real_score(bgr, bbox)
                total_time += time.time() - start_time
                print("Real score for image name " + image_name + " in class " + class_name + " is: ",
                      real_score)

                if class_name == "fake":
                    if real_score < 0.5:
                        is_correct = True
                    else:
                        cv2.imshow("fake", bgr)
                        cv2.waitKey(0)
                        is_correct = False
                    errors.append(real_score)

                else:
                    if real_score >= 0.5:
                        is_correct = True
                    else:
                        cv2.imshow("test", bgr)
                        cv2.waitKey(0)
                        is_correct = False
                    errors.append(1 - real_score)

                if is_correct:
                    correct_count+=1

                all_count+=1
                print("Correct prediction: ", is_correct)

    print("--- Average time for each face: ", total_time/all_count, " seconds")
    print("--- Total count: ", all_count)
    print("--- Correct count: ", correct_count)
    print("--- Average error: ", (sum(errors)/len(errors))*100,"%")
    print("End ----------------------------- ", type(spoof_detector))
