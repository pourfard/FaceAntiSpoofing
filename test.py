import time

import cv2

from detector.insight_detector.model import InsightDetector
from models.m6.model import M6FaceAntiSpoofing
from models.m8.model import M8FaceAntiSpoofing
from models.m9.model import M9FaceAntiSpoofing

import os

face_detector = InsightDetector()
spoof_detectors = [M6FaceAntiSpoofing(), M8FaceAntiSpoofing(),M9FaceAntiSpoofing()]
benchmark_dir = "benchmarks"
is_face = False
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

            if bgr is None:
                continue
            if not is_face:
                face_bboxes = face_detector.get_face_bboxes(bgr)
            else:
                face_bboxes = [[0, 0, bgr.shape[1], bgr.shape[0]]]

            if len(face_bboxes) == 0:
                print("No face found " + image_name + " in class " + class_name)
            for box in face_bboxes:
                bbox = box["box"]
                start_time = time.time()
                crop = bgr[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                if "crop" in box:
                    crop = box["crop"]
                #cv2.imshow("crop", crop)
                #cv2.waitKey(0)
                real_score = spoof_detector.get_real_score(bgr, bbox)

                total_time += time.time() - start_time
                print("Real score for image name " + image_name + " in class " + class_name + " is: ",
                      real_score)

                if class_name == "fake":
                    if real_score < 0.5:
                        is_correct = True
                    else:
                        is_correct = False
                    errors.append(real_score)

                else:
                    if real_score >= 0.5:
                        is_correct = True
                    else:
                        is_correct = False
                    errors.append(1 - real_score)

                if is_correct:
                    correct_count += 1

                all_count += 1
                print("Correct prediction: ", is_correct)

    print("--- Average time for each face: ", total_time / all_count, " seconds")
    print("--- Total count: ", all_count)
    print("--- Correct count: ", correct_count)
    print("--- Accuracy: ", correct_count/all_count)
    print("--- Average error: ", (sum(errors) / len(errors)) * 100, "%")
    print("End ----------------------------- ", type(spoof_detector))
