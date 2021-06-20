from models.FaceAntiSpoofing import FaceAntiSpoofingInterface
import cv2
import numpy as np
from models.m3.tsn_predict import TSNPredictor as CelebASpoofDetector


class M3FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self):
        self.model = CelebASpoofDetector()

    def get_real_score(self, bgr, face_bbox):
        crop = bgr[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2], :]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        real_score = self.model.predict(np.array([crop_rgb]))[0][0]

        return real_score
