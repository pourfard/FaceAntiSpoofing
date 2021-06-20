from keras.models import load_model
from models.FaceAntiSpoofing import FaceAntiSpoofingInterface
import cv2
import numpy as np


class M1FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self):
        self.model = load_model("models/m1/files/fas.h5")

    def get_real_score(self, bgr, face_bbox):
        crop = bgr[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2], :]
        crop = (cv2.resize(crop, (224, 224)) - 127.5) / 127.5
        real_score = float(self.model.predict(np.array([crop]))[0][0])
        return real_score
