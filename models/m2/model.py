from models.FaceAntiSpoofing import FaceAntiSpoofingInterface
import cv2
import numpy as np
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
from models.m2.rPPG.rPPG_Extracter import *
from models.m2.rPPG.rPPG_lukas_Extracter import *


class M2FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self):
        yaml_file = open("models/m2/files/RGB_rPPG_merge_softmax_.yaml", 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.model = model_from_yaml(loaded_model_yaml)
        self.model.load_weights("models/m2/files/RGB_rPPG_merge_softmax_.h5")
        self.dim = (128, 128)

    def get_rppg_pred(self, face_crop):
        use_classifier = True  # Toggles skin classifier
        use_flow = False  # (Mixed_motion only) Toggles PPG detection with Lukas Kanade optical flow
        sub_roi = []  # If instead of skin classifier, forhead estimation should be used set to [.35,.65,.05,.15]
        use_resampling = False  # Set to true with webcam

        fftlength = 300
        fs = 20
        f = np.linspace(0, fs / 2, fftlength // 2 + 1) * 60;

        timestamps = []
        time_start = [0]

        break_ = False

        rPPG_extracter = rPPG_Extracter()
        rPPG_extracter_lukas = rPPG_Lukas_Extracter()
        bpm = 0

        dt = time.time() - time_start[0]
        time_start[0] = time.time()
        if len(timestamps) == 0:
            timestamps.append(0)
        else:
            timestamps.append(timestamps[-1] + dt)

        rPPG = []

        rPPG_extracter.measure_rPPG(face_crop, use_classifier, sub_roi)
        rPPG = np.transpose(rPPG_extracter.rPPG)

        # Extract Pulse
        if rPPG.shape[1] > 10:
            if use_resampling:
                t = np.arange(0, timestamps[-1], 1 / fs)

                rPPG_resampled = np.zeros((3, t.shape[0]))
                for col in [0, 1, 2]:
                    rPPG_resampled[col] = np.interp(t, timestamps, rPPG[col])
                rPPG = rPPG_resampled
            num_frames = rPPG.shape[1]

            t = np.arange(num_frames) / fs
        return rPPG

    def make_pred(self, li):
        [single_img, rppg] = li
        single_img = cv2.resize(single_img, self.dim)
        single_x = img_to_array(single_img)
        single_x = np.expand_dims(single_x, axis=0)
        single_pred = self.model.predict([single_x, rppg])
        return single_pred

    def get_real_score(self, bgr, face_bbox):
        crop = bgr[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2], :]

        rppg_s = self.get_rppg_pred(crop)
        rppg_s = rppg_s.T

        real_score = float(self.make_pred([crop, rppg_s])[0][0])

        return real_score
