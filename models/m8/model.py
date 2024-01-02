from models.FaceAntiSpoofing import FaceAntiSpoofingInterface
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
# import torch.sigmoid as sigmoid
import cv2 as cv
import torch

from torchvision import transforms

import cv2
import numpy as  np

class VehicleClassifier_CV():
    def __init__(self, cfg, weight, names, size, use_gpu=False):
        self.weight = weight
        self.cfg = cfg
        self.names = names
        self.net = cv2.dnn.readNetFromDarknet(self.cfg, self.weight)
        self.size = size
        if use_gpu:
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    def classify(self,bgr):
        bgr = get_fixed(bgr)
        #cv2.imshow("test22", bgr)
        #cv2.waitKey(1)
        try:
            blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, self.size, swapRB=True, crop=False)
        except Exception as e:
            print(e)
            return []

        # Run a model
        self.net.setInput(blob)
        out = self.net.forward()

        # Get a class with a highest score.
        out = out.flatten()
        classId = np.argmax(out)
        confidence = out[classId]

        # Put efficiency information.
        #t, _ = self.net.getPerfProfile()

        return {'label':self.names[classId],'conf':float(confidence), "class_id": int(classId)}

def get_fixed(img):
    if img.shape[0] > img.shape[1]:
        new_img = np.zeros((img.shape[0], img.shape[0], 3), dtype="uint8")
        movement = (img.shape[0] - img.shape[1]) // 2
        new_img[:, movement: movement + img.shape[1], :] = img
    elif img.shape[1] > img.shape[0]:
        new_img = np.zeros((img.shape[1], img.shape[1], 3), dtype="uint8")
        movement = (img.shape[1] - img.shape[0]) // 2
        new_img[movement: movement + img.shape[0], :, :] = img
    else:
        new_img = img

    return new_img
def get_classifier(model_dir, size):
    CFC = model_dir + "/c"
    CFN = model_dir + "/n"
    CFW = model_dir + "/w"

    with open(CFN) as file:
        names = file.readlines()
        names = [name.strip() for name in names]

    modelConfiguration = CFC
    modelWeights = CFW

    return VehicleClassifier_CV(modelConfiguration, modelWeights, names, size)

def get_bbox_with_margin(box, margin_w_left, margin_w_right, margin_h, bgr):
    crop_width = int(box[2] - box[0])
    crop_height = int(box[3] - box[1])
    crop_x = int(box[0])
    crop_y = int(box[1])

    crop_x -= int(crop_width * margin_w_left)
    crop_y -= int(crop_height * margin_h)
    crop_width = int(crop_width + crop_width * (margin_w_left + margin_w_right))
    crop_height = int(crop_height + crop_height * margin_h * 2)

    if crop_x < 0:
        crop_x = 0
    if crop_y < 0:
        crop_y = 0
    if crop_width > bgr.shape[1]:
        crop_width = bgr.shape[1]

    if crop_height > bgr.shape[0]:
        crop_height = bgr.shape[0]

    return [crop_x, crop_y, crop_x + crop_width, crop_y + crop_height]

class M8FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self):
        self.model = get_classifier("models/m8/files/model", (256, 256))

    def get_real_score(self, bgr, bbox, is_crop = False):
        bbox = get_bbox_with_margin(bbox, 1.5,1.5,1.5, bgr)
        x, y, x1, y1 = bbox
        bgr = bgr[y:y1, x:x1] if is_crop is False else bgr
        result = self.model.classify(bgr)
        if result["label"] == "real":
            return result["conf"]
        else:
            return 1 - result["conf"]
