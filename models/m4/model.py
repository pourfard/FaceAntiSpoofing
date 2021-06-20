from models.FaceAntiSpoofing import FaceAntiSpoofingInterface
import cv2
import numpy as np
import dlib
from models.m4.m4models import MyresNet34
import torch

class M4FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self):
        self.shape_predictor = dlib.shape_predictor("models/m4/files/dlib_landmarks.dat")
        MODEL_PATH = "models/m4/files/5.pth"
        self.model = MyresNet34().eval()
        self.model.load(MODEL_PATH)
        self.model.train(False)
        self.scale = 3.5
        self.image_size = 224

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def crop_with_ldmk(self, image, landmark):
        ct_x, std_x = landmark[:, 0].mean(), landmark[:, 0].std()
        ct_y, std_y = landmark[:, 1].mean(), landmark[:, 1].std()

        std_x, std_y = self.scale * std_x, self.scale * std_y

        src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
        dst = np.float32([((self.image_size - 1) / 2.0, (self.image_size - 1) / 2.0),
                          ((self.image_size - 1), (self.image_size - 1)),
                          ((self.image_size - 1), (self.image_size - 1) / 2.0)])
        retval = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image, retval, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        return result

    def get_real_score(self, bgr, face_bbox):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rect = dlib.rectangle(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3])
        shape = self.shape_predictor(rgb, rect)
        shape_np = []
        shape_np.append(self.shape_to_np(shape))

        ldmk = np.asarray(shape_np, dtype=np.float32)
        ldmk = ldmk[np.argsort(np.std(ldmk[:, :, 1], axis=1))[-1]]
        result = self.crop_with_ldmk(bgr, ldmk)
        data = np.transpose(np.array(result, dtype=np.float32), (2, 0, 1))

        data = data[np.newaxis, :]
        data = torch.FloatTensor(data)
        with torch.no_grad():
            outputs = self.model(data)
            outputs = torch.softmax(outputs, dim=-1)
            preds = outputs.to('cpu').numpy()
            attack_prob = preds[:, 0]  # 0 attack 1 genuine

        return 1 - float(attack_prob[0])
