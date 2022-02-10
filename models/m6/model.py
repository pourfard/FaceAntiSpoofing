from models.FaceAntiSpoofing import FaceAntiSpoofingInterface
import torch
import os
from models.m6.src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from models.m6.src.data_io import transform as trans
from models.m6.src.utility import get_kernel, parse_model_name
import torch.nn.functional as F
from models.m6.src.generate_patches import CropImage
import numpy as np

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class AntiSpoofPredict():
    def __init__(self, device_id, model_path):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")

        self._load_model("models/m6/files/" + model_path)
        self.model.eval()
        self.name = model_path

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input, )
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result


class M6FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self):
        self.image_cropper = CropImage()
        self.m1 = AntiSpoofPredict(0, "2.7_80x80_MiniFASNetV2.pth")
        self.m2 = AntiSpoofPredict(0, "4_0_0_80x80_MiniFASNetV1SE.pth")

    def predict_one(self, bgr, bbox, model):
        h_input, w_input, model_type, scale = parse_model_name(model.name)
        param = {
            "org_img": bgr,
            "bbox": bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = self.image_cropper.crop(**param)
        return model.predict(img)

    def get_real_score(self, bgr, face_bbox):
        bbox = [face_bbox[0], face_bbox[1], face_bbox[2] - face_bbox[0] + 1, face_bbox[3] - face_bbox[1] + 1]
        prediction = np.zeros((1, 3))

        prediction += self.predict_one(bgr, bbox, self.m1)
        prediction += self.predict_one(bgr, bbox, self.m2)

        return prediction[0][1] / 2
