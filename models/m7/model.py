from models.FaceAntiSpoofing import FaceAntiSpoofingInterface
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
# import torch.sigmoid as sigmoid
import cv2 as cv
import torch

from torchvision import transforms


class DeePixBiS(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        dense = models.densenet161(pretrained=pretrained)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = F.sigmoid(dec)
        # print(out_map.shape)
        out = self.linear(out_map.view(-1, 14 * 14))
        out = F.sigmoid(out)
        out = torch.flatten(out)
        return out_map, out

class M7FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self):
        self.model = DeePixBiS(False)
        self.model.load_state_dict(torch.load('models/m7/files/DeePixBiS.pth'))
        self.model.eval()

        self.tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_real_score(self, bgr, bbox, is_crop = False):
        x, y, x1, y1 = bbox
        faceRegion = bgr[y:y1, x:x1] if is_crop is False else bgr
        faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
        faceRegion = self.tfms(faceRegion)
        faceRegion = faceRegion.unsqueeze(0)

        mask, binary = self.model.forward(faceRegion)
        res = torch.mean(mask).item()

        return res
