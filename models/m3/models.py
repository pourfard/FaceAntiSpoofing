import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


from torch.nn import Parameter

BN = nn.BatchNorm2d



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




# AENet_C,S,G is based on ResNet-18
class AENet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, sync_stats=False):
        
        global BN


        self.inplanes = 64
        super(AENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Three classifiers of semantic informantion
        self.fc_live_attribute = nn.Linear(512 * block.expansion, 40)
        self.fc_attack = nn.Linear(512 * block.expansion, 11)
        self.fc_light = nn.Linear(512 * block.expansion, 5)
        # One classifier of Live/Spoof information
        self.fc_live = nn.Linear(512 * block.expansion, 2)

    
        # Two embedding modules of geometric information
        self.upsample14 = nn.Upsample((14, 14), mode='bilinear')
        self.depth_final = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1,bias=False)
        self.reflect_final = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1,bias=False)
        # The ground truth of depth map and reflection map has been normalized[torchvision.transforms.ToTensor()]
        self.sigmoid = nn.Sigmoid()




        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        depth_map = self.depth_final(x)
        reflect_map = self.reflect_final(x)

        depth_map = self.sigmoid(depth_map)
        depth_map = self.upsample14(depth_map)


        reflect_map = self.sigmoid(reflect_map)
        reflect_map = self.upsample14(reflect_map)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        x_live_attribute = self.fc_live_attribute(x)
        x_attack = self.fc_attack(x)
        x_light = self.fc_light(x)
        x_live = self.fc_live(x)




        return x_live




if __name__=="__main__":
    model = AENet()
    model.eval()
    print('wewew')


