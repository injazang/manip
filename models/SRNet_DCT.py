import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_dct as dct
from models.DCTNet_prev import dct2d_Conv_Layer
from module.mish import  Mish

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=True)


class BlockType1(nn.Module):

    def __init__(self, inplanes, planes, groups=1):
        super(BlockType1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class BlockType2(nn.Module):

    def __init__(self, inplanes, planes, groups=1):
        super(BlockType2, self).__init__()
        self.type1 = BlockType1(inplanes, planes, groups=groups)
        self.conv1 = conv3x3(planes, planes, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x

        out = self.type1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        out += identity
        return out


class BlockType3(nn.Module):

    def __init__(self, inplanes, planes, groups=1):
        super(BlockType3, self).__init__()
        self.conv0 = conv1x1(inplanes, planes, stride=2, groups=groups)
        self.bn0 = nn.BatchNorm2d(planes)

        self.type1 = BlockType1(inplanes, planes, groups=groups)
        self.conv1 = conv3x3(planes, planes, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(3, 2, padding=1)

    def forward(self, x):
        identity = self.bn0(self.conv0(x))

        out = self.type1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.avgpool(out)

        out += identity
        return out


class BlockType4(nn.Module):

    def __init__(self, inplanes, planes, groups=1):
        super(BlockType4, self).__init__()
        self.type1 = BlockType1(inplanes, planes, groups=groups)
        self.conv1 = conv3x3(planes, planes, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.type1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        #out = self.gvp(out)

        return out


class SRNet(nn.Module):
    def __init__(self, jpeg=False):
        super(SRNet, self).__init__()

        self.inplanes = 192
        self.name = 'srdct'
        self.dct8 = dct2d_Conv_Layer(scale=8, start=0, num_filters=192)
        self.layer1_8 = self._make_layer(BlockType1, [384, 64], groups=1)
        self.layer2_8 = self._make_layer(BlockType2, [64,64,64,64,64], groups=1)
        self.layer3_8 = self._make_layer(BlockType3, [64, 128, 256], groups=1)
        self.layer4_8 = self._make_layer(BlockType4, [512, ], groups=1)
        self.inplanes = 192
        #self.dct16 = dct2d_Conv_Layer(scale=16, start=0, num_filters=192)
        #self.layer1_16 = self._make_layer(BlockType1, [384, 64], groups=1)
        #self.layer2_16 = self._make_layer(BlockType2, [64, 64, 64, 64, 64], groups=1)
        #self.layer3_16 = self._make_layer(BlockType3, [64, 128], groups=1)
        #self.layer4_16 = self._make_layer(BlockType4, [256, ], groups=1)

        self.gvp8 = nn.AdaptiveAvgPool2d((1, 1))
        self.gvp16 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 5, bias=False)
        self.sc = nn.Softmax()
        self.conv_weights = []
        self.non_conv_weights = []
        self.jpeg=jpeg

        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.conv_weights.append(param)
            else:
                self.non_conv_weights.append(param)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _make_layer(self, block, num_filters, groups=1):
        layers = []
        for planes in num_filters:
            layers.append(block(self.inplanes, planes, groups=groups))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):
        if self.jpeg:
            im_c, im_q = input
            bs, _, w, h, _, _ = im_c.size()
            deq = (im_c * im_q).float()

            # restore the pixels from the DCT coefficients, not rounding and truncation operation
            im_dct = dct.idct_2d(deq, norm='ortho').permute(0, 1, 2, 4, 3, 5) + 128
            img = im_dct.reshape((-1, 3, w * 8, h * 8))[:, 0, ...].view(bs, 1, w * 8, h * 8)
        else:
            img = input
        x = self.dct8(img)
        x = self.layer1_8(x)
        x = self.layer2_8(x)
        x = self.layer3_8(x)
        x8 = self.layer4_8(x)


        #x = self.dct16(img)
        #x = self.layer1_16(x)
        #x = self.layer2_16(x)
        #x = self.layer3_16(x)
        #x16 = self.layer4_16(x)

        x = self.gvp8(x8)#torch.cat([self.gvp8(x8), self.gvp16(x16)], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logit = self.sc(x)
        return logit