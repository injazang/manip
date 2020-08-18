import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_dct as dct
from models.DCTNet_prev import dct2d_Conv_Layer
from module.modules import augment, Mish

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)

class Preprocess(nn.Module):

    def __init__(self, inplanes, planes,  groups=1):
        super(Preprocess, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


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
    def __init__(self,scale, num_labels, jpeg=False,load=False, groups=True, finetune=False ):
        super(SRNet, self).__init__()
        self.scale = scale
        self.num_labels = num_labels
        self.load = load
        self.finetune = finetune
        self.augment = augment()
        self.groups = groups
        if self.scale==4:
            self.inplanes = 48
            self.dct = dct2d_Conv_Layer(scale=4, start=0, num_filters=48)
            if groups:
                group = [4, 1, 1]
            else:
                group = [1,1,1]
            self.layer1 = self._make_layer(BlockType1, [48, 96, 32], group)
            if groups:
                group = [1,1,1,1,1,1]
            else:
                group = [1,1,1,1,1,1]
            self.layer2 = self._make_layer(BlockType2, [32, 32, 32, 32, 32,32], groups=group)
            if groups:
                group = [1,1,1]
            else:
                group = [1,1,1]
            self.layer3 = self._make_layer(BlockType3, [32, 64, 128], groups=group)
            self.inplanes=3
            self.layer1_1 = self._make_layer(BlockType1, [64, 16], groups=[1,1])
            self.layer2_1 = self._make_layer(BlockType2, [16, 16, 16, 16, 16], groups=[1,1,1,1,1])
            self.layer3_1 = self._make_layer(BlockType3, [16, 64, 128, 256], groups=[1,1,1,1])
            self.inplanes=384
            self.layer4 = self._make_layer(BlockType4, [512], groups=[1])

        elif self.scale==8:
            self.inplanes = 192
            self.name = 'srdct'
            self.dct = dct2d_Conv_Layer(scale=8, start=0, num_filters=192)
            self.layer1 = self._make_layer(BlockType1, [384, 64], groups=1)
            self.layer2 = self._make_layer(BlockType2, [64,64,64,64,64], groups=1)
            if not self.load:
                self.layer3 = self._make_layer(BlockType3, [64, 128, 256], groups=1)
                self.layer4 = self._make_layer(BlockType4, [512, ], groups=1)
        elif self.scale==16:

            self.inplanes = 768
            self.dct = dct2d_Conv_Layer(scale=16, start=0, num_filters=768)
            self.layer1 = self._make_layer(BlockType1, [1024,128], groups=1)
            self.layer2 = self._make_layer(BlockType2, [128, 128, 128, 128, 128], groups=1)
            self.layer3 = self._make_layer(BlockType3, [128, 256], groups=1)
            self.layer4 = self._make_layer(BlockType4, [512], groups=1)

        self.gvp = nn.AdaptiveAvgPool2d((1, 1))
        if finetune==True:
            self.fc2 = nn.Linear(512, 1, bias=False)

        else:
            self.fc = nn.Linear(512, self.num_labels, bias=False)

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
                if m.kernel_size==(self.scale,self.scale):
                    continue
                nn.init.kaiming_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0.2)

    def _make_layer(self, block, num_filters, groups):
        layers = []
        for plane, group in zip(num_filters, groups):
            layers.append(block(self.inplanes, plane, groups=group))
            self.inplanes = plane

        return nn.Sequential(*layers)

    def forward(self, input, prob):
        if self.jpeg:
            im_c, im_q = input
            bs, _, w, h, _, _ = im_c.size()
            deq = (im_c * im_q).float()

            # restore the pixels from the DCT coefficients, not rounding and truncation operation
            im_dct = dct.idct_2d(deq, norm='ortho').permute(0, 1, 2, 4, 3, 5) + 128
            img = im_dct.reshape((-1, 3, w * 8, h * 8))[:, 0, ...].view(bs, 1, w * 8, h * 8)
        else:
            img = input
        img = self.augment(img, prob)
        x = self.dct(img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_ = self.layer1_1(img)
        x_ = self.layer2_1(x_)
        x_ = self.layer3_1(x_)
        x = torch.cat([x,x_],dim=1)
        x = nn.functional.normalize(x, p=2)
        x = self.layer4(x)

        if not self.load:
            x = self.gvp(x)  # torch.cat([self.gvp8(x8), self.gvp16(x16)], dim=1)
            x = x.view(x.size(0), -1)
            if self.finetune:
                x = self.fc2(x)

            else:
                x = self.fc(x)

        return x

