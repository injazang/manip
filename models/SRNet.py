import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_dct as dct
from module.modules import augment

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


class BlockType1(nn.Module):

    def __init__(self, inplanes, planes, groups=1):
        super(BlockType1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            #                nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            #                nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            #                nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            #                nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
    def forward(self, x):
        out = self.type1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        #out = self.gvp(out)

        return out


class SRNet(nn.Module):
    def __init__(self, num_labels, load=False):
        super(SRNet, self).__init__()
        self.load=load
        self.num_labels= num_labels
        self.inplanes = 3
        self.name = 'srnet'
        self.augment = augment()
        self.layer1 = self._make_layer(BlockType1, [64, 16], groups=1)
        self.layer2 = self._make_layer(BlockType2, [16, 16, 16, 16, 16], groups=1)
        if not load:
            self.layer3 = self._make_layer(BlockType3, [16, 64, 128, 256], groups=1)
            self.layer4 = self._make_layer(BlockType4, [512, ], groups=1)
            self.gvp = nn.AdaptiveAvgPool2d((1, 1))

            self.fc = nn.Linear(512, num_labels, bias=False)
            self.sc = nn.Softmax()
        self.conv_weights = []
        self.non_conv_weights = []

        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.conv_weights.append(param)
            else:
                self.non_conv_weights.append(param)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)
#                nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _make_layer(self, block, num_filters, groups=1):
        layers = []
        for planes in num_filters:
            layers.append(block(self.inplanes, planes, groups=groups))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input, prob):
        img = input
        img = self.augment(img, prob)
        x = self.layer1(img)
        x = self.layer2(x)
        if not self.load:
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.gvp(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x