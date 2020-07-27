from time import struct_time

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_dct as dct
from models.DCTNet_prev import dct2d_Conv_Layer
from models.SRNet_DCT_scale import SRNet as SRNet_scale
from module import modules
def conv3x3(in_planes, out_planes, kernel_size=3, stride=1,padding=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', groups=groups, bias=False)


def conv3x3_de(in_planes, out_planes, stride=1, groups=1, padding=1, output_padding=0):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False, output_padding=output_padding)


def conv1x1(in_planes, out_planes, stride=1, groups=1, padding=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, groups=groups, bias=False)




class SeparableConvBlock(nn.Module):
    def __init__(self, in_planes, kernel_size, stride):
        super(SeparableConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='replicate', bias=False)
        self.bn = nn.BatchNorm2d(in_planes)        # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')

    def forward(self, input):
        input = self.conv(input)
        input = self.bn(input)
        return input

class Preprocess(nn.Module):

    def __init__(self, inplanes, planes, groups=1):
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

    def __init__(self, inplanes, planes, groups=1, padding=1):
        super(BlockType1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, groups=groups, padding=padding)
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

    def __init__(self, inplanes, planes, padding=1, groups=1, kernel_size=3):
        super(BlockType3, self).__init__()
        self.conv0 = conv1x1(inplanes, planes, stride=2, groups=groups, padding=padding)
        self.bn0 = nn.BatchNorm2d(planes)

        self.type1 = BlockType1(inplanes, planes, groups=groups, padding=padding)
        self.conv1 = conv3x3(planes, planes, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        identity = self.bn0(self.conv0(x[:,:,1:-1, 1:-1]))

        out = self.type1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.avgpool(out)

        out += identity
        return out


class BlockType4(nn.Module):

    def __init__(self, inplanes, planes, groups=1, padding=1, kernel_size=3):
        super(BlockType4, self).__init__()
        self.type1 = BlockType1(inplanes, planes, groups=groups, padding=padding)
        self.conv1 = conv3x3(planes, planes, groups=groups, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.type1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        # out = self.gvp(out)

        return out


class SRNet_pyramid(nn.Module):
    def __init__(self, M1, M4, M8):
        super(SRNet_pyramid, self).__init__()
        self.M1 = M1
        self.M4 = M4
        self.M8 = M8

        self.p8_U = conv3x3_de(in_planes=64, out_planes=32, stride=2, padding=0)
        self.p8_U_bn = nn.BatchNorm2d(32)
        self.p4_td_add = modules.BiFPNAdd(len_features=2)
        self.p4_td_act = nn.ReLU()
        self.p4_td_sep = SeparableConvBlock(in_planes=32, kernel_size=3, stride=1)

        self.p4_U = conv3x3_de(in_planes=32, out_planes=16, stride=2, padding=0, output_padding=1)
        self.p4_U_bn = nn.BatchNorm2d(16)
        self.p1_td_add = modules.BiFPNAdd(len_features=2)
        self.p1_td_act = nn.ReLU()
        self.p1_out = SeparableConvBlock(in_planes=16, kernel_size=3, stride=1)

        self.p1_D = conv3x3(in_planes=16, out_planes=32, stride=2, padding=0)
        self.p1_D_bn = nn.BatchNorm2d(32)
        self.p4_out_add = modules.BiFPNAdd(len_features=3)
        self.p4_out_act = nn.ReLU()
        self.p4_out_sep = SeparableConvBlock(in_planes=32, kernel_size=3, stride=1)

        #self.p4_D = conv3x3(in_planes=32, out_planes=64, stride=2, padding=0)
        #self.p4_D_bn = nn.BatchNorm2d(64)
        #self.p8_out_add = modules.BiFPNAdd(len_features=2)
        #self.p8_out_act = nn.ReLU()
        #self.p8_out_sep = SeparableConvBlock(in_planes=64, kernel_size=3, stride=1)

        self.inplanes=16
        self.layer3_1 = self._make_layer(BlockType3, num_filters=[32], padding=0, kernel_size=4)
        self.inplanes = 64
        self.layer3_4 = self._make_layer(BlockType3, num_filters=[128], padding=0)
        self.inplanes = 128
        #self.layer3_8 = self._make_layer(BlockType3, num_filters=[384], padding=1)
        #self.inplanes = 384
        self.layer4 = self._make_layer(BlockType4, [256], groups=1)
        self.gvp = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, 5, bias=False)
        self.conv_weights = []
        self.non_conv_weights = []

        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.conv_weights.append(param)
            else:
                self.non_conv_weights.append(param)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)



    def _make_layer(self, block, num_filters, groups=1, padding=1, kernel_size=3):
        layers = []
        for planes in num_filters:
            layers.append(block(self.inplanes, planes, groups=groups, padding=padding, kernel_size=kernel_size))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def pyramid(self, input):
        x1 = self.M1(input) #128x128x16
        x4 = self.M4(input) #63x63x32
        x8 = self.M8(input) #31x31x64

        x8_U = self.p8_U(x8)
        x8_U = self.p8_U_bn(x8_U)
        x4_td = self.p4_td_add([x8_U, x4])
        x4_td = self.p4_td_act(x4_td)
        x4_td = self.p4_td_sep(x4_td)

        x4_U = self.p4_U(x4_td)
        x4_U = self.p4_U_bn(x4_U)
        x1_td = self.p1_td_add([x4_U, x1])
        x1_td = self.p1_td_act(x1_td)
        x1_out = self.p1_out(x1_td)

        x1_D = self.p1_D(x1_out)
        x1_D = self.p1_D_bn(x1_D)
        x4_out = self.p4_out_add([x1_D, x4, x4_td])
        x4_out = self.p4_out_act(x4_out)
        x4_out = self.p4_out_sep(x4_out)

        #x4_D = self.p4_D(x4_out)
        #x4_D = self.p4_D_bn(x4_D)
        #x8_out = self.p8_out_add([x4_D, x8])
        #x8_out = self.p8_out_act(x8_out)
        #x8_out = self.p8_out_sep(x8_out)
        return x1_out, x4_out#, x8_out

    def forward(self, input):
        x1, x4 = self.pyramid(input)

        x1_4 = self.layer3_1(x1)
        x4_8 = self.layer3_4(torch.cat([x1_4, x4], dim=1))
        #x = self.layer3_8(torch.cat([x4_8, x8], dim=1))
        x = self.layer4(x4_8)
        x = self.gvp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x