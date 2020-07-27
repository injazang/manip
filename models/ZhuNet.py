import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch_dct as dct

SRM_Kernels = torch.from_numpy(np.load('models/SRM_Kernels.npy')).permute(3, 2, 0, 1)


def conv5x5(in_planes, out_planes, stride=1, groups=1, bias=False):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, groups=groups, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)


class PrepLayer(nn.Module):
    def __init__(self):
        super(PrepLayer, self).__init__()
        self.conv3x3 = conv3x3(1, 25, bias=False)
        self.conv5x5 = conv5x5(1, 5, bias=False)

    def forward(self, x):
        feature3x3 = self.conv3x3(x)
        feature5x5 = self.conv5x5(x)

        return torch.cat([feature3x3, feature5x5], dim=1)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, input):
        return torch.abs(input)


class SepconvLayer(nn.Module):
    def __init__(self):
        super(SepconvLayer, self).__init__()
        self.first_block = nn.Sequential(
            conv3x3(30, 60, bias=False, groups=30),
            Abs(),
            conv1x1(60, 30, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
        )
        self.second_block = nn.Sequential(
            conv3x3(30, 60, bias=False, groups=30),
            conv1x1(60, 30, bias=False),
            nn.BatchNorm2d(30),
        )

    def forward(self, x):
        identity = x
        x = self.first_block(x)
        x = self.second_block(x)
        x = x + identity
        return x


class SPPLayer(nn.Module):

    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                  stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, spp=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if spp:
            self.pool = SPPLayer(3)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        return out


class ZhuNet(nn.Module):

    def __init__(self, jpeg=False):
        super(ZhuNet, self).__init__()
        self.name = 'zhunet'
        self.prep = PrepLayer()
        self.sepconv = SepconvLayer()
        self.blocks = nn.Sequential(
            BasicBlock(30, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 64),
            BasicBlock(64, 128, spp=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(2688, 1024, bias=True),
            nn.Linear(1024, 1, bias=True)
        )
        self.act = nn.Sigmoid()
        self.jpeg=jpeg
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

        self.prep.conv3x3.weight.data = SRM_Kernels[:25, :, 1:4, 1:4]
        self.prep.conv5x5.weight.data = SRM_Kernels[25:, ...]

        self.conv_weights = []
        self.non_conv_weights = []
        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.conv_weights.append(param)
            else:
                self.non_conv_weights.append(param)

    def forward(self, input, k=0):
        if self.jpeg:
            im_c, im_q = input
            bs, _, w, h, _, _ = im_c.size()
            deq = (im_c * im_q).float()

            # restore the pixels from the DCT coefficients, not rounding and truncation operation
            im_dct = dct.idct_2d(deq, norm='ortho').permute(0, 1, 2, 4, 3, 5) + 128
            img = im_dct.reshape((-1, 3, w * 8, h * 8))[:, 0, ...].view(bs, 1, w * 8, h * 8)
        else:
            img = input

        x = self.prep(img / 255)
        x = self.sepconv(x)
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.act(x)