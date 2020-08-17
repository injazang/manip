import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_dct as dct


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1 ,padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, dilation=dilation, stride=stride, padding=padding, padding_mode='replicate', groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


class Preprocess(nn.Module):

    def __init__(self, inplanes, planes, groups=1):
        super(Preprocess, self).__init__()
        self.conv1 = conv3x3(11, 64, groups=groups, dilation=8, padding=8)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1= nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 64, groups=groups)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu2(out)
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


class HistNet(nn.Module):
    def __init__(self,num_labels=5 ,load=False):
        super(HistNet, self).__init__()

        self.inplanes = 11
        self.name = 'srnet'
        self.layer1 = self._make_layer(Preprocess, [64], groups=1)
        self.layer2 = self._make_layer(BlockType2, [64, 64, 64, 64], groups=1)
        self.layer3 = self._make_layer(BlockType3, [64, 64, 128, 256], groups=1)
        self.layer4 = self._make_layer(BlockType4, [512, ], groups=1)
        self.gvp = nn.AdaptiveAvgPool2d((1, 1))
        self.load=load
        if not load:
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
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _make_layer(self, block, num_filters, groups=1):
        layers = []
        for planes in num_filters:
            layers.append(block(self.inplanes, planes, groups=groups))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def hist(self, im_c, threshold):
        bs, _, w, h = im_c.size()
        im_c = torch.abs(im_c)
        im_c[im_c>threshold] =threshold
        output=[]
        for i in range(threshold+1):
            template = torch.zeros(size=[bs,1,w,h]).float().cuda()
            template[im_c == i] = 1
            output.append(template)
        output = torch.cat(output,dim=1)
        return output

    def forward(self, input, prob):
        im_c = input
        hist = self.hist(im_c, 10)

        x = self.layer1(hist)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gvp(x)
        x = x.view(x.size(0), -1)
        if not self.load:

            x = self.fc(x)
        return x