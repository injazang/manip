import torch.nn as nn
import torch
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=20):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)
        
class ExpandBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, se=False):
        super(ExpandBlock, self).__init__()
        self.se = se
        self.bn1 = nn.GroupNorm(30, inplanes)
        self.conv1 = conv1x1(inplanes, planes, stride, groups=groups)

        self.bn2 = nn.GroupNorm(30, planes)
        self.conv2 = conv3x3(planes, planes, groups=groups)

        self.bn3 = nn.GroupNorm(30, planes)
        self.conv3 = conv1x1(planes, planes, groups=groups)
        
        if self.se: self.selayer = SELayer(planes)

        self.relu = nn.ReLU(inplace=True)  
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        if self.downsample is not None: identity = self.downsample(x)
        else: identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.se: out = self.selayer(out)

        out += identity
        
        return out


class FSNet(nn.Module):
    def __init__(self, block, layers=[3,3,3], expansion=1, groups=3):
        super(FSNet, self).__init__()
        self.groups = groups
        if self.groups == 3: self.channel = 192
        else: self.channel = 64
        self.conv0_1 = Conv2d(self.channel, 60*expansion, kernel_size=3, stride=1, padding=1, bias=False, groups=self.groups)
       
        self.first = 60*expansion
        self.inplanes = self.first
        self.glayer = self._make_layer(block, 60*expansion, layers[0], stride=1, groups=self.groups)

        self.inplanes = 120*expansion
        self.layer1 = self._make_layer(block, 120*expansion, layers[1], stride=1, groups=self.groups)

        self.inplanes = 240*expansion
        self.layer2 = self._make_layer(block, 240*expansion, layers[2], stride=1, groups=1)

        self.bn = nn.GroupNorm(30, self.inplanes*2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.inplanes*2, 1)

        # Group the weights
        self.conv_weights = []
        self.non_conv_weights = []
        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.conv_weights.append(param)
            else:
                self.non_conv_weights.append(param)

        # Initialize
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.orthogonal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(30, self.inplanes),
                nn.ReLU(inplace=True),
                conv1x1(self.inplanes, planes, stride, groups=groups),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=groups))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)
 
    def forward(self, im_c, im_q):
        bs, _, w, h, _, _ = im_c.size()
        im_c = im_c.permute(0, 1, 4, 5, 2, 3).contiguous().view(bs, 192, w, h)

        conv0 = self.conv0_1(im_c)
        
        cat = conv0
        x = self.glayer(cat)

        # Ensuring the groups for the next group-convolution
        if self.groups==3: cat = torch.cat([cat.view(bs,3,self.first*2//6,w,h), x.view(bs,3,self.first*2//6,w,h)], 1)[:,[0,3,1,4,2,5],...].view(bs,self.first*2,w,h)
        else: cat = torch.cat([cat, x], 1)
        x = self.layer1(cat)

        # Ensuring the groups for the next group-convolution
        if self.groups==3: cat = torch.cat([cat.view(bs,3,self.first*4//6,w,h), x.view(bs,3,self.first*4//6,w,h)], 1)[:,[0,3,1,4,2,5],...].view(bs,self.first*4,w,h)
        else: cat = torch.cat([cat, x], 1)
        x = self.layer2(cat)
        
        cat = torch.cat([cat, x], 1)
        x = self.bn(cat)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return torch.sigmoid(x)

def model_factory(expand_block=True, layers=[2,3,4]):
    if layers is None: raise ValueError('layers argument should be assigned')
    model = FSNet(ExpandBlock, layers)

    return model
