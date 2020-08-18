import torch.nn as nn
import torch
from module.modules import augment

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)

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

class ensenble(nn.Module):
    def __init__(self,model1, model2, num_labels):
        super(ensenble, self).__init__()
        self.model1= model1
        self.model2 = model2
        self.inplanes=640
        self.layer4 = self._make_layer(BlockType4, [1280, ], groups=1)

        self.gvp1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gvp = nn.AdaptiveAvgPool2d((1, 1))

        #self.fc1 = nn.Linear(1024, 1024, bias=False)
        #self.drop1 = nn.Dropout(0.3)

        #self.act1 = nn.ReLU(inplace=True)
        #self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Linear(1280, num_labels, bias=False)

        self.augment=augment()
        self.trainable_parameters=[]
        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.trainable_parameters.append(param)
            else:
                self.trainable_parameters.append(param)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0.2)


    def _make_layer(self, block, num_filters, groups=1):
        layers = []
        for planes in num_filters:
            layers.append(block(self.inplanes, planes, groups=groups))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input, prob):
        im, im_c = input
        im = self.augment(im, prob)

        out1 = self.model1(im, 1)
        out2 = self.model2(im_c,1)
        out = torch.cat([out1, out2], dim=1)
        out = nn.functional.normalize(out, p=2)
        out = self.layer4(out)
        out = self.gvp(out)

        out = out.view(out.size(0), -1)
        #out = self.drop1(out)
        #out = self.fc1(out)
        #out = self.act1(out)
        #out = self.drop2(out)
        out =self.fc(out)
        return out

