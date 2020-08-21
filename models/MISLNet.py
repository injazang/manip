import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_dct as dct
import numpy as np
from module.modules import augment

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding_mode='replicate', bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias, padding_mode=padding_mode)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=True)

def conv5x5(in_planes, out_planes, stride=1, groups=1, bias=True,  padding_mode='replicate'):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, padding_mode=padding_mode, groups=groups, bias=bias)






class MISLNet(nn.Module):
    def __init__(self, num_labels):
        super(MISLNet, self).__init__()

        self.mask = np.zeros([3, 3, 5, 5])
        self.mask[:, :, 2, 2] = 1
        self.mask = nn.Parameter(torch.from_numpy(self.mask).float())
        self.mask.requires_grad = False

        self.aug = augment()
        self.bayar = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=0, groups=1, bias=False)

        self.conv2 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3, padding_mode='replicate', groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(96, 64, kernel_size=5, stride=1, padding=2, padding_mode='replicate', groups=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tanh3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, padding_mode='replicate', groups=1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.tanh4 = nn.Tanh()
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, groups=1,
                               bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.tanh5 = nn.Tanh()
        self.pool5 = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(in_features=128, out_features=200, bias=True)
        self.tanh6 = nn.Tanh()

        self.fc2 = nn.Linear(in_features=200, out_features=200, bias=True)
        self.tanh7 = nn.Tanh()

        self.fc3 = nn.Linear(in_features=200, out_features=num_labels, bias=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad==True:
                    nn.init.kaiming_uniform_(m.weight)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)


    def constraint(self):
        w = self.bayar.weight.data
        w *= (1 - self.mask)
        rest_sum = torch.sum(w, dim=(2,3), keepdim=True)
        w /= rest_sum + 1e-10
        w -= self.mask
        self.bayar.weight.data = w



    def forward(self, input, prob):
        input = self.aug(input, prob)
        x = self.bayar(input)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.tanh3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.tanh4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.tanh5(x)
        x = self.pool5(x)
        x = x.view(-1, 128)
        
        x = self.fc1(x)
        x = self.tanh6(x)
        x = self.fc2(x)
        x = self.tanh7(x)
        x = self.fc3(x)

        return x