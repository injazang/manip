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





class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()
        self.regular = conv5x5(3, 10, bias=False)
        self.bayar = conv5x5(3, 3, bias=False)
        self.srm = conv5x5(3, 3, bias=False)
        self.srm_kernel = self._build_SRM_kernel()


    def _get_srm_list( self ) :
        # srm kernel 1
        srm1 = np.zeros([5,5]).astype('float32')
        srm1[1:-1,1:-1] = np.array([[-1, 2, -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]] )
        srm1 /= 4.
        # srm kernel 2
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.
        # srm kernel 3
        srm3 = np.zeros([5,5]).astype('float32')
        srm3[2,1:-1] = np.array([1,-2,1])
        srm3 /= 2.
        return [ srm1, srm2, srm3 ]
    def _build_SRM_kernel( self ) :
        kernel = []
        srm_list = self._get_srm_list()
        this_ch_kernel = np.zeros([3, 5, 5]).astype('float32')
        for idx, srm in enumerate( srm_list ):
            this_ch_kernel[idx, :,:] = srm
            kernel.append( this_ch_kernel )
        kernel = np.stack(kernel, axis=0)
        srm_kernel = torch.from_numpy(kernel).float()
        return srm_kernel

    def forward(self, x):
        feat_reg = self.regular(x)
        feat_bayar = self.bayar(x)
        feat_srm = self.srm(x)
        return torch.cat([feat_reg, feat_bayar, feat_srm], dim=1)




class MantrNet(nn.Module):
    def __init__(self):
        super(MantrNet, self).__init__()
        self.aug = augment()
        self.inplanes = 16
        self.name = 'srnet'
        self.pre = Preprocess()
        self.conv_layers = self._make_layer([32,64,64, 128,128,256,256,256,256,256])
        self.last = conv3x3(in_planes=self.inplanes, out_planes=20)
        self.conv_weights = []
        self.non_conv_weights = []

        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.conv_weights.append(param)
            else:
                self.non_conv_weights.append(param)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad==True:
                    nn.init.kaiming_uniform_(m.weight)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

        self.pre.srm.weight.data = self.pre.srm_kernel
        self.pre.srm.weight.requires_grad = False
        self.mask =np.zeros([3,3,5,5])
        self.mask[:,:,2,2]=1
        self.mask = nn.Parameter(torch.from_numpy(self.mask))
        self.mask.requires_grad=False
        #self.constraint()

    def constraint(self):
        w = self.pre.bayar.weight.data
        w *= (1 - self.mask)
        rest_sum = torch.sum(w, dim=(2,3), keepdim=True)
        w /= rest_sum + 1e-10
        w -= self.mask
        self.pre.bayar.weight.data = w

    def _make_layer(self, num_filters, groups=1):
        layers = []
        for planes in num_filters[:-1]:
            layers.append(conv3x3(self.inplanes, planes, groups=groups, bias=True))
            layers.append(nn.ReLU())
            self.inplanes = planes
        layers.append(conv3x3(self.inplanes, num_filters[-1], groups=groups, bias=True))
        return nn.Sequential(*layers)

    def forward(self, input, prob):
        input = self.aug(input, prob)
        x = self.pre(input)
        x = self.conv_layers(x)
        x = nn.functional.normalize(x, p=2)
        x = self.last(x)
        return x