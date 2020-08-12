import torch.nn as nn
import torch
import numpy as np


class dct2d_Conv_Layer(nn.Module):
    def __init__(self, scale, start, num_filters):
        super(dct2d_Conv_Layer, self).__init__()
        self.scale = scale
        self.start = start
        self.num_filters= num_filters//3
        self.dct_base = self.load_DCT_basis()
        self.dctConvs = [ self.dct2d_Conv(),  self.dct2d_Conv(),  self.dct2d_Conv()]

        for conv in self.dctConvs:
            conv.weight.data = self.dct_base
            conv.weight.requires_grad=False
        self.swap = []
        for i in range(self.num_filters):
            self.swap+=[i, i+self.num_filters, i+self.num_filters * 2]

    def cal_scale(self, p, q):
        if p == 0:
            ap = 1 / (np.sqrt(self.scale))
        else:
            ap = np.sqrt(2 / self.scale)
        if q == 0:
            aq = 1 / (np.sqrt(self.scale))
        else:
            aq = np.sqrt(2 / self.scale)

        return ap, aq

    def cal_basis(self, p, q):
        basis = np.zeros((self.scale, self.scale))
        ap, aq = self.cal_scale(p, q)
        for m in range(0, self.scale):
            for n in range(0, self.scale):
                basis[m, n] = ap * aq * np.cos(np.pi * (2 * m + 1) * p / (2 * self.scale)) * np.cos(
                    np.pi * (2 * n + 1) * q / (2 * self.scale))
        return basis

    def load_DCT_basis(self):
        basis_64 = np.zeros((self.num_filters, self.scale, self.scale))
        idx = 0
        for i in range(self.scale * 2-1):
            cur = max(0, i-self.scale+1)
            for j in range(cur,i-cur + 1):
                if idx >= self.num_filters + self.start:
                    return torch.from_numpy(basis_64).view(self.num_filters, 1, self.scale, self.scale).cuda().float()
                if idx >= self.start:
                    basis_64[idx-self.start, :, :] = self.cal_basis(j, i - j)
                idx = idx + 1
                if idx >= self.num_filters + self.start:
                    return torch.from_numpy(basis_64).view(self.num_filters, 1, self.scale, self.scale).cuda().float()

    def dct2d_Conv(self):
        return nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=self.scale, stride=self.scale // 2,
                         padding_mode='replicate', bias=False)

    def forward(self, input):
        #bs, _,_,_ = input.shape()
        #input.view(bs*(128//self.scale)**2, 3, self.scale,self.scale)
        dct_outs= torch.cat([self.dctConvs[i](input[:,i:i+1,...]) for i in range(3)], dim=1)
        dct_reallocate = torch.cat([dct_outs[:,index:index+1,...] for index in self.swap], dim=1)
        return dct_reallocate



class BiFPNAdd(nn.Module):
    def __init__(self, epsilon = 1e-4, len_features=3):
        super(BiFPNAdd, self).__init__()
        self.epsilon = epsilon
        self.act = nn.ReLU()
        self.len_features = len_features
        self.w = torch.ones(size=[self.len_features], requires_grad=True).cuda().float() /self.len_features
    def forward(self, inputs):
        w = self.act(self.w)
        x = torch.stack([w[i] * inputs[i] for i in range(len(inputs))], dim=1).sum(dim=1)
        x = x / (w.sum() + self.epsilon)
        return x
'''
Applies the mish function element-wise:
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
'''


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.mish(input)

class augment(nn.Module):
    def __init__(self):
        super(augment, self).__init__()

    def Rotate90sFlips(self, imgs, p, planes=[2, 3], k=0):
        if p < 0.125:
            return imgs.rot90(1, planes)
        elif p < 0.25:
            return imgs.rot90(2, planes)
        elif p < 0.375:
            return imgs.rot90(3, planes)
        elif p < 0.5:
            return imgs.flip(planes[0])
        elif p < 0.625:
            return imgs.rot90(1, planes).flip(planes[0])
        elif p < 0.75:
            return imgs.rot90(2, planes).flip(planes[0])
        elif p < 0.875:
            return imgs.flip(planes[0]).rot90(1, planes)
        return imgs

    def forward(self, input, param=None):
        if param==None:
            param = random.random()
        input = self.Rotate90sFlips(input, param)
        return input