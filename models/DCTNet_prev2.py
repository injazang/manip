import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch_dct as dct
from module.mish import Mish


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
        return nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=self.scale, stride=self.scale//2, padding=self.scale//4, padding_mode='replicate', bias=False)

    def forward(self, input):
        #bs, _,_,_ = input.shape()
        #input.view(bs*(128//self.scale)**2, 3, self.scale,self.scale)
        dct_outs= torch.cat([self.dctConvs[i](input[:,i:i+1,...]) for i in range(3)], dim=1)
        dct_reallocate = torch.cat([dct_outs[:,index:index+1,...] for index in self.swap], dim=1)
        return dct_reallocate

def conv3x3(in_planes, out_planes, groups=1, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     padding=1, padding_mode='replicate', bias=bias)


class resBlock(nn.Module):
    def __init__(self, in_planes, out_planes, mish=True, groups=1):
        super(resBlock, self).__init__()
        self.norm1 = nn.GroupNorm(groups, in_planes)
        if mish:
            self.act1 = Mish()
        else:
            self.act1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes=in_planes, out_planes=out_planes, bias=True)

        self.norm2 = nn.GroupNorm(groups, in_planes)
        if mish:
            self.act2 = Mish()
        else:
            self.act2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes=in_planes, out_planes=out_planes, bias=True)

        self.norm3 = nn.GroupNorm(groups, in_planes)
        if mish:
            self.act3 = Mish()
        else:
            self.act3 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(in_planes=in_planes, out_planes=out_planes, bias=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, input):
        x = self.norm1(input)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.conv3(x)
        x = x + input
        return x



class ResModule(nn.Module):
    def __init__(self,in_planes, out_planes, groups, num_blocks, downsample):
        super(ResModule, self).__init__()
        self.resBlocks = [resBlock(in_planes, out_planes, groups=groups).cuda() for i in range(num_blocks)]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, input):
        x = input
        for block in self.resBlocks:
            x =block(x)
        x = input + x
        return x

class DCTNet(nn.Module):
    def __init__(self):
        super(DCTNet, self).__init__()
        self.name = 'dctnet'
        self.dct8 = dct2d_Conv_Layer(scale=8, start=0, num_filters=192)
        self.conv8_8 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True)
        self.resblock8 = ResModule(in_planes=64, out_planes=64, num_blocks=2, groups=16, downsample=False)
        self.conv8_16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, padding_mode='replicate', stride=2, bias=True)

        self.dct16 = dct2d_Conv_Layer(scale=16, start=0, num_filters=192)
        self.conv16_16 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True)
        self.resblock16 = ResModule(in_planes=192, out_planes=192, num_blocks=3, groups=32, downsample=False)
        #self.conv16_32 = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1, padding_mode='replicate', stride=2, bias=True)

        self.dct32 = dct2d_Conv_Layer(scale=32, start=0, num_filters=384)
        self.conv32_32 = nn.Conv2d(in_channels=384  , out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True)
        self.resblock32 = ResModule(in_planes=384, out_planes=384, num_blocks=4, groups=32, downsample=False)

        self.norm8 = nn.GroupNorm(16, 128)
        self.relu8 = Mish()

        self.norm16 = nn.GroupNorm(32, 320)
        self.relu16 = Mish()

        self.gvp = nn.AvgPool2d((8, 8))

        self.fc = nn.Conv2d(in_channels=704, out_channels=5, kernel_size=1, bias=False)
        self.sc = nn.Softmax()
        #self.dct16 = dct2d_Conv_Layer(scale=16)
        #self.dct32 = dct2d_Conv_Layer(scale=32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)


        self.conv_weights = []
        self.non_conv_weights = []
        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.conv_weights.append(param)
            else:
                self.non_conv_weights.append(param)



    def forward(self, input, k=0):
        x8 = self.dct8(input)#[:,3:3+180,...]
        x8 = self.conv8_8(x8)
        x8 = torch.cat([self.resblock8(x8), x8], dim=1)

        x8_16 = self.norm8(x8)
        x8_16 = self.relu8(x8_16)
        x8_16 = self.conv8_16(x8_16)

        x16 = self.dct16(input)#[:,9:9+180,...]
        x16 = self.conv16_16(x16)
        x16 = torch.cat([x8_16, x16], dim=1)
        x16 = torch.cat([self.resblock16(x16), x8_16], dim=1)

        x16_32 = self.norm16(x16)
        x16_32 = self.relu16(x16_32)
        x16_32 = self.conv16_32(x16_32)

        x32 = self.dct32(input)#[:,18:18+360,...]
        x32 = self.conv32_32(x32)
        x32 = torch.cat([x16_32, x32], dim=1)
        x32 = torch.cat([self.resblock32(x32), x16_32], dim=1)

        out = self.gvp(x32)
        #out = out.view(out.size(0), -1)

        out = self.fc(out)
        out = self.sc(out)
        return out
#model = ZhuNet()