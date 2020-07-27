import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch_dct as dct
from module.mish import Mish

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=8):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class dct2d_Conv_Layer(nn.Module):
    def __init__(self, scale):
        super(dct2d_Conv_Layer, self).__init__()
        self.scale = scale
        self.dct_base = self.load_DCT_basis()
        self.dctConvs = [ self.dct2d_Conv(),  self.dct2d_Conv(),  self.dct2d_Conv()]

        for conv in self.dctConvs:
            conv.weight.data = self.dct_base
            conv.weight.requires_grad=False

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
        basis_64 = np.zeros((self.scale **2, self.scale, self.scale))
        idx = 0
        for i in range(self.scale * 2-1):
            cur = max(0, i-self.scale+1)
            for j in range(cur,i-cur + 1):
                basis_64[idx,:, : ] = self.cal_basis(j, i-j)
                idx = idx + 1
        return torch.from_numpy(basis_64).view(self.scale**2, 1, self.scale, self.scale).cuda().float()


    def dct2d_Conv(self):
        return nn.Conv2d(in_channels=1, out_channels=self.scale**2, kernel_size=self.scale, stride=self.scale//2,  bias=False)

    def forward(self, input):
        #bs, _,_,_ = input.shape()
        #input.view(bs*(128//self.scale)**2, 3, self.scale,self.scale)
        return torch.cat([self.dctConvs[i](input[:,i:i+1,...]) for i in range(3)], dim=1)


def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     padding=1, bias=False)


class BasicMoudle(nn.Module):
    def __init__(self, inplanes, stride=1, downsample_ch=False, downsample_res=False, k_size=8, mish=False, use_bn=False):
        super(BasicMoudle, self).__init__()
        self.downsample_ch = downsample_ch
        self.downsample_res = downsample_res
        self.conv1 = conv3x3(inplanes, inplanes, stride=stride, groups=3)
        if use_bn:
            self.norm1 = nn.BatchNorm2d(inplanes)
        else:
            self.norm1 = nn.GroupNorm(3, inplanes)
        if mish:
            self.act = Mish()
        else:
            self.act = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, inplanes, stride=1, groups=3)
        if use_bn:
            self.norm2 = nn.BatchNorm2d(inplanes)
        else:
            self.norm2 = nn.GroupNorm(3, inplanes)
        self.ecas = [eca_layer(inplanes//3, k_size).cuda() for i in range(3)]
        if self.downsample_ch:
            self.down_ch = nn.MaxPool3d((2,1,1),stride=(2,1,1))
        if self.downsample_res and not self.downsample_ch:
            self.down_res = nn.Sequential(
                nn.Conv2d(inplanes, inplanes,
                          kernel_size=1, stride=2, bias=False, padding=1, padding_mode='replicate',groups=3),
                nn.GroupNorm(3, inplanes),
            )
        if self.downsample_res and self.downsample_ch:
            self.down_res = nn.Sequential(
                nn.Conv2d(inplanes//2, inplanes//2,
                          kernel_size=1, stride=2, bias=False, padding=1, padding_mode='replicate',groups=3),
                nn.GroupNorm(3, inplanes//2),
            )

        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
    def forward(self, x):
        bs, c, _, _ = x.size()
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = torch.cat([self.ecas[i](out[:,c//3*i:c//3*(i+1),...]) for i in range(3)], dim=1)
        out += residual
        out = self.act(out)
        if self.downsample_ch:
            bs, c, w, h = out.size()
            out = out.view(bs, 1, c, w, h)
            out = self.down_ch(out).view(bs,c//2, w, h)
        if self.downsample_res:
            out = self.down_res(out)
            bs, c, w, h = out.size()
            out = out[:,:,1:w-1, 1:h-1]
        return out



class DCTMoudule(nn.Module):
    def __init__(self, size, ratio, num_convs, mish=False, use_bn=False, num_downsample=2):
        super(DCTMoudule, self).__init__()
        self.size = size
        self.num_downsample=num_downsample
        self.ratio = ratio
        self.vis_kernels = int(self.size ** 2 * self.ratio)
        self.num_kernels = self.size ** 2
        self.freq_kernels = self.num_kernels-self.vis_kernels
        self.mish=mish
        self.freq_convs = [BasicMoudle(3 * self.freq_kernels // 2**i, k_size=self.size//2-1, mish=mish, use_bn=use_bn, downsample_ch=True, downsample_res = i>=num_convs- num_downsample).cuda() for i in range(num_convs)]
        self.vis_convs = [BasicMoudle(3 * (self.vis_kernels-1), k_size=self.size//2-1, mish=mish, use_bn=use_bn,  downsample_res = i>=num_convs- num_downsample ).cuda() for i in range(num_convs)]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
    def separate(self, input):
        vis = torch.cat([input[:, self.num_kernels * i+1: self.num_kernels * i + self.vis_kernels, ...] for i in range(3)], dim=1)
        freq = torch.cat([input[:, self.num_kernels * i + self.vis_kernels:self.num_kernels * (i + 1), ...] for i in range(3)],dim=1)

        return vis, freq

    def forward(self, input):
        vis, freq = self.separate(input)
        for vis_conv in self.vis_convs:
            vis = vis_conv(vis)
        for freq_conv in self.freq_convs:
            freq = freq_conv(freq)

        out = torch.cat([vis, freq], dim=1)
        return out

class DCTNet(nn.Module):
    def __init__(self):
        super(DCTNet, self).__init__()
        self.name = 'dctnet'
        self.dct8 = dct2d_Conv_Layer(scale=8)
        self.dct8_moudle = DCTMoudule(size=8, ratio=1/8, num_convs=3, mish=True, use_bn=False, num_downsample=2)

        self.dct16 = dct2d_Conv_Layer(scale=16)
        self.dct16_moudle = DCTMoudule(size=16, ratio=1 / 16, num_convs=4, mish=True, use_bn=False, num_downsample=1)

        self.dct32 = dct2d_Conv_Layer(scale=32)
        self.dct32_moudle = DCTMoudule(size=32, ratio=1 / 32, num_convs=5, mish=True, use_bn=False, num_downsample=0)
        self.gvp = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(318, 5, bias=False)
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
        input *= 255
        input = input.cuda()
        x8 = self.dct8(input)
        x8 = self.dct8_moudle(x8)

        x16 = self.dct16(input)
        x16 = self.dct16_moudle(x16)

        x32 = self.dct32(input)
        x32 = self.dct32_moudle(x32)

        out = torch.cat([x8, x16, x32], dim=1)
        out = self.gvp(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        out = self.sc(out)
        return out

#model = ZhuNet()