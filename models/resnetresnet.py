import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Function
from math import sqrt
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.utils import spectral_norm
F_conv = torch.nn.functional.conv2d

''' fix the last four layers and put a small net on the head
'''
style_dim = 256
stdd = 0.1

''' ------ conv3d small size h 
no style'''

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class AdaptiveInstanceNorm_H(nn.Module):
    def __init__(self, in_channel, map_size):
        super().__init__()

        # self.norm = nn.InstanceNorm2d(in_channel)
        self.norm = nn.LayerNorm([map_size, map_size])
        # self.style = EqualLinear(style_dim, in_channel * 2)
        #
        # self.style.linear.bias.data[:in_channel] = 1
        # self.style.linear.bias.data[in_channel:] = 0

        self.weight = nn.Parameter(1000.0 + torch.randn(1, in_channel, 1, 1))
        self.beta = nn.Parameter(0.0 + torch.randn(1, in_channel, 1, 1))

    def forward(self, input, style=0):
        out = self.norm(input)
        # out = self.weight * out + self.beta
        # out = self.weight * input + self.beta
        out = 1e-2 * out + out.detach() * self.weight + self.beta

        return out

class StyleBlock_noise(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True, initial=False, upsample=False, fused=False):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        kernel_size = 3
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.ConvTranspose2d(self.fin, self.fout, 4, stride=2, padding=1)
        self.noise_0 = NoiseInjection(self.fout)
        self.adain_0 = AdaptiveInstanceNorm(self.fout, style_dim)
        self.lrelu_0 = nn.LeakyReLU(0.2)

        self.conv_1 = EqualConv2d(self.fout, self.fout, 3, stride=1, padding=1)
        self.noise_1 = NoiseInjection(self.fout)
        self.adain_1 = AdaptiveInstanceNorm(self.fout, style_dim)
        self.lrelu_1 = nn.LeakyReLU(0.2)

        self.conv_s = nn.ConvTranspose2d(self.fin, self.fout, 4, stride=2, padding=1, bias=False)

    def forward(self, x, style, noise=0):

        x_s = self.conv_s(x)

        out = self.lrelu_0(x)
        out = self.conv_0(out)

        batch_size, cc, hh, ww = out.shape
        noise_0 = torch.randn(batch_size, 1, hh, ww, device=x[0].device)
        out = self.noise_0(out, noise_0.data)
        out = self.adain_0(out, style)

        out = self.lrelu_1(out)
        out = self.conv_1(out)

        batch_size, cc, hh, ww = out.shape
        noise_1 = torch.randn(batch_size, 1, hh, ww, device=x[0].device)
        out = self.noise_1(out, noise_1.data)
        out = self.adain_1(out, style)

        out = x_s + 0.1 * out

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        self.z_dim = z_dim
        small_nf = self.small_nf = 64

        # Submodules
        self.small_embedding = nn.Embedding(nlabels, embed_size)
        self.small_fc = nn.Linear(z_dim, 8 * small_nf * s0 * s0)

        # self.small_net_1 = StyleBlock_firstLayer(8 * small_nf, 8 * small_nf, initial=True)
        self.small_net_2 = StyleBlock_noise(8 * small_nf, 8 * small_nf, upsample=True)
        self.small_net_3 = StyleBlock_noise(8 * small_nf, 8 * small_nf, upsample=True)

        # self.small_Attn = Self_Attn(8 * small_nf)

        self.small_H = AdaptiveInstanceNorm_H(8 * small_nf, 16)

        self.resnet_3_0 = ResnetBlock_style(8 * nf, 4 * nf)
        self.resnet_3_1 = ResnetBlock_style(4 * nf, 4 * nf)

        # self.small_Attn = Self_Attn(4 * nf)

        self.resnet_4_0 = ResnetBlock_style(4 * nf, 2 * nf)
        self.resnet_4_1 = ResnetBlock_style(2 * nf, 2 * nf)

        self.resnet_5_0 = ResnetBlock_style(2 * nf, 1 * nf)
        self.resnet_5_1 = ResnetBlock_style(1 * nf, 1 * nf)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

        layers = [PixelNorm()]
        # layers = []
        layers.append(EqualLinear(z_dim, style_dim))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(7):
            layers.append(EqualLinear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.small_style = nn.Sequential(*layers)

    def forward(self, z, y, FLAG=500):
        assert (z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.small_embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        yz = torch.cat([z, yembed], dim=1)
        style_w = self.small_style(z)
        # print('yembed ============ ', yembed.shape)
        out = self.small_fc(z)
        out = out.view(batch_size, 8 * self.small_nf, self.s0, self.s0)

        out = self.small_net_2(out, style_w)
        out_h = self.small_net_3(out, style_w)
        out_h = self.small_H(out_h)

        out = F.interpolate(out_h, scale_factor=2)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        # out = self.small_Attn(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out0 = self.conv_img(actvn(out))
        out = torch.tanh(out0)

        loss_w = style_w.pow(2).sum(1)

        return out, loss_w
