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


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    # EqualLR.apply(module, name)

    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        # conv.weight.data.normal_()
        # conv.weight.data.normal_(0.0, stdd)
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_(0.0, stdd)
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        gamma_c = gamma.clamp(-1.0, 1.0)
        out = gamma_c * out + out.detach() * (gamma - gamma_c) + beta

        # out = gamma * input + beta

        return out


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1,
                                        padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1,
                                      padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1,
                                    padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, stride=1,
                                       padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma * attn_g
        return out


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


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.weight = nn.Parameter(torch.randn(1, channel, 1, 1) * 0.1)

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyleBlock_firstLayer(nn.Module):
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
        self.conv_0 = ConstantInput(self.fin)
        self.noise_0 = equal_lr(NoiseInjection(self.fout))

    def forward(self, x, style, noise=0):

        out = self.conv_0(x)
        batch_size, cc, hh, ww = out.shape
        noise_0 = torch.randn(batch_size, 1, hh, ww, device=x[0].device)
        out = self.noise_0(out, noise_0.data)

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
        self.noise_0 = equal_lr(NoiseInjection(self.fout))
        self.adain_0 = AdaptiveInstanceNorm(self.fout, style_dim)
        self.lrelu_0 = nn.LeakyReLU(0.2)

        self.conv_1 = EqualConv2d(self.fout, self.fout, 3, stride=1, padding=1)
        self.noise_1 = equal_lr(NoiseInjection(self.fout))
        self.adain_1 = AdaptiveInstanceNorm(self.fout, style_dim)
        self.lrelu_1 = nn.LeakyReLU(0.2)

        self.conv_s = nn.ConvTranspose2d(self.fin, self.fout, 4, stride=2, padding=1, bias=False)

    def forward(self, x, style, noise=0):

        x_s = self.conv_s(x)

        # out = self.lrelu_0(x)
        # out = self.adain_0(out, style)
        # out = self.conv_0(out)
        # batch_size, cc, hh, ww = out.shape
        # noise_0 = torch.randn(batch_size, 1, hh, ww, device=x[0].device)
        # out = self.noise_0(out, noise_0.data)
        #
        # out = self.lrelu_1(out)
        # out = self.adain_1(out, style)
        # out = self.conv_1(out)
        # batch_size, cc, hh, ww = out.shape
        # noise_1 = torch.randn(batch_size, 1, hh, ww, device=x[0].device)
        # out = self.noise_1(out, noise_1.data)

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

        # self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        # self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)
        #
        # self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        # self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)
        #
        # self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        # self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)

        self.resnet_3_0 = ResnetBlock_adafm(8 * nf, 4 * nf)
        self.resnet_3_1 = ResnetBlock_adafm(4 * nf, 4 * nf)

        # self.small_Attn = Self_Attn(4 * nf)

        self.resnet_4_0 = ResnetBlock_adafm(4 * nf, 2 * nf)
        self.resnet_4_1 = ResnetBlock_adafm(2 * nf, 2 * nf)

        self.resnet_5_0 = ResnetBlock_adafm(2 * nf, 1 * nf)
        self.resnet_5_1 = ResnetBlock_adafm(1 * nf, 1 * nf)

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

        # out = self.small_net_1(z, style_w)
        # out = F.interpolate(out, scale_factor=2)
        out = self.small_net_2(out, style_w)
        # out = F.interpolate(out, scale_factor=2)
        out_h = self.small_net_3(out, style_w)

        # out_h = self.small_Attn(out_h)

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


''' ORIGINAL'''


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        ny = nlabels

        # Submodules
        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1 * nf, 1 * nf)
        self.resnet_0_1 = ResnetBlock(1 * nf, 2 * nf)

        self.resnet_1_0 = ResnetBlock(2 * nf, 2 * nf)
        self.resnet_1_1 = ResnetBlock(2 * nf, 4 * nf)

        self.resnet_2_0 = ResnetBlock(4 * nf, 4 * nf)
        self.resnet_2_1 = ResnetBlock(4 * nf, 8 * nf)

        self.resnet_3_0 = ResnetBlock(8 * nf, 8 * nf)
        self.resnet_3_1 = ResnetBlock(8 * nf, 16 * nf)

        self.resnet_4_0 = ResnetBlock(16 * nf, 16 * nf)
        self.resnet_4_1 = ResnetBlock(16 * nf, 16 * nf)

        self.resnet_5_0 = ResnetBlock(16 * nf, 16 * nf)
        self.resnet_5_1 = ResnetBlock(16 * nf, 16 * nf)

        self.fc = nn.Linear(16 * nf * s0 * s0, nlabels)
        print('nlabels ============ ', nlabels)

    def forward(self, x, y):
        assert (x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = out.view(batch_size, 16 * self.nf * self.s0 * self.s0)
        out = self.fc(actvn(out))

        # index = torch.LongTensor(range(out.size(0)))
        # if y.is_cuda:
        #     index = index.cuda()
        # out = out[index, y]

        return out



class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x, intitial=False):
        x_s = self._shortcut(x)
        if intitial:
            dx = self.conv_0(x)
        else:
            dx = self.conv_0(actvn(x))

        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


def special_reshape_3d(x):
    Nn = x.shape[0]
    out = x.squeeze().permute(0, 2, 3, 1)  # [4, 64, 64, 32]
    out = out.view(Nn, 16, 4, 64, 32).permute(0, 1, 3, 4, 2)  # [4, 16, 64, 32, 4]
    out = out.view(Nn, 16, 16, 4, 32, 4)  # [4, 16, 16, 4, 32, 4]
    out = out.reshape(Nn, 16, 16, 32 * 4 * 4).permute(0, 3, 1, 2)
    return out


def special_reshape_2d(x, s):
    Nn, Cc, Ww, Hh = x.shape
    out = x.squeeze().permute(0, 2, 3, 1)  # [4, 64, 64, 32]
    out = out.view(Nn, s[1], (Ww / s[1]).__int__(), Hh, Cc).permute(0, 1, 3, 4, 2)  # [4, 16, 64, 32, 4]
    out = out.view(Nn, s[1], s[2], (Hh / s[2]).__int__(), Cc, (Ww / s[1]).__int__())  # [4, 16, 16, 4, 32, 4]
    out = out.reshape(Nn, s[1], s[2], (Hh / s[2]).__int__() * Cc * (Ww / s[1]).__int__()).permute(0, 3, 1, 2)
    return out


class StyleBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.lrelu_0 = nn.LeakyReLU(0.2)
        self.conv_0 = nn.ConvTranspose2d(self.fin, self.fout, 4, stride=2, padding=1)
        self.style_adain_0 = AdaptiveInstanceNorm(self.fout, style_dim)

    def forward(self, x, style, noise=0):
        dx = self.lrelu_0(x)
        dx = self.conv_0(dx)
        dx = self.style_adain_0(dx, style)

        out = dx

        return out


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
                         weight[:, :, 1:, 1:]
                         + weight[:, :, :-1, 1:]
                         + weight[:, :, 1:, :-1]
                         + weight[:, :, :-1, :-1]
                 ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class ResnetBlock_adafm(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.lrelu_0 = nn.LeakyReLU(0.2)
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.small_adafm_0 = AdaFM(self.fhidden, self.fin, style_dim)
        # self.style_noise_0 = NoiseInjection(self.fhidden)

        self.lrelu_1 = nn.LeakyReLU(0.2)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        self.small_adafm_1 = AdaFM(self.fout, self.fhidden, style_dim)
        # self.style_noise_1 = NoiseInjection(self.fhidden)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x, style=0, noise=0):
        x_s = self._shortcut(x)

        dx = self.lrelu_0(x)
        F_weight0 = self.small_adafm_0(self.conv_0.weight, style)
        # dx = self.conv_0(dx)
        dx = F_conv(dx, F_weight0, bias=self.conv_0.bias, stride=1, padding=1)

        dx = self.lrelu_1(dx)
        F_weight1 = self.small_adafm_1(self.conv_1.weight, style)
        # dx = self.conv_1(dx)
        dx = F_conv(dx, F_weight1, bias=self.conv_1.bias, stride=1, padding=1)

        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class AdaFM(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim=0):
        super().__init__()

        self.style_gama = nn.Parameter(torch.ones(in_channel, out_channel, 1, 1))
        self.style_beta = nn.Parameter(torch.zeros(in_channel, out_channel, 1, 1))

    def forward(self, input, style=0):
        h = input.shape[2]
        # input = self.norm(input)
        gamma = self.style_gama.repeat(1, 1, h, h)
        beta = self.style_beta.repeat(1, 1, h, h)
        out = gamma * input + beta
        return out
