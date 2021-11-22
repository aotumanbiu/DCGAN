import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import os
from collections import OrderedDict
from torch.nn.init import kaiming_normal_


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super(ApplyNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn_like(x).unsqueeze(1)
        return x + self.weight * noise


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):

        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                         channels * 2,
                         gain=1.0,
                         use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


if __name__ == "__main__":
    a = ApplyNoise(3).cuda()
    x = torch.randn(3, 5, 5).cuda()
    b = a(x)
