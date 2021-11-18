# model.py

import torch
import torch.nn as nn
import netron
from torchvision.models import resnet18


class Discrimiator(nn.Module):
    def __init__(self, num_channels):
        super(Discrimiator, self).__init__()

        self.conv = nn.Conv2d(in_channels=num_channels,
                              out_channels=64,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.block1 = self._block(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.block2 = self._block(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.block3 = self._block(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.final_conv = nn.Conv2d(in_channels=512,
                                    out_channels=1,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0)

        self.activation = nn.Sigmoid()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)  # [32, 64, 32, 32]
        x = self.leaky_relu(x)  # [32, 64, 32, 32]

        x = self.block1(x)  # [32, 128, 16, 16]
        x = self.block2(x)  # [32, 256, 8, 8]
        x = self.block3(x)  # [32, 512, 4, 4]

        x = self.final_conv(x)  # [32, 1, 1, 1]
        x = self.activation(x)  # [32, 1, 1, 1]

        return x


class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels):
        super(Generator, self).__init__()

        self.block1 = self._block(in_channels=noise_channels, out_channels=512, kernel_size=4, stride=1, padding=0)

        self.block2 = self._block(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.block3 = self._block(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.block4 = self._block(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.upsample = nn.ConvTranspose2d(in_channels=64, out_channels=img_channels, kernel_size=4,
                                           stride=2, padding=1)  # N x C x 64 x 64
        self.activation = nn.Tanh()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.block1(x)  # [32, 512, 4, 4]
        x = self.block2(x)  # [32, 256, 8, 8]
        x = self.block3(x)  # [32, 128, 16, 16]
        x = self.block4(x)  # [32, 64, 32, 32]

        x = self.upsample(x)  # [32, 3, 64, 64]
        x = self.activation(x)

        return x


def initialize_wieghts(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    # device = torch.device("cuda")
    x = torch.randn((32, 100, 1, 1))
    disc = Generator(noise_channels=100, img_channels=3)
    # # path = "2.onnx"
    res = disc(x)
    # # netron.start(path)
    # x = torch.rand(32, 3, 64, 64)
    # disc = Discrimiator(num_channels=3)
    # disc(x)
