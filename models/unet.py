import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['UNet']


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, channels=(64, 96, 128, 192, 256, 384, 512), strides=(2,)*6):
        super().__init__()
        self.depth = len(channels) - 1
        assert len(strides) == self.depth, 'The length of strides should equal to len(channels) - 1'

        self.first = ConvBlock(in_channels, channels[0])

        for i in range(self.depth):
            setattr(self, f'down{i}', Down(channels[i], channels[i+1], strides[i]))

        for i in range(self.depth):
            setattr(self, f'up{i}', Up(channels[-i-1], channels[-i-2], strides[-i-1]))

        self.final = nn.Conv2d(channels[0], num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = [self.first(x)]
        for i in range(self.depth):
            x = getattr(self, f'down{i}')(outs[-1])
            outs.append(x)

        for i in range(self.depth):
            x = getattr(self, f'up{i}')(outs[-i-1], outs[-i-2])
            outs[-i-2] = x
        x = self.final(outs[0])

        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__(
            nn.MaxPool2d(scale_factor, stride=scale_factor),
            ConvBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, scale_factor, stride=scale_factor, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = ConvBlock(2*out_channels, out_channels)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.conv2(torch.cat([y, x], dim=1))
        return x


if __name__ == '__main__':
    from torchinfo import summary

    model = UNet(3, 1)
    summary(model, input_size=(2, 3, 256, 256))  # N, C, H, W
