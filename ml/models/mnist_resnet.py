import os
from warnings import warn

import torch
from torch import nn


# 3*3 convolutino
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HalfMNISTResNet(nn.Module):
    def __init__(self, layers, input_channels, output_channels=64, split='top', pretrained=False):
        super(HalfMNISTResNet, self).__init__()
        self.conv = conv3x3(input_channels, 16)
        self.in_channels = 16
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(ResidualBlock, 16, layers[0])
        self.layer2 = self.make_layer(ResidualBlock, 32, layers[0], 2)
        self.layer3 = self.make_layer(ResidualBlock, 64, layers[1], 2)
        self.fc = nn.Linear(1792, output_channels)

        if pretrained:
            pretrained_path = f'pretrained/{split}_model.pth.tar'
            if not os.path.exists(pretrained_path):
                warn(f'There is no pretrained model in the required path: {pretrained_path}, using random init')
                return
            pretrained_state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_state_dict)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        # out = out / torch.sum(out ** 2, dim=-1, keepdim=True)
        return out
