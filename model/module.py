import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    """
    (standard) Conv => BN => ReLU
    (linear)   Conv => BN
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, linear=False):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride,
            padding=padding, 
            bias=False, 
            groups=groups)
        
        self.bn = nn.BatchNorm2d(out_channels)
        if not linear:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class InversedDepthwiseSeparable(nn.Module):
    """
    Inversed Depthwise Separable
    Conv 1x1 => Conv 3x3 => BN => ReLU
    """    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InversedDepthwiseSeparable, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            stride=1, 
            padding=0, 
            bias=False)
        self.conv2 = ConvModule(
            out_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=1, 
            groups=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DepthwiseSeparable(nn.Module):
    """
    Inversed Depthwise Separable
     dwConv 3x3 => Conv 1x1 => BN => ReLU
    """    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InversedDepthwiseSeparable, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            stride=1, 
            padding=0, 
            bias=False)
        self.conv2 = ConvModule(
            out_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=1, 
            groups=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SEModule(nn.Module):
    """
    input 
        => GAP 
        => FC channel discounted 
        => ReLU 
        => FC channel restored 
        => Sigmoid * input
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelGate, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(
            in_channels,
            in_channels//reduction, 
            bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            in_channels//reduction, 
            in_channels, 
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, _, _ = x.shape
        y = self.global_avgpool(x)
        y = self.fc1(x.view(n, c))
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(n, c, 1, 1)
        return x * y.expand_as(x)

class Res2NetStem(nn.Module):
    """
    Conv 3x3xs2 => Conv 3x3 => Conv 3x3 => MAX 3x3xs2
    """
    def __init__(self, in_channels, out_channels):
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),        
            nn.MaxPool2d(
                3,
                stride=2,
                padding=1
            )   
        )
    def forward(self, x):
        return self.stem(x)

class biFPN(nn.Module):
    def __init__(self, )