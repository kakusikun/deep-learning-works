import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.module.base_module import ConvModule

'''
Code is borrowed from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/hacnn.py
'''

class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvModule(1, 1, 3, stride=2, padding=1)
        self.conv2 = ConvModule(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.interpolate(x, scale_factor=2 ,mode='bilinear', align_corners=True)
        # scaling conv
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvModule(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvModule(in_channels // reduction_rate, in_channels, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = self.gap(x)
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvModule(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels, n_stream=4):
        super(HardAttn, self).__init__()
        self.n_stream = n_stream
        self.fc = nn.Linear(in_channels, self.n_stream * 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(
            torch.tensor(
                [0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float
            )
        )

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = self.gap(x).view(x.size(0), -1)
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, self.n_stream, 2)
        return theta


class HABlock(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HABlock, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta

