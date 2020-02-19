from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
from manager.utility import ConvFC, AttentionIncorporation
import torchvision
from model.module import ConvModule, Res2NetStem, InversedDepthwiseSeparable, SEModule

##########
# Basic layers
##########
class OSBlock(nn.Module):
    """
    Omni-scale feature learning block.
                        x
                        |
                    Conv 1x1
                        |
                        |                           |                           |                           |
            InversedDepthwiseSeparable  InversedDepthwiseSeparable  InversedDepthwiseSeparable  InversedDepthwiseSeparable
            InversedDepthwiseSeparable  InversedDepthwiseSeparable  InversedDepthwiseSeparable              |
            InversedDepthwiseSeparable  InversedDepthwiseSeparable              |                           |
            InversedDepthwiseSeparable              |                           |                           |
                        |                           |                           |                           |
    """
    
    def __init__(self, in_channels, out_channels, bottleneck_reduction=4):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1
        )
        self.conv2a = InversedDepthwiseSeparable(
            mid_channels, 
            mid_channels,
            3
        ) 
        self.conv2b = nn.Sequential(
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            )
        )
        self.conv2c = nn.Sequential(
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            )
        )
        self.conv2d = nn.Sequential(
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            )
        )
        self.gate = SEModule(mid_channels)
        self.conv3 = ConvModule(
            mid_channels,
            out_channels,
            1,
            linear=True)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvModule(
                in_channels, 
                out_channels,
                1,
                linear=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = self.relu(x3 + residual)
        return out


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ArXiv preprint, 2019.
          https://arxiv.org/abs/1905.00953
    """

    def __init__(self, blocks, layers, channels, feature_dim=512, task='reid'):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1 
        self.feature_dim = feature_dim
       
        # convolutional backbone
        self.stem = Res2NetStem(3, channels[0]) # output stride 4
        self.stages = nn.ModuleList()
        if task == 'cifar10':
            self.stages.append(self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True))
            self.stages.append(self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=False))
            self.stages.append(ConvModule(channels[2], channels[2], 1))
            self.feature_dim = 384
        else:
            self.stages.append(self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True))
            self.stages.append(self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True))
            self.stages.append(self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False))
            self.stages.append(ConvModule(channels[3], channels[3], 1))
            self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size):
        layers = []
        
        layers.append(block(in_channels, out_channels))
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels))
        
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    ConvModule(out_channels, out_channels, 1),
                    nn.AvgPool2d(2, stride=2)
                )
            )
        
        return nn.Sequential(*layers)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        stage_feats = []
        for stage in self.stages:
            x = stage(x)
            stage_feats.append(x)
        return stage_feats
        
##########
# Instantiation
##########
def osnet(cfg):
    # standard size (width x1.0)
    return OSNet(blocks=[OSBlock, OSBlock, OSBlock], 
                 layers=[2, 2, 2],
                 channels=[64, 256, 384, 512],
                 task=cfg.TASK)
