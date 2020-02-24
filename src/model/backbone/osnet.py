import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.module import ConvModule, Res2NetStem, InversedDepthwiseSeparable, SEModule, DropChannel

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
            DropChannel(),
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
            DropChannel(),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            DropChannel(),
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
            DropChannel(),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            DropChannel(),
            InversedDepthwiseSeparable(
                mid_channels, 
                mid_channels,
                3
            ),
            DropChannel(),
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

    def __init__(self, first_channel, stage_repeat, arch):
        super(OSNet, self).__init__()
        self.stem = Res2NetStem(3, first_channel) # output stride 4
        self.stages = nn.ModuleList()
        a_idx = 0
        for s_idx in range(len(stage_repeat)):
            stage = []
            for _ in range(stage_repeat[s_idx]):
                block, inc, ouc, s = arch[a_idx]
                a_idx += 1
                stage.append(self._make_layer(block, inc, ouc, s))
            self.stages.append(nn.Sequential(*stage))
        self.last_channel = ouc
        self._init_params()

    def _make_layer(self, block, inc, ouc, s):
        if isinstance(block, OSBlock):
            if s == 1:
                return nn.Sequential(block(inc, ouc))
            else:
                return nn.Sequential(
                    ConvModule(inc, ouc, 1),
                    nn.AvgPool2d(2, stride=2)
                )
        if isinstance(block, ConvModule):
            return nn.Sequential(block(inc, ouc, 1))
    
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
def osnet(task='reid', **kwargs):
    return OSNet(
        first_channel=64,
        stage_repeat=[3, 3, 4],
        arch=[
            (OSBlock, 64, 256, 1),
            (OSBlock, 256, 256, 1),
            (OSBlock, 256, 256, 2),
            (OSBlock, 256, 384, 1),
            (OSBlock, 384, 384, 1),
            (OSBlock, 384, 384, 2),
            (OSBlock, 384, 512, 1),
            (OSBlock, 512, 512, 1),
            (OSBlock, 512, 512, 1),
            (ConvModule, 512, 512, 1)
        ]
    )
